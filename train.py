#!/usr/bin/env python
# coding: utf-8
from datetime import datetime

import torch
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_linear import TrackModel
from utils.dataset import JSONTrackDataset
from utils.transformations import position_displacement, add_noise

FRAMES_COUNT = 4
EPOCHS_COUNT = 200
LEARNING_RATE = 1e-4
META_LEN = 0
FORCE_CPU = False

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_name = '4f_' + datetime.now().strftime('%d_%H%M%S')

print('starting_train:', train_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if FORCE_CPU:
    device = torch.device('cpu')

print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

writer = SummaryWriter()

train_dataset = JSONTrackDataset('data_4/train_eq.json', transform=[position_displacement, add_noise])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=180, num_workers=7, pin_memory=FORCE_CPU,
                              drop_last=True)

val_dataset = JSONTrackDataset('data_4/val_eq_clear.json')
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=50, num_workers=7, pin_memory=FORCE_CPU,
                            drop_last=True)

test_dataset = JSONTrackDataset('data_4/test_eq_clear.json')
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=50, num_workers=7, pin_memory=FORCE_CPU,
                             drop_last=True)

model = TrackModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.00005)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5)

best_loss = 1000000

for epoch in range(EPOCHS_COUNT):
    # training
    criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.get_imbalance_weights()).to(device)

    loss_sum = 0
    ep_preds = []
    ep_labels = []
    model.train()
    for data_row in tqdm(train_dataloader, desc='Epoch {}/{}'.format(epoch, EPOCHS_COUNT - 1), ncols=80):
        data = data_row['data'].to(device)
        label = data_row['label'].to(device)

        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, label.argmax(dim=1))
        loss_sum += loss.item()

        np_label = label.argmax(dim=1).cpu().data.numpy()
        np_preds = y_pred.argmax(dim=1).cpu().data.numpy()

        for i in np_label:
            ep_labels.append(i)
        for i in np_preds:
            ep_preds.append(i)

        loss.backward()
        optimizer.step()
    scheduler.step(loss_sum / len(train_dataloader))
    writer.add_scalar('train_loss', loss_sum / len(train_dataloader), epoch)

    # validation
    criterion = torch.nn.CrossEntropyLoss(weight=val_dataset.get_imbalance_weights()).to(device)
    model.eval()
    loss_sum = 0
    ep_preds = []
    ep_labels = []
    for val_row in tqdm(val_dataloader, desc='Val', ncols=80):
        data = val_row['data'].to(device)
        label = val_row['label'].to(device)

        y_pred = model(data)

        loss = criterion(y_pred, label.argmax(dim=1))
        loss_sum += loss.item()

        np_label = label.argmax(dim=1).cpu().data.numpy()
        np_preds = y_pred.argmax(dim=1).cpu().data.numpy()

        for i in np_label:
            ep_labels.append(i)
        for i in np_preds:
            ep_preds.append(i)

    accuracy = metrics.balanced_accuracy_score(ep_labels, ep_preds)
    recall = metrics.recall_score(ep_labels, ep_preds)
    precision = metrics.precision_score(ep_labels, ep_preds)
    f1 = metrics.f1_score(ep_labels, ep_preds)
    roc_auc = metrics.roc_auc_score(ep_labels, ep_preds)

    r_loss = loss_sum / len(val_dataloader)

    if best_loss > r_loss:
        best_loss = r_loss
        torch.save(model.state_dict(), f'cache/cache_{train_name}_e{epoch}_{r_loss}.pth')

    print('')
    print(tabulate([
        ['Validation', r_loss, accuracy, recall, precision, f1, roc_auc]
    ], headers=['Stage', 'Loss', 'Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC'], tablefmt='orgtbl'))
    print('')

    writer.add_scalar('val_loss', loss_sum / len(val_dataloader), epoch)
    writer.add_scalar('val_acc', accuracy, epoch)
    writer.add_scalar('val_recall', recall, epoch)
    writer.add_scalar('val_prec', precision, epoch)
    writer.add_scalar('val_f1', f1, epoch)
    writer.add_scalar('val_roc_au', roc_auc, epoch)

criterion = torch.nn.CrossEntropyLoss(weight=test_dataset.get_imbalance_weights()).to(device)

loss_sum = 0
ep_preds = []
ep_labels = []
for data_row in tqdm(test_dataloader, desc='Test', ncols=80):
    data = data_row['data'].to(device)
    label = data_row['label'].to(device)

    optimizer.zero_grad()

    y_pred = model(data)
    loss = criterion(y_pred, label.argmax(dim=1))
    loss_sum += loss.item()

    np_label = label.argmax(dim=1).cpu().data.numpy()
    np_preds = y_pred.argmax(dim=1).cpu().data.numpy()

    for i in np_label:
        ep_labels.append(i)
    for i in np_preds:
        ep_preds.append(i)

accuracy = metrics.balanced_accuracy_score(ep_labels, ep_preds)
recall = metrics.recall_score(ep_labels, ep_preds)
precision = metrics.precision_score(ep_labels, ep_preds)
f1 = metrics.f1_score(ep_labels, ep_preds)
roc_auc = metrics.roc_auc_score(ep_labels, ep_preds)

print('')
print(tabulate([
    ['Test', loss_sum / len(train_dataloader), accuracy, recall, precision, f1, roc_auc]
], headers=['Stage', 'Loss', 'Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC'], tablefmt='orgtbl'))
print('')

writer.close()

torch.save(model, 'model.pt')
