import axios from 'axios';

const getTestObjectsList = async () => {
  const response = await axios.get('/api/dataset');
  return response.data;
};

const evalModel = async (params) => {
  const response = await axios.post('/api/predict', {
    data: params
  });
  return  response.data;
};

export {
  getTestObjectsList,
  evalModel
}
