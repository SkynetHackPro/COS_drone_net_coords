<template>
  <div class="app">
    <h3>Выберите тест объект: <span class="help">(drone - дрон, bg - другой объект)</span></h3>
    <CSelect
      :choices="testObjectsList"
      v-model="currentObject"
    />
    <h3>Вероятность дрона: {{ probability }}%</h3>
    <h3>Визуализация трека:</h3>
    <ObjectPreview
      :object="currentObjectData"
    />
  </div>
</template>

<script>
  import {evalModel, getTestObjectsList} from "./http.js";
  import CSelect from "./components/CSelect.vue";
  import ObjectPreview from "./components/ObjectPreview.vue";

  export default {
    name: "App",
    components: {ObjectPreview, CSelect},
    data() {
      return {
        testObjectsList: [],
        currentObject: null,
        probability: 0
      }
    },
    computed: {
      currentObjectData() {
        if (!this.currentObject) {
          return null;
        }
        return this.currentObject.data
      }
    },
    watch: {
      async currentObject() {
        const percentage = (await evalModel(this.currentObject.data)).drone;
        this.probability = percentage * 100;
      }
    },
    async mounted() {
      this.testObjectsList = await getTestObjectsList()
    }
  }
</script>

<style>
  .help {
    color: #6e6e6e;
  }
</style>
