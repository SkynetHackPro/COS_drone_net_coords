<template>
  <select v-model="selectedId">
    <option
      v-for="(item, key) in choices"
      :value="key"
    >
      {{ item.label }} - {{ key }}
    </option>
  </select>
</template>

<script>
  export default {
    name: "CSelect",
    props: ['choices', 'currentObject'],
    data() {
      return {
        selectedId: null
      }
    },
    watch: {
      selectedId(id) {
        this.$emit('input', this.choices[id]);
      }
    }
  }
</script>

<style scoped>

</style>
