<template>
  <div class="object-preview">
    <div class="track-notations">
      <div class="track-notations__item" v-for="(color, index) in COLORMAP">
        <div class="track-notation__color" :style="{background: color}"/>
        - кадр № {{ index + 1 }}
      </div>
    </div>
    <canvas
      ref="canvas"
      class="object-preview__screen"
      :width="viewWidth"
      :height="viewHeight"
    />
  </div>
</template>


<script>
  const COLORMAP = [
    '#ff0000',
    '#00ff00',
    '#0091ff',
    '#000000',
  ];
  export default {
    name: "ObjectPreview",
    props: ['object'],
    data() {
      return {
        ctx: null,
        vidWidth: 1920,
        vidHeight: 1440,
        scale: 0.7,
        COLORMAP
      }
    },
    computed: {
      viewWidth() {
        return this.vidWidth * this.scale;
      },
      viewHeight() {
        return this.vidHeight * this.scale;
      }
    },
    mounted() {
      const canvas = this.$refs.canvas;
      this.ctx = canvas.getContext('2d');
    },
    watch: {
      object() {
        if (this.object) {
          this.clear();
          this.object.forEach((i, count) => {
            this.drawFrame(i, count);
          });
          this.drawTrack(this.object);
        }
      }
    },
    methods: {
      clear() {
        this.ctx.fillStyle = "#fff";
        this.ctx.fillRect(0, 0, this.viewWidth, this.viewHeight);
      },
      scaleParams(params) {
        return {
          x: params.x * this.scale,
          y: params.y * this.scale,
          w: params.w * this.scale,
          h: params.h * this.scale
        }
      },
      drawTrack(items) {
        this.ctx.beginPath();
        const normalizedFirst = this.scaleParams(items[0]);
        this.ctx.moveTo(normalizedFirst.x, normalizedFirst.y);

        items.forEach((i) => {
          const normalized = this.scaleParams(i);
          this.ctx.lineTo(normalized.x, normalized.y);
        });

        this.ctx.strokeStyle = '#ff0000';
        this.ctx.stroke();
      },
      drawFrame(rawParams, colorId) {
        const color = COLORMAP[colorId];
        const params = this.scaleParams(rawParams);
        this.ctx.beginPath();
        const xDisp = params.x - params.w / 2;
        const yDisp = params.y - params.h / 2;
        this.ctx.rect(xDisp, yDisp, params.w, params.h);
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = color;
        this.ctx.stroke();
      }
    }
  }
</script>

<style>
  .object-preview__screen {
    border: solid 1px #c4c4c4;
  }

  .track-notations {
    display: flex;
    flex-direction: row;
  }

  .track-notations__item {
    display: flex;
    flex-direction: row;
    align-items: center;
    margin-right: 30px;
  }

  .track-notation__color {
    margin-right: 3px;
    width: 10px;
    height: 10px;
  }
</style>
