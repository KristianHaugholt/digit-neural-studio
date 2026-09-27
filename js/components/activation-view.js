/**
 * Layer Activations and Saliency Visualizer
 * Renders the intermediate Conv2D feature maps showing edges, loops, and stroke detectors.
 */

export class ActivationViewer {
  constructor(containerElement) {
    this.container = containerElement;
    this.filterCanvases = [];
    this.initFilterCanvases();
  }

  initFilterCanvases() {
    this.container.innerHTML = '';
    this.filterCanvases = [];

    // Layer 1 has 16 feature maps (28x28)
    for (let i = 0; i < 16; i++) {
      const chip = document.createElement('div');
      chip.className = 'feature-chip';
      chip.title = `Conv2D Filter #${i + 1}`;

      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      chip.appendChild(canvas);
      this.container.appendChild(chip);

      this.filterCanvases.push(canvas);
    }
  }

  updateActivations(activations) {
    if (!activations || activations.length === 0) return;

    for (let k = 0; k < Math.min(activations.length, this.filterCanvases.length); k++) {
      const act = activations[k];
      const canvas = this.filterCanvases[k];
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(28, 28);

      // Find max activation for normalization
      let maxVal = 0.001;
      for (let i = 0; i < act.length; i++) {
        if (act[i] > maxVal) maxVal = act[i];
      }

      for (let i = 0; i < 784; i++) {
        const norm = Math.min(1.0, act[i] / maxVal);
        // Cool neon cyan-indigo heat color mapping
        const r = Math.round(norm * 99);
        const g = Math.round(norm * 182);
        const b = Math.round(norm * 241);

        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = norm > 0.05 ? 255 : 30;
      }
      ctx.putImageData(imgData, 0, 0);
    }
  }
}
