/**
 * Digit Morphing & Latent Interpolation Engine
 * Animates smooth geodesic transitions between digit classes and latent points.
 */

export class DigitMorpher {
  constructor(cvae, targetCanvas, onFrameCallback) {
    this.cvae = cvae;
    this.canvas = targetCanvas;
    this.ctx = this.canvas.getContext('2d');
    this.onFrame = onFrameCallback || (() => {});

    this.isPlaying = false;
    this.animationId = null;
    this.progress = 0;
    this.speed = 0.015;
    this.direction = 1;

    this.sourceDigit = 3;
    this.targetDigit = 8;
    this.sourceZ = [-1.5, -1.0];
    this.targetZ = [1.5, 1.0];

    this.canvas.width = 28;
    this.canvas.height = 28;
  }

  setEndpoints(sourceDigit, targetDigit, sourceZ = [-1.5, -1.0], targetZ = [1.5, 1.0]) {
    this.sourceDigit = sourceDigit;
    this.targetDigit = targetDigit;
    this.sourceZ = sourceZ;
    this.targetZ = targetZ;
    this.progress = 0;
    this.renderCurrentFrame();
  }

  setProgress(p) {
    this.progress = Math.max(0, Math.min(1, p));
    this.renderCurrentFrame();
  }

  setSpeed(s) {
    this.speed = s;
  }

  start() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.loop();
  }

  pause() {
    this.isPlaying = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  toggle() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.start();
    }
  }

  loop() {
    if (!this.isPlaying) return;

    this.progress += this.speed * this.direction;
    if (this.progress >= 1.0) {
      this.progress = 1.0;
      this.direction = -1;
    } else if (this.progress <= 0.0) {
      this.progress = 0.0;
      this.direction = 1;
    }

    this.renderCurrentFrame();
    this.animationId = requestAnimationFrame(() => this.loop());
  }

  renderCurrentFrame() {
    // Smooth S-curve easing
    const t = this.progress < 0.5 
      ? 2 * this.progress * this.progress 
      : 1 - Math.pow(-2 * this.progress + 2, 2) / 2;

    // Interpolate latent vector z
    const zInterp = [
      (1 - t) * this.sourceZ[0] + t * this.targetZ[0],
      (1 - t) * this.sourceZ[1] + t * this.targetZ[1]
    ];

    // Decode with smooth blend between source and target digit classes
    const rasterA = this.cvae.decode(zInterp, this.sourceDigit);
    const rasterB = this.cvae.decode(zInterp, this.targetDigit);
    const blended = new Float32Array(784);

    for (let i = 0; i < 784; i++) {
      blended[i] = (1 - t) * rasterA[i] + t * rasterB[i];
    }

    // Render to canvas
    const imgData = this.ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const val = Math.round(blended[i] * 255);
      imgData.data[i * 4] = val;
      imgData.data[i * 4 + 1] = Math.round(val * 0.95);
      imgData.data[i * 4 + 2] = Math.round(val * 1.1); // subtle cool tint
      imgData.data[i * 4 + 3] = 255;
    }
    this.ctx.putImageData(imgData, 0, 0);

    this.onFrame({
      progress: this.progress,
      easedProgress: t,
      z: zInterp,
      sourceDigit: this.sourceDigit,
      targetDigit: this.targetDigit,
      blendedRaster: blended
    });
  }
}
