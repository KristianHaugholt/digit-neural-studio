/**
 * Latent Space 2D Explorer Component
 * Renders an interactive 2D coordinate plane for continuous latent vector navigation (z1, z2).
 * Supports mouse/touch dragging, crosshair positioning, trajectory trails, and live decoding.
 */

export class LatentExplorer {
  constructor(canvasElement, onCoordinateChange) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');
    this.onCoordinateChange = onCoordinateChange || (() => {});

    this.zRange = 3.0; // [-3.0, +3.0]
    this.currentZ = [0.0, 0.0];
    this.isDragging = false;
    this.trail = [];
    this.maxTrail = 30;

    this.initCanvas();
    this.bindEvents();
    this.render();
  }

  initCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || 360;
    this.height = rect.height || 360;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  bindEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.onPointerDown(e));
    window.addEventListener('mousemove', (e) => this.onPointerMove(e));
    window.addEventListener('mouseup', () => this.onPointerUp());

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.onPointerDown(e.touches[0]);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this.onPointerMove(e.touches[0]);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.onPointerUp();
    }, { passive: false });

    window.addEventListener('resize', () => {
      this.initCanvas();
      this.render();
    });
  }

  posToZ(x, y) {
    const z1 = ((x / this.width) * 2 - 1) * this.zRange;
    const z2 = -((y / this.height) * 2 - 1) * this.zRange;
    return [
      Math.max(-this.zRange, Math.min(this.zRange, z1)),
      Math.max(-this.zRange, Math.min(this.zRange, z2))
    ];
  }

  zToPos(z1, z2) {
    const x = ((z1 / this.zRange + 1) / 2) * this.width;
    const y = ((-z2 / this.zRange + 1) / 2) * this.height;
    return { x, y };
  }

  onPointerDown(e) {
    this.isDragging = true;
    const rect = this.canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (this.width / rect.width);
    const py = (e.clientY - rect.top) * (this.height / rect.height);
    this.setCoordinatesFromPos(px, py);
  }

  onPointerMove(e) {
    if (!this.isDragging) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (this.width / rect.width);
    const py = (e.clientY - rect.top) * (this.height / rect.height);
    this.setCoordinatesFromPos(px, py);
  }

  onPointerUp() {
    this.isDragging = false;
  }

  setCoordinatesFromPos(px, py) {
    const [z1, z2] = this.posToZ(px, py);
    this.setZ(z1, z2);
  }

  setZ(z1, z2, emit = true) {
    this.currentZ = [
      Math.max(-this.zRange, Math.min(this.zRange, z1)),
      Math.max(-this.zRange, Math.min(this.zRange, z2))
    ];

    const pos = this.zToPos(this.currentZ[0], this.currentZ[1]);
    this.trail.push(pos);
    if (this.trail.length > this.maxTrail) {
      this.trail.shift();
    }

    this.render();
    if (emit) {
      this.onCoordinateChange(this.currentZ);
    }
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Background
    ctx.fillStyle = '#080c16';
    ctx.fillRect(0, 0, w, h);

    // Subtle Radial Glow at Center
    const radGlow = ctx.createRadialGradient(w/2, h/2, 5, w/2, h/2, w*0.6);
    radGlow.addColorStop(0, 'rgba(99, 102, 241, 0.15)');
    radGlow.addColorStop(1, 'rgba(8, 12, 22, 0)');
    ctx.fillStyle = radGlow;
    ctx.fillRect(0, 0, w, h);

    // Grid Lines
    ctx.lineWidth = 1;
    const steps = 6;
    for (let i = 0; i <= steps; i++) {
      const x = (i / steps) * w;
      const y = (i / steps) * h;

      ctx.strokeStyle = (i === steps / 2) ? 'rgba(99, 102, 241, 0.4)' : 'rgba(255, 255, 255, 0.05)';
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Concentric Standard Deviation Rings (1σ, 2σ, 3σ)
    for (let sigma = 1; sigma <= 3; sigma++) {
      const radius = (sigma / this.zRange) * (w / 2);
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.12)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Label
      ctx.fillStyle = 'rgba(6, 182, 212, 0.35)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`${sigma}σ`, w / 2 + radius + 4, h / 2 - 4);
    }

    // Trajectory Trail
    if (this.trail.length > 1) {
      ctx.lineWidth = 2;
      for (let i = 1; i < this.trail.length; i++) {
        const p0 = this.trail[i - 1];
        const p1 = this.trail[i];
        const alpha = i / this.trail.length;
        ctx.strokeStyle = `rgba(236, 72, 153, ${alpha * 0.7})`;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      }
    }

    // Current Target Crosshair
    const currentPos = this.zToPos(this.currentZ[0], this.currentZ[1]);

    // Outer Glow Ring
    ctx.beginPath();
    ctx.arc(currentPos.x, currentPos.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(99, 102, 241, 0.25)';
    ctx.fill();

    // Main Point
    ctx.beginPath();
    ctx.arc(currentPos.x, currentPos.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#06b6d4';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Coordinate HUD Text
    ctx.fillStyle = '#f8fafc';
    ctx.font = '11px "JetBrains Mono", monospace';
    const z1Str = this.currentZ[0] >= 0 ? `+${this.currentZ[0].toFixed(2)}` : this.currentZ[0].toFixed(2);
    const z2Str = this.currentZ[1] >= 0 ? `+${this.currentZ[1].toFixed(2)}` : this.currentZ[1].toFixed(2);
    ctx.fillText(`z: [${z1Str}, ${z2Str}]`, 12, h - 12);
  }
}
