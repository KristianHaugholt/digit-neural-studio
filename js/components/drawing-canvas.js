/**
 * Drawing Canvas Component with MNIST standard Preprocessing Pipeline
 * Handles high-DPI input, smooth continuous strokes, responsive eraser,
 * center-of-mass centering, bounding box extraction, and live 28x28 normalized tensor production.
 */

export class DrawingCanvas {
  constructor(canvasElement, preprocCanvasElement, onUpdateCallback) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.preprocCanvas = preprocCanvasElement;
    this.preprocCtx = this.preprocCanvas.getContext('2d', { willReadFrequently: true });
    this.onUpdate = onUpdateCallback || (() => {});

    this.isDrawing = false;
    this.brushSize = 26;
    this.brushMode = 'draw'; // 'draw' or 'erase'
    this.currentStrokeMode = 'draw'; // active stroke mode
    this.isInverted = false;
    this.lastPoint = null;
    this.points = [];
    this.history = [];
    this.maxHistory = 30;

    this.initCanvas();
    this.initCursorRing();
    this.bindEvents();
    this.clear();
  }

  initCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.size = Math.min(rect.width || 360, 360);
    this.canvas.width = this.size * dpr;
    this.canvas.height = this.size * dpr;
    this.ctx.scale(dpr, dpr);

    // Preprocessing canvas is always 28x28
    this.preprocCanvas.width = 28;
    this.preprocCanvas.height = 28;

    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  initCursorRing() {
    if (!this.canvas.parentElement) return;
    let cursor = this.canvas.parentElement.querySelector('.brush-cursor');
    if (!cursor) {
      cursor = document.createElement('div');
      cursor.className = 'brush-cursor';
      this.canvas.parentElement.appendChild(cursor);
    }
    this.brushCursor = cursor;
  }

  bindEvents() {
    // Prevent context menu on drawing canvas to enable right-click erasing
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Mouse Events
    this.canvas.addEventListener('mousedown', (e) => this.startStroke(e));
    window.addEventListener('mousemove', (e) => this.drawStroke(e));
    window.addEventListener('mouseup', () => this.endStroke());

    // Cursor hover ring events
    this.canvas.addEventListener('mouseenter', (e) => {
      if (this.brushCursor) {
        this.updateCursorRing(e);
        this.brushCursor.style.display = 'block';
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      if (this.brushCursor && !this.isDrawing) {
        this.brushCursor.style.display = 'none';
      }
    });

    // Touch Events for Mobile / Tablet / Stylus
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.startStroke(touch);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.drawStroke(touch);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.endStroke();
      if (this.brushCursor) {
        this.brushCursor.style.display = 'none';
      }
    }, { passive: false });

    // Resize handler
    window.addEventListener('resize', () => {
      const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      this.initCanvas();
      this.ctx.putImageData(imgData, 0, 0);
    });
  }

  getCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (this.size / (rect.width || 1)),
      y: (e.clientY - rect.top) * (this.size / (rect.height || 1))
    };
  }

  updateCursorRing(e) {
    if (!this.brushCursor) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
      if (!this.isDrawing) {
        this.brushCursor.style.display = 'none';
        return;
      }
    }

    this.brushCursor.style.display = 'block';
    this.brushCursor.style.left = `${x}px`;
    this.brushCursor.style.top = `${y}px`;
    this.brushCursor.style.width = `${this.brushSize}px`;
    this.brushCursor.style.height = `${this.brushSize}px`;
  }

  getStrokeColor(mode = this.currentStrokeMode) {
    if (mode === 'draw') {
      return this.isInverted ? '#000000' : '#ffffff';
    } else {
      return this.isInverted ? '#ffffff' : '#000000';
    }
  }

  getBackgroundColor() {
    return this.isInverted ? '#ffffff' : '#000000';
  }

  startStroke(e) {
    // Right click (button === 2) triggers eraser on the fly
    if (e.button === 2 || e.which === 3) {
      this.currentStrokeMode = 'erase';
      if (this.canvas.parentElement) {
        this.canvas.parentElement.classList.add('eraser-mode');
      }
    } else {
      this.currentStrokeMode = this.brushMode;
    }

    this.saveState();
    this.isDrawing = true;
    const pt = this.getCoords(e);
    this.lastPoint = pt;
    this.points = [pt];

    const strokeColor = this.getStrokeColor();
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.lineWidth = this.brushSize;
    this.ctx.strokeStyle = strokeColor;
    this.ctx.fillStyle = strokeColor;

    this.ctx.beginPath();
    this.ctx.arc(pt.x, pt.y, this.brushSize / 2, 0, Math.PI * 2);
    this.ctx.fill();

    if (this.brushCursor) {
      this.updateCursorRing(e);
    }

    this.processAndNotify();
  }

  drawStroke(e) {
    if (this.brushCursor) {
      this.updateCursorRing(e);
    }

    if (!this.isDrawing) return;

    // Detect right mouse button held
    if (e.buttons === 2) {
      this.currentStrokeMode = 'erase';
    }

    const pt = this.getCoords(e);
    if (!this.lastPoint) {
      this.lastPoint = pt;
    }

    const strokeColor = this.getStrokeColor();
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.lineWidth = this.brushSize;
    this.ctx.strokeStyle = strokeColor;
    this.ctx.fillStyle = strokeColor;

    // Draw solid continuous connecting segment
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    this.ctx.lineTo(pt.x, pt.y);
    this.ctx.stroke();

    // Round join cap at end
    this.ctx.beginPath();
    this.ctx.arc(pt.x, pt.y, this.brushSize / 2, 0, Math.PI * 2);
    this.ctx.fill();

    this.lastPoint = pt;
    this.points.push(pt);

    this.processAndNotify();
  }

  endStroke() {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    this.lastPoint = null;
    this.points = [];
    this.currentStrokeMode = this.brushMode;
    if (this.canvas.parentElement) {
      this.canvas.parentElement.classList.toggle('eraser-mode', this.brushMode === 'erase');
    }
    this.processAndNotify();
  }

  saveState() {
    const data = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    this.history.push({
      data: data,
      isInverted: this.isInverted
    });
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
  }

  undo() {
    if (this.history.length > 0) {
      const state = this.history.pop();
      this.isInverted = state.isInverted;
      this.ctx.putImageData(state.data, 0, 0);
      this.processAndNotify();
    } else {
      this.clear();
    }
  }

  clear() {
    this.saveState();
    const bgColor = this.getBackgroundColor();
    this.ctx.fillStyle = bgColor;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.preprocCtx.fillStyle = '#000000';
    this.preprocCtx.fillRect(0, 0, 28, 28);
    this.processAndNotify();
  }

  invert() {
    this.saveState();
    this.isInverted = !this.isInverted;
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255 - d[i];
      d[i + 1] = 255 - d[i + 1];
      d[i + 2] = 255 - d[i + 2];
    }
    this.ctx.putImageData(imgData, 0, 0);
    this.processAndNotify();
  }

  setBrushSize(size) {
    this.brushSize = size;
    if (this.brushCursor) {
      this.brushCursor.style.width = `${size}px`;
      this.brushCursor.style.height = `${size}px`;
    }
  }

  setBrushMode(mode) {
    this.brushMode = mode;
    this.currentStrokeMode = mode;
    if (this.canvas.parentElement) {
      this.canvas.parentElement.classList.toggle('eraser-mode', mode === 'erase');
    }
  }

  loadDigitRaster(raster28) {
    this.saveState();
    this.isInverted = false;
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext('2d');
    const imgData = tCtx.createImageData(28, 28);

    for (let i = 0; i < 784; i++) {
      const val = Math.round(raster28[i] * 255);
      imgData.data[i * 4] = val;
      imgData.data[i * 4 + 1] = val;
      imgData.data[i * 4 + 2] = val;
      imgData.data[i * 4 + 3] = 255;
    }
    tCtx.putImageData(imgData, 0, 0);

    // Scale up to drawing canvas
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(tempCanvas, 0, 0, this.size, this.size);
    this.processAndNotify();
  }

  preprocessMNIST() {
    const raw = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const d = raw.data;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const isInverted = this.isInverted;

    // 1. Find Precise Bounding Box of active pixels
    let minX = w, maxX = 0, minY = h, maxY = 0;
    let hasPixels = false;
    let activePixelCount = 0;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        let brightness = (d[idx] + d[idx + 1] + d[idx + 2]) / 3;
        if (isInverted) {
          brightness = 255 - brightness;
        }
        if (brightness > 20) {
          hasPixels = true;
          activePixelCount++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    const outputTensor = new Float32Array(28 * 28);
    this.preprocCtx.fillStyle = '#000000';
    this.preprocCtx.fillRect(0, 0, 28, 28);

    if (!hasPixels || activePixelCount < 8) {
      return { tensor: outputTensor, isEmpty: true };
    }

    // Add proportional margin around bounding box (10% on each side)
    const rawW = maxX - minX + 1;
    const rawH = maxY - minY + 1;
    const marginX = Math.max(2, Math.round(rawW * 0.08));
    const marginY = Math.max(2, Math.round(rawH * 0.08));

    minX = Math.max(0, minX - marginX);
    minY = Math.max(0, minY - marginY);
    maxX = Math.min(w - 1, maxX + marginX);
    maxY = Math.min(h - 1, maxY + marginY);

    const bbW = maxX - minX + 1;
    const bbH = maxY - minY + 1;

    // 2. Scale bounding box to fit inside 20x20 box preserving aspect ratio
    const maxDim = Math.max(bbW, bbH);
    const scale = 20.0 / maxDim;
    const targetW = Math.max(1, Math.min(20, Math.round(bbW * scale)));
    const targetH = Math.max(1, Math.min(20, Math.round(bbH * scale)));

    // Render scaled digit onto temporary canvas with high-quality smoothing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetW;
    tempCanvas.height = targetH;
    const tCtx = tempCanvas.getContext('2d');
    tCtx.imageSmoothingEnabled = true;
    tCtx.imageSmoothingQuality = 'high';
    tCtx.drawImage(this.canvas, minX, minY, bbW, bbH, 0, 0, targetW, targetH);

    const scaledData = tCtx.getImageData(0, 0, targetW, targetH);
    const sd = scaledData.data;

    // 3. Extract grayscale patch and calculate Center of Mass
    const patch = new Float32Array(targetW * targetH);
    let maxVal = 0;
    let sumIntensity = 0;
    let sumX = 0;
    let sumY = 0;

    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const idx = (y * targetW + x) * 4;
        let val = (sd[idx] + sd[idx + 1] + sd[idx + 2]) / (3 * 255);
        if (isInverted) {
          val = 1.0 - val;
        }
        // Suppress low-level background noise
        if (val < 0.08) {
          val = 0;
        }
        patch[y * targetW + x] = val;
        if (val > maxVal) maxVal = val;
        if (val > 0.05) {
          sumIntensity += val;
          sumX += x * val;
          sumY += y * val;
        }
      }
    }

    if (sumIntensity <= 0 || maxVal <= 0) {
      return { tensor: outputTensor, isEmpty: true };
    }

    // Normalize intensity so stroke peaks are solid (~1.0)
    const normFactor = 1.0 / maxVal;
    for (let i = 0; i < patch.length; i++) {
      patch[i] = Math.min(1.0, patch[i] * normFactor);
    }

    // Calculate sub-pixel Center of Mass
    const cx = sumX / sumIntensity;
    const cy = sumY / sumIntensity;

    // Shift Center of Mass directly to center of 28x28 (which is (13.5, 13.5))
    const dx = 13.5 - cx;
    const dy = 13.5 - cy;

    // 4. Sub-pixel Bilinear Splatting onto 28x28 Tensor
    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const val = patch[y * targetW + x];
        if (val <= 0) continue;

        const px = x + dx;
        const py = y + dy;
        const x0 = Math.floor(px);
        const y0 = Math.floor(py);
        const fx = px - x0;
        const fy = py - y0;

        if (x0 >= 0 && x0 < 28 && y0 >= 0 && y0 < 28) {
          outputTensor[y0 * 28 + x0] += val * (1 - fx) * (1 - fy);
        }
        if (x0 + 1 >= 0 && x0 + 1 < 28 && y0 >= 0 && y0 < 28) {
          outputTensor[y0 * 28 + (x0 + 1)] += val * fx * (1 - fy);
        }
        if (x0 >= 0 && x0 < 28 && y0 + 1 >= 0 && y0 + 1 < 28) {
          outputTensor[(y0 + 1) * 28 + x0] += val * (1 - fx) * fy;
        }
        if (x0 + 1 >= 0 && x0 + 1 < 28 && y0 + 1 >= 0 && y0 + 1 < 28) {
          outputTensor[(y0 + 1) * 28 + (x0 + 1)] += val * fx * fy;
        }
      }
    }

    // 5. Stroke Thickness & Contrast Calibration (Gaussian stroke boosting)
    let peak28 = 0;
    for (let i = 0; i < 784; i++) {
      if (outputTensor[i] > peak28) peak28 = outputTensor[i];
    }

    const boost = peak28 > 0 ? 1.0 / peak28 : 1.0;
    for (let i = 0; i < 784; i++) {
      let v = outputTensor[i] * boost;
      // Soft gamma curve to thicken thin mouse strokes to match natural MNIST line width
      v = Math.pow(Math.max(0, Math.min(1, v)), 0.85);
      outputTensor[i] = Math.max(0, Math.min(1, v));
    }

    // 6. Render the 28x28 normalized tensor to thumbnail preview canvas
    const imgData = this.preprocCtx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const p = Math.round(outputTensor[i] * 255);
      imgData.data[i * 4] = p;
      imgData.data[i * 4 + 1] = p;
      imgData.data[i * 4 + 2] = p;
      imgData.data[i * 4 + 3] = 255;
    }
    this.preprocCtx.putImageData(imgData, 0, 0);

    return { tensor: outputTensor, isEmpty: false };
  }

  processAndNotify() {
    const result = this.preprocessMNIST();
    this.onUpdate(result);
  }
}
