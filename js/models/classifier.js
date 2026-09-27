/**
 * Deep Neural Network Classifier Engine for Hand-Drawn Digits
 * Supports Convolutional & Deep Dense forward passes, Softmax probability distribution,
 * and intermediate feature map / activation extraction.
 */

export class DigitClassifier {
  constructor() {
    this.isLoaded = false;
    this.weights = null;
    this.tfModel = null;
    this.conv1Kernels = null;
    this.initDefaultFilters();
  }

  initDefaultFilters() {
    // 16 specialized 3x3 convolution kernels (Horizontal edges, vertical edges, diagonals, loops, corners)
    this.conv1Kernels = [
      // 0: Horizontal Edge Top
      [-1, -1, -1,  2,  2,  2, -1, -1, -1],
      // 1: Horizontal Edge Bottom
      [ 1,  1,  1, -2, -2, -2,  1,  1,  1],
      // 2: Vertical Edge Left
      [-1,  2, -1, -1,  2, -1, -1,  2, -1],
      // 3: Vertical Edge Right
      [ 1, -2,  1,  1, -2,  1,  1, -2,  1],
      // 4: Diagonal Top-Left to Bottom-Right
      [ 2, -1, -1, -1,  2, -1, -1, -1,  2],
      // 5: Diagonal Top-Right to Bottom-Left
      [-1, -1,  2, -1,  2, -1,  2, -1, -1],
      // 6: Center Point / Peak
      [-1, -1, -1, -1,  8, -1, -1, -1, -1],
      // 7: Laplacian Edge Detector
      [ 0,  1,  0,  1, -4,  1,  0,  1,  0],
      // 8: Box Blur Smoother
      [ 1,  1,  1,  1,  1,  1,  1,  1,  1].map(v => v / 9),
      // 9: Ridge detector
      [-1,  2, -1,  2, -4,  2, -1,  2, -1],
      // 10: High pass
      [-1, -1, -1, -1,  9, -1, -1, -1, -1].map(v => v / 2),
      // 11: Arc Top
      [-2,  2, -2,  2,  0,  2, -2, -2, -2],
      // 12: Arc Bottom
      [-2, -2, -2,  2,  0,  2, -2,  2, -2],
      // 13: Loop detector
      [ 1,  2,  1,  2, -4,  2,  1,  2,  1],
      // 14: Diagonal Thick
      [ 0,  2,  2,  2,  2,  2,  2,  2,  0].map(v => v / 8),
      // 15: Cross / Intersection
      [ 0,  2,  0,  2,  2,  2,  0,  2,  0]
    ];
  }

  loadWeights(weightsData) {
    if (weightsData && weightsData.classifier) {
      this.weights = weightsData.classifier;
      this.isLoaded = true;
      console.log("Classifier weights successfully loaded into memory.");
    }
  }

  /**
   * Evaluates Conv2D Layer 1 on 28x28 input tensor
   * Returns array of 16 feature maps (each 28x28 or 14x14)
   */
  computeLayerActivations(inputTensor) {
    const activations = [];
    const width = 28;
    const height = 28;

    for (let k = 0; k < this.conv1Kernels.length; k++) {
      const kernel = this.conv1Kernels[k];
      const featureMap = new Float32Array(width * height);

      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let sum = 0;
          let kidx = 0;
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const pixel = inputTensor[(y + ky) * width + (x + kx)];
              sum += pixel * kernel[kidx++];
            }
          }
          // ReLU Activation
          featureMap[y * width + x] = Math.max(0, sum);
        }
      }
      activations.push(featureMap);
    }
    return activations;
  }

  /**
   * Runs forward pass through Deep Neural Network Classifier
   * Returns: { probabilities: Array(10), topDigit: Number, confidence: Number, activations: Array }
   */
  predict(inputTensor) {
    if (!inputTensor || inputTensor.every(v => v === 0)) {
      return {
        probabilities: new Array(10).fill(0.1),
        topDigit: null,
        confidence: 0,
        activations: this.computeLayerActivations(new Float32Array(784))
      };
    }

    const activations = this.computeLayerActivations(inputTensor);

    // If trained weights are available, run full MLP forward pass
    if (this.weights && this.weights.coefs && this.weights.coefs.length >= 2) {
      let currentActivation = inputTensor;

      for (let l = 0; l < this.weights.coefs.length; l++) {
        const W = this.weights.coefs[l]; // [in_dim][out_dim]
        const b = this.weights.intercepts[l]; // [out_dim]
        const inDim = currentActivation.length;
        const outDim = b.length;
        const nextActivation = new Float32Array(outDim);

        for (let j = 0; j < outDim; j++) {
          let sum = b[j];
          for (let i = 0; i < inDim; i++) {
            sum += currentActivation[i] * W[i][j];
          }
          // If hidden layer: ReLU, if final layer: linear logits
          if (l < this.weights.coefs.length - 1) {
            nextActivation[j] = Math.max(0, sum);
          } else {
            nextActivation[j] = sum;
          }
        }
        currentActivation = nextActivation;
      }

      // Compute Softmax
      const logits = currentActivation;
      const maxLogit = Math.max(...logits);
      let expSum = 0;
      const exps = new Float32Array(10);
      for (let i = 0; i < 10; i++) {
        exps[i] = Math.exp(logits[i] - maxLogit);
        expSum += exps[i];
      }

      const probs = [];
      let topDigit = 0;
      let maxProb = 0;

      for (let i = 0; i < 10; i++) {
        const p = exps[i] / expSum;
        probs.push(p);
        if (p > maxProb) {
          maxProb = p;
          topDigit = i;
        }
      }

      return {
        probabilities: probs,
        topDigit,
        confidence: maxProb,
        activations
      };
    }

    // Fallback template-matching classifier if weights not yet loaded
    return this.fallbackPredict(inputTensor, activations);
  }

  fallbackPredict(inputTensor, activations) {
    const scores = new Float32Array(10);
    // Simple structural heuristic metrics
    let topHalf = 0, botHalf = 0, leftHalf = 0, rightHalf = 0, centerCore = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = inputTensor[y * 28 + x];
        if (y < 14) topHalf += v; else botHalf += v;
        if (x < 14) leftHalf += v; else rightHalf += v;
        if (x >= 9 && x <= 18 && y >= 9 && y <= 18) centerCore += v;
      }
    }

    const total = topHalf + botHalf + 1e-5;
    const topRatio = topHalf / total;
    const leftRatio = leftHalf / (leftHalf + rightHalf + 1e-5);
    const centerRatio = centerCore / total;

    // Heuristics
    scores[0] = (1 - centerRatio) * 2.0;
    scores[1] = (1 - Math.abs(leftRatio - 0.5)) * (1 - centerRatio * 0.5) * 1.5;
    scores[8] = centerRatio * 2.5 + 0.5;
    scores[7] = topRatio * 2.2;
    scores[6] = (1 - topRatio) * 1.8 + leftRatio * 1.2;
    scores[9] = topRatio * 1.8 + (1 - leftRatio) * 1.2;
    scores[3] = (1 - leftRatio) * 2.0;
    scores[4] = centerRatio * 1.5 + (1 - topRatio) * 1.2;
    scores[2] = topRatio * 1.2 + (1 - leftRatio) * 1.2;
    scores[5] = leftRatio * 1.3 + (1 - topRatio) * 1.3;

    // Softmax
    const maxS = Math.max(...scores);
    let sumExp = 0;
    const exps = scores.map(s => { const e = Math.exp(s - maxS); sumExp += e; return e; });
    const probs = Array.from(exps).map(e => e / sumExp);
    const topDigit = probs.indexOf(Math.max(...probs));

    return {
      probabilities: probs,
      topDigit,
      confidence: probs[topDigit],
      activations
    };
  }
}
