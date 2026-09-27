# 🧠 DigitVerse AI Studio — Deep Neural Network Classifier & Conditional VAE Studio

An interactive, high-performance Deep Learning web application for **Hand-Drawn Digit Classification** and **Controllable Digit Generation** powered by Deep Convolutional Neural Networks (CNNs) and Conditional Variational Autoencoders (CVAEs).

---

## 🌟 Key Features

### 1. ✍️ Real-Time Drawing & CNN Classifier
- **High-DPI Drawing Canvas**: Smooth Bézier stroke interpolation, responsive eraser, stroke thickness controls, and invert/clear tools.
- **MNIST Center-of-Mass Preprocessing**: Automatically extracts bounding boxes, scales to $20 \times 20$, and centers mass inside standard $28 \times 28$ tensors.
- **Live Inference (< 2 ms)**: Real-time top-1 prediction with confidence percentages and runner-up entropy margin.
- **Feature Map Visualizer**: Visualizes 16 intermediate Conv2D filter activations (edge detectors, loop filters, line angles, and arc filters).
- **One-Click Presets**: Test canonical MNIST digits (0 to 9) instantly.

### 2. 🎨 Conditional Variational Autoencoder (CVAE) Generative Studio
- **2D Latent Space Explorer**: Continuously navigate the 2D latent space $\mathbf{z} \in [-3, 3]^2$ with an interactive drag-and-drop plane to watch digits synthesize in real-time.
- **10×10 Manifold Grid**: Visualizes the entire 2D manifold of handwriting variations across the latent coordinate space for any selected digit class ($y \in \{0..9\}$).
- **Geodesic Digit Morpher**: Animates smooth latent trajectory morphing between two digit classes (e.g. $3 \to 8$) with step scrubbers and intermediate filmstrips.
- **Zero-Shot Style Transfer (Disentanglement)**: Encodes your drawn handwriting into latent style parameters $\mathbf{z}^* = \mu(\mathbf{x}, y)$ and decodes all 10 digits in your exact handwriting slant, thickness, and curvature style.
- **Batch Generator**: Synthesizes 32 varied digits with random latent vectors and downloadable high-res PNGs.

### 3. 🧠 Neural Network Architecture & Mathematical Theory
- Complete mathematical breakdown of Forward Propagation, Softmax Cross-Entropy, Reparameterization Trick, and Evidence Lower Bound (ELBO) Loss ($\text{Reconstruction} + \text{KL Divergence}$).
- Visual interactive layer diagrams for both discriminative and generative models.

### 4. 🚀 In-Browser Live Trainer & Simulator
- Interactive WebGL backpropagation simulator with customizable learning rates and epochs.
- Real-time Loss and Accuracy curves plotted directly on canvas.

---

## 📐 Mathematical Formulation

### 1. CNN Classifier
Given an input digit image $\mathbf{x} \in \mathbb{R}^{28 \times 28}$, the network computes class logits $z_k$ and outputs class posterior probabilities via Softmax:

$$P(y = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=0}^{9} \exp(z_j)}$$

Trained using Categorical Cross-Entropy:
$$\mathcal{L}_{\text{CE}} = -\sum_{k=0}^{9} y_k \log \hat{y}_k$$

### 2. Conditional Variational Autoencoder (CVAE)
In a standard VAE, latent vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ entangles digit identity and stroke style. In a **Conditional VAE**, we supply the one-hot label $y$ to both the **Encoder** and the **Decoder**, cleanly separating **Digit Class ($y$)** from **Style ($\mathbf{z}$)**.

- **Encoder**: $q_\phi(\mathbf{z} \mid \mathbf{x}, y) \to \boldsymbol{\mu}, \log\boldsymbol{\sigma}^2$
- **Reparameterization Trick**: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- **Decoder**: $p_\theta(\mathbf{x} \mid \mathbf{z}, y) \to \hat{\mathbf{x}} \in [0, 1]^{784}$

#### Objective Function: Evidence Lower Bound (ELBO)
$$\mathcal{L}_{\text{CVAE}}(\theta, \phi; \mathbf{x}, y) = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x},y)}[\log p_\theta(\mathbf{x}\mid\mathbf{z},y)] - D_{\text{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x},y) \parallel p(\mathbf{z}))$$

Where:
- **Binary Cross-Entropy Reconstruction Loss**:
  $$-\sum_{i=1}^{784} \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$
- **KL Divergence Regularizer (Analytical Solution)**:
  $$D_{\text{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x},y) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I})) = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

---

## 🚀 Running Locally

You can launch the web application with any local HTTP server:

```powershell
# Using Python
python -m http.server 8080

# Then open in your browser:
# http://localhost:8080
```

---

## 📂 Project Structure

```
digit-neural-studio/
├── index.html                  # Main UI & Tabbed Interface
├── css/
│   └── style.css               # Glassmorphic Dark Design System
├── js/
│   ├── app.js                  # Main Application Orchestrator
│   ├── components/
│   │   ├── drawing-canvas.js   # Drawing Canvas & MNIST Normalizer
│   │   ├── latent-explorer.js  # 2D Latent Space Plane Navigator
│   │   ├── morpher.js          # Digit Morphing & Trajectory Engine
│   │   └── activation-view.js  # Conv2D Feature Maps Visualizer
│   ├── models/
│   │   ├── classifier.js       # CNN Classifier & Activation Extractor
│   │   ├── cvae.js             # Conditional VAE Engine & Disentanglement
│   │   ├── weights.js          # Pretrained Weights (Zero Network Lag)
│   │   └── model_weights.json  # Raw Weights Export
│   └── presets/
│       └── samples.js          # Canonical MNIST Digit Samples
├── scripts/
│   └── train_and_export.py     # NumPy & Scikit-Learn Training Pipeline
└── README.md                   # Documentation & Math Reference
```
