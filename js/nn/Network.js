/**
 * Network.js — a fully-connected neural network written from scratch.
 *
 * No TensorFlow, no PyTorch, no ML libraries: just typed arrays and calculus.
 *
 *   • Xavier / He weight initialisation
 *   • tanh · ReLU · leaky ReLU · sigmoid · sine activations
 *   • softmax output + cross-entropy loss
 *   • backpropagation (hand-derived gradients, verified by numerical gradient checking)
 *   • mini-batch SGD with momentum, or Adam
 *   • L2 weight decay
 *
 * Runs unchanged in the browser and in Node (see tests/network.test.mjs).
 */

/** Small, fast, seedable PRNG (mulberry32), so experiments are reproducible. */
export function createRng(seed = 1) {
  let s = seed >>> 0;
  return function rng() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box–Muller. */
export function randn(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/**
 * Activation functions. `d(y, z)` returns dActivation/dz, given the output y
 * and the pre-activation z (whichever is cheaper to use).
 */
export const ACTIVATIONS = {
  tanh: { f: (z) => Math.tanh(z), d: (y) => 1 - y * y, init: 'xavier' },
  relu: { f: (z) => (z > 0 ? z : 0), d: (y, z) => (z > 0 ? 1 : 0), init: 'he' },
  leaky: { f: (z) => (z > 0 ? z : 0.01 * z), d: (y, z) => (z > 0 ? 1 : 0.01), init: 'he' },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (y) => y * (1 - y), init: 'xavier' },
  sin: { f: (z) => Math.sin(z), d: (y, z) => Math.cos(z), init: 'xavier' },
};

export class Network {
  /**
   * @param {object} opts
   * @param {number[]} opts.layers     sizes, e.g. [2, 12, 12, 3] = 2 inputs, two hidden layers, 3 classes
   * @param {string}   opts.activation key of ACTIVATIONS, used by every hidden layer
   * @param {number}   opts.seed       RNG seed for weight init and shuffling
   */
  constructor({ layers = [2, 8, 8, 2], activation = 'tanh', seed = 1 } = {}) {
    if (!ACTIVATIONS[activation]) throw new Error(`Unknown activation "${activation}"`);
    this.sizes = layers.slice();
    this.L = layers.length - 1; // number of weight matrices
    this.activation = activation;
    this.act = ACTIVATIONS[activation];
    this.rng = createRng(seed);
    this.t = 0; // optimiser step counter (Adam bias correction)

    const alloc = (n) => new Float64Array(n);
    this.W = []; this.b = [];   // parameters
    this.gW = []; this.gb = []; // gradient accumulators
    this.mW = []; this.mb = []; // 1st moment (Adam) / velocity (momentum)
    this.vW = []; this.vb = []; // 2nd moment (Adam)

    for (let l = 0; l < this.L; l++) {
      const nIn = layers[l], nOut = layers[l + 1];
      const W = alloc(nIn * nOut);
      // He init for ReLU-family, Xavier/Glorot for saturating activations.
      const scale = this.act.init === 'he' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
      for (let k = 0; k < W.length; k++) W[k] = randn(this.rng) * scale;
      this.W.push(W);                this.b.push(alloc(nOut));
      this.gW.push(alloc(W.length)); this.gb.push(alloc(nOut));
      this.mW.push(alloc(W.length)); this.mb.push(alloc(nOut));
      this.vW.push(alloc(W.length)); this.vb.push(alloc(nOut));
    }

    // Per-layer buffers, reused for every sample (no allocation in the hot loop).
    this.z = layers.map((n) => alloc(n));     // pre-activations
    this.a = layers.map((n) => alloc(n));     // activations (a[0] = input, a[L] = softmax probs)
    this.delta = layers.map((n) => alloc(n)); // dLoss/dz per layer
  }

  get paramCount() {
    let n = 0;
    for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length;
    return n;
  }

  /** Forward pass for one sample. Returns the softmax probabilities (a view, copy if you keep it). */
  forward(input) {
    const { sizes, L, W, b, z, a, act } = this;
    const a0 = a[0];
    for (let i = 0; i < sizes[0]; i++) a0[i] = input[i];

    for (let l = 0; l < L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const Wl = W[l], bl = b[l], aIn = a[l], zOut = z[l + 1], aOut = a[l + 1];
      for (let j = 0; j < nOut; j++) {
        let s = bl[j];
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) s += Wl[row + i] * aIn[i];
        zOut[j] = s;
      }
      if (l < L - 1) {
        for (let j = 0; j < nOut; j++) aOut[j] = act.f(zOut[j]);
      } else {
        // Numerically stable softmax.
        let max = -Infinity;
        for (let j = 0; j < nOut; j++) if (zOut[j] > max) max = zOut[j];
        let sum = 0;
        for (let j = 0; j < nOut; j++) { aOut[j] = Math.exp(zOut[j] - max); sum += aOut[j]; }
        for (let j = 0; j < nOut; j++) aOut[j] /= sum;
      }
    }
    return a[L];
  }

  /**
   * Backpropagation for the sample last passed to forward().
   * Adds dLoss/dW and dLoss/db into the gradient accumulators.
   *
   * Softmax + cross-entropy gives the famously clean output error:  δ_L = p − y
   * Then, layer by layer:   ∂L/∂W = δ · aᵀ,   δ_prev = (Wᵀ δ) ⊙ f'(z)
   */
  backward(label) {
    const { sizes, L, W, gW, gb, a, z, delta, act } = this;

    // Output layer: gradient of cross-entropy w.r.t. the softmax logits.
    const p = a[L], dOut = delta[L];
    for (let k = 0; k < sizes[L]; k++) dOut[k] = p[k] - (k === label ? 1 : 0);

    for (let l = L - 1; l >= 0; l--) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const Wl = W[l], gWl = gW[l], gbl = gb[l];
      const aIn = a[l], dNext = delta[l + 1];

      // Accumulate weight and bias gradients:  ∂L/∂W_ji = δ_j · a_i
      for (let j = 0; j < nOut; j++) {
        const d = dNext[j];
        gbl[j] += d;
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) gWl[row + i] += d * aIn[i];
      }

      // Propagate the error to the previous hidden layer (not to the inputs).
      if (l > 0) {
        const dCur = delta[l], zIn = z[l];
        for (let i = 0; i < nIn; i++) {
          let s = 0;
          for (let j = 0; j < nOut; j++) s += Wl[j * nIn + i] * dNext[j];
          dCur[i] = s * act.d(aIn[i], zIn[i]);
        }
      }
    }
  }

  zeroGrad() {
    for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
  }

  /** forward + backward for one sample; returns its cross-entropy loss. */
  accumulate(input, label) {
    const p = this.forward(input);
    this.backward(label);
    return -Math.log(p[label] + 1e-12);
  }

  /**
   * Apply one optimiser step using the gradients accumulated over `n` samples.
   * @param {number} n batch size the gradients were summed over
   */
  step(n, { lr = 0.01, l2 = 0, optimizer = 'adam', beta1 = 0.9, beta2 = 0.999, eps = 1e-8, momentum = 0.9 } = {}) {
    this.t++;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const inv = 1 / n;

    for (let l = 0; l < this.L; l++) {
      const groups = [
        [this.W[l], this.gW[l], this.mW[l], this.vW[l], l2],
        [this.b[l], this.gb[l], this.mb[l], this.vb[l], 0], // no decay on biases
      ];
      for (const [P, G, M, V, decay] of groups) {
        for (let k = 0; k < P.length; k++) {
          const g = G[k] * inv + decay * P[k];
          if (optimizer === 'adam') {
            M[k] = beta1 * M[k] + (1 - beta1) * g;
            V[k] = beta2 * V[k] + (1 - beta2) * g * g;
            P[k] -= (lr * (M[k] / bc1)) / (Math.sqrt(V[k] / bc2) + eps);
          } else {
            M[k] = momentum * M[k] - lr * g; // SGD with momentum
            P[k] += M[k];
          }
        }
      }
    }
  }

  /**
   * One pass over the dataset in shuffled mini-batches.
   * @returns {{loss:number, acc:number}} mean loss and accuracy over the epoch
   */
  trainEpoch(X, Y, { batchSize = 16, ...opts } = {}) {
    const n = X.length;
    if (n === 0) return { loss: 0, acc: 0 };
    this._cursor = n; // force a fresh shuffle
    const steps = Math.ceil(n / batchSize);
    this.trainSteps(X, Y, steps, { batchSize, ...opts });
    return this.lastEpoch;
  }

  /**
   * Run `steps` mini-batch updates, continuing through the data across calls and
   * reshuffling at every epoch boundary. The live demo calls this once per
   * animation frame, so you can watch the decision boundary move.
   * @returns {number} epochs completed during this call
   */
  trainSteps(X, Y, steps, { batchSize = 16, ...opts } = {}) {
    const n = X.length;
    if (n === 0) return 0;
    if (!this._order || this._order.length !== n) {
      this._order = Uint32Array.from({ length: n }, (_, i) => i);
      this._cursor = n;
    }
    const idx = this._order;
    let epochs = 0;

    for (let s = 0; s < steps; s++) {
      if (this._cursor >= n) {
        for (let i = n - 1; i > 0; i--) { // Fisher–Yates shuffle
          const j = Math.floor(this.rng() * (i + 1));
          const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        this._cursor = 0; this._runLoss = 0; this._runCorrect = 0;
      }
      const start = this._cursor, end = Math.min(start + batchSize, n);
      this.zeroGrad();
      for (let k = start; k < end; k++) {
        const i = idx[k];
        this._runLoss += this.accumulate(X[i], Y[i]);
        if (argmax(this.a[this.L]) === Y[i]) this._runCorrect++;
      }
      this.step(end - start, opts);
      this._cursor = end;
      if (end >= n) {
        epochs++;
        this.lastEpoch = { loss: this._runLoss / n, acc: this._runCorrect / n };
      }
    }
    return epochs;
  }

  /** Mean cross-entropy over a dataset (forward only). */
  lossOf(X, Y) {
    let loss = 0;
    for (let s = 0; s < X.length; s++) loss -= Math.log(this.forward(X[s])[Y[s]] + 1e-12);
    return loss / X.length;
  }

  /** Accuracy over a dataset (forward only). */
  accuracyOf(X, Y) {
    let c = 0;
    for (let s = 0; s < X.length; s++) if (argmax(this.forward(X[s])) === Y[s]) c++;
    return X.length ? c / X.length : 0;
  }

  predict(input) {
    return argmax(this.forward(input));
  }
}

export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}
