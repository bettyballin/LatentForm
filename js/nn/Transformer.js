/**
 * Transformer.js — a tiny GPT (decoder-only transformer) written from scratch.
 *
 *   token + position embeddings
 *   N × [ LayerNorm → causal multi-head self-attention → residual
 *         LayerNorm → MLP (GELU)                        → residual ]
 *   LayerNorm → linear → softmax over the next character
 *
 * Every backward pass below is derived and written by hand; tests/transformer.test.mjs
 * checks each one against numerical gradients. No ML libraries.
 */
import { createRng, randn } from './Network.js';

const GELU_C = Math.sqrt(2 / Math.PI);

export class Transformer {
  /**
   * @param {object} o
   * @param {number} o.vocabSize  number of distinct tokens (characters)
   * @param {number} o.blockSize  context length T
   * @param {number} o.dModel     embedding width d
   * @param {number} o.nHead      attention heads (d must be divisible)
   * @param {number} o.nLayer     transformer blocks
   * @param {Function} o.dtype    Float32Array (fast) or Float64Array (for gradient checks)
   */
  constructor({ vocabSize, blockSize = 32, dModel = 48, nHead = 4, nLayer = 2, seed = 1, dtype = Float32Array } = {}) {
    if (dModel % nHead) throw new Error('dModel must be divisible by nHead');
    Object.assign(this, { V: vocabSize, T: blockSize, d: dModel, H: nHead, hd: dModel / nHead, L: nLayer, dtype });
    this.rng = createRng(seed);
    this.t = 0; // Adam step
    this.params = [];

    const { V, T, d } = this;
    const std = 0.08, projStd = std / Math.sqrt(2 * nLayer); // residual projections start small
    const P = (size, init = 'zeros', std_ = std) => {
      const w = new dtype(size);
      if (init === 'normal') for (let i = 0; i < size; i++) w[i] = randn(this.rng) * std_;
      if (init === 'ones') w.fill(1);
      const p = { w, g: new dtype(size), m: new dtype(size), v: new dtype(size) };
      this.params.push(p);
      return p;
    };

    this.wte = P(V * d, 'normal');
    this.wpe = P(T * d, 'normal');
    this.layers = [];
    for (let l = 0; l < nLayer; l++) {
      this.layers.push({
        ln1g: P(d, 'ones'), ln1b: P(d),
        Wqkv: P(d * 3 * d, 'normal'), bqkv: P(3 * d),
        Wo: P(d * d, 'normal', projStd), bo: P(d),
        ln2g: P(d, 'ones'), ln2b: P(d),
        W1: P(d * 4 * d, 'normal'), b1: P(4 * d),
        W2: P(4 * d * d, 'normal', projStd), b2: P(d),
      });
    }
    this.lnfg = P(d, 'ones'); this.lnfb = P(d);
    this.Wout = P(d * V, 'normal'); this.bout = P(V);

    // Activation caches (one sequence at a time) and gradient scratch buffers.
    const A = (n) => new dtype(n);
    this.cache = this.layers.map(() => ({
      xIn: A(T * d), ln1: A(T * d), ln1hat: A(T * d), ln1rstd: A(T),
      qkv: A(T * 3 * d), att: A(nHead * T * T), attOut: A(T * d),
      xMid: A(T * d), ln2: A(T * d), ln2hat: A(T * d), ln2rstd: A(T),
      h: A(T * 4 * d), a: A(T * 4 * d),
    }));
    Object.assign(this, {
      xFinal: A(T * d), lnf: A(T * d), lnfhat: A(T * d), lnfrstd: A(T),
      tmp: A(T * d), logits: A(T * V), probs: A(T * V),
      dres: A(T * d), dln: A(T * d), dattOut: A(T * d), dqkv: A(T * 3 * d),
      dh: A(T * 4 * d), da: A(T * 4 * d), dlogits: A(T * V), datt: A(T),
    });
  }

  get paramCount() {
    return this.params.reduce((n, p) => n + p.w.length, 0);
  }

  // ───────────────────────────── forward ─────────────────────────────

  /** Forward pass over a sequence of token ids (length ≤ T). Returns logits (n × V). */
  forward(idx) {
    const n = idx.length, { d, V, L } = this;
    this.idx = idx; this.n = n;

    const x0 = L ? this.cache[0].xIn : this.xFinal;
    for (let t = 0; t < n; t++) {
      const eo = idx[t] * d, po = t * d;
      for (let i = 0; i < d; i++) x0[po + i] = this.wte.w[eo + i] + this.wpe.w[po + i];
    }

    for (let l = 0; l < L; l++) {
      const c = this.cache[l], P = this.layers[l];
      layerNorm(c.ln1, c.ln1hat, c.ln1rstd, c.xIn, P.ln1g.w, P.ln1b.w, n, d);
      matmul(c.qkv, c.ln1, P.Wqkv.w, P.bqkv.w, n, d, 3 * d);
      this.attention(c.attOut, c.att, c.qkv, n);
      matmul(this.tmp, c.attOut, P.Wo.w, P.bo.w, n, d, d);
      for (let k = 0; k < n * d; k++) c.xMid[k] = c.xIn[k] + this.tmp[k];

      layerNorm(c.ln2, c.ln2hat, c.ln2rstd, c.xMid, P.ln2g.w, P.ln2b.w, n, d);
      matmul(c.h, c.ln2, P.W1.w, P.b1.w, n, d, 4 * d);
      for (let k = 0; k < n * 4 * d; k++) c.a[k] = gelu(c.h[k]);
      matmul(this.tmp, c.a, P.W2.w, P.b2.w, n, 4 * d, d);
      const next = l + 1 < L ? this.cache[l + 1].xIn : this.xFinal;
      for (let k = 0; k < n * d; k++) next[k] = c.xMid[k] + this.tmp[k];
    }

    layerNorm(this.lnf, this.lnfhat, this.lnfrstd, this.xFinal, this.lnfg.w, this.lnfb.w, n, d);
    matmul(this.logits, this.lnf, this.Wout.w, this.bout.w, n, d, V);
    return this.logits;
  }

  /** Causal multi-head self-attention: each position mixes the values of itself and earlier positions. */
  attention(out, att, qkv, n) {
    const { d, H, hd, T } = this, s3 = 3 * d, scale = 1 / Math.sqrt(hd);
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < n; t++) {
        const qo = t * s3 + h * hd, row = (h * T + t) * T;
        let max = -Infinity;
        for (let u = 0; u <= t; u++) {
          const ko = u * s3 + d + h * hd;
          let s = 0;
          for (let e = 0; e < hd; e++) s += qkv[qo + e] * qkv[ko + e];
          s *= scale;
          att[row + u] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let u = 0; u <= t; u++) { const e = Math.exp(att[row + u] - max); att[row + u] = e; sum += e; }
        for (let u = 0; u <= t; u++) att[row + u] /= sum;
        for (let u = t + 1; u < T; u++) att[row + u] = 0;

        const oo = t * d + h * hd;
        for (let e = 0; e < hd; e++) out[oo + e] = 0;
        for (let u = 0; u <= t; u++) {
          const w = att[row + u], vo = u * s3 + 2 * d + h * hd;
          for (let e = 0; e < hd; e++) out[oo + e] += w * qkv[vo + e];
        }
      }
    }
  }

  /** Mean cross-entropy of predicting tgt[t] at every position (forward only). */
  loss(idx, tgt) {
    this.forward(idx);
    return this._softmaxLoss(tgt);
  }

  _softmaxLoss(tgt) {
    const { n, V, logits, probs } = this;
    let loss = 0;
    for (let t = 0; t < n; t++) {
      const o = t * V;
      let max = -Infinity;
      for (let k = 0; k < V; k++) if (logits[o + k] > max) max = logits[o + k];
      let sum = 0;
      for (let k = 0; k < V; k++) { probs[o + k] = Math.exp(logits[o + k] - max); sum += probs[o + k]; }
      for (let k = 0; k < V; k++) probs[o + k] /= sum;
      loss -= Math.log(probs[o + tgt[t]] + 1e-12);
    }
    return loss / n;
  }

  // ───────────────────────────── backward ─────────────────────────────

  /**
   * Backpropagate the loss of the last forward() into every parameter's .g.
   * @param {ArrayLike<number>} tgt  next-token targets
   * @param {number} scale           gradient scale, e.g. 1 / (n · batch) for a mean
   */
  backward(tgt, scale) {
    const { n, d, V, L, dres, dln, dattOut, dqkv, dh, da, dlogits } = this;

    // Softmax + cross-entropy: dL/dlogits = p − onehot(target)
    for (let t = 0; t < n; t++) {
      const o = t * V;
      for (let k = 0; k < V; k++) dlogits[o + k] = (this.probs[o + k] - (k === tgt[t] ? 1 : 0)) * scale;
    }
    dln.fill(0);
    matmulBack(dln, this.Wout.g, this.bout.g, dlogits, this.lnf, this.Wout.w, n, d, V);
    dres.fill(0);
    layerNormBack(dres, dln, this.lnfhat, this.lnfrstd, this.lnfg.w, this.lnfg.g, this.lnfb.g, n, d);

    for (let l = L - 1; l >= 0; l--) {
      const c = this.cache[l], P = this.layers[l];

      // MLP branch:  xOut = xMid + W2·gelu(W1·LN2(xMid))
      da.fill(0);
      matmulBack(da, P.W2.g, P.b2.g, dres, c.a, P.W2.w, n, 4 * d, d);
      for (let k = 0; k < n * 4 * d; k++) dh[k] = da[k] * geluGrad(c.h[k]);
      dln.fill(0);
      matmulBack(dln, P.W1.g, P.b1.g, dh, c.ln2, P.W1.w, n, d, 4 * d);
      layerNormBack(dres, dln, c.ln2hat, c.ln2rstd, P.ln2g.w, P.ln2g.g, P.ln2b.g, n, d); // dres is now dL/dxMid

      // Attention branch:  xMid = xIn + Wo·attention(Wqkv·LN1(xIn))
      dattOut.fill(0);
      matmulBack(dattOut, P.Wo.g, P.bo.g, dres, c.attOut, P.Wo.w, n, d, d);
      dqkv.fill(0);
      this.attentionBackward(dqkv, dattOut, c.att, c.qkv, n);
      dln.fill(0);
      matmulBack(dln, P.Wqkv.g, P.bqkv.g, dqkv, c.ln1, P.Wqkv.w, n, d, 3 * d);
      layerNormBack(dres, dln, c.ln1hat, c.ln1rstd, P.ln1g.w, P.ln1g.g, P.ln1b.g, n, d); // dres is now dL/dxIn
    }

    for (let t = 0; t < n; t++) {
      const eo = this.idx[t] * d, po = t * d;
      for (let i = 0; i < d; i++) { this.wte.g[eo + i] += dres[po + i]; this.wpe.g[po + i] += dres[po + i]; }
    }
  }

  /**
   * Backward pass of causal self-attention.
   *   out_t = Σ_u a_tu v_u,   a_t = softmax(q_t · k_u / √d_head)
   *   ⇒ dv_u += a_tu·dout_t,   da_tu = dout_t · v_u,
   *     ds_tu = a_tu (da_tu − Σ_u' a_tu' da_tu'),   dq_t += ds_tu k_u / √d,   dk_u += ds_tu q_t / √d
   */
  attentionBackward(dqkv, dout, att, qkv, n) {
    const { d, H, hd, T, datt } = this, s3 = 3 * d, scale = 1 / Math.sqrt(hd);
    for (let h = 0; h < H; h++) {
      for (let t = 0; t < n; t++) {
        const row = (h * T + t) * T, oo = t * d + h * hd, qo = t * s3 + h * hd;
        let dot = 0;
        for (let u = 0; u <= t; u++) {
          const vo = u * s3 + 2 * d + h * hd, a = att[row + u];
          let s = 0;
          for (let e = 0; e < hd; e++) { s += dout[oo + e] * qkv[vo + e]; dqkv[vo + e] += a * dout[oo + e]; }
          datt[u] = s;
          dot += a * s;
        }
        for (let u = 0; u <= t; u++) {
          const ds = att[row + u] * (datt[u] - dot) * scale, ko = u * s3 + d + h * hd;
          for (let e = 0; e < hd; e++) { dqkv[qo + e] += ds * qkv[ko + e]; dqkv[ko + e] += ds * qkv[qo + e]; }
        }
      }
    }
  }

  // ───────────────────────────── training ─────────────────────────────

  zeroGrad() {
    for (const p of this.params) p.g.fill(0);
  }

  /** One optimiser step on a mini-batch of random windows from `data` (token ids). Returns mean loss. */
  trainStep(data, { batchSize = 8, lr = 3e-3, clip = 1, beta1 = 0.9, beta2 = 0.99, eps = 1e-8 } = {}) {
    const T = this.T;
    this.zeroGrad();
    let loss = 0;
    for (let b = 0; b < batchSize; b++) {
      const s = Math.floor(this.rng() * (data.length - T - 1));
      const idx = data.subarray(s, s + T), tgt = data.subarray(s + 1, s + T + 1);
      loss += this.loss(idx, tgt);
      this.backward(tgt, 1 / (T * batchSize));
    }

    // Global gradient-norm clipping keeps early training stable.
    let norm = 0;
    for (const p of this.params) for (let i = 0; i < p.g.length; i++) norm += p.g[i] * p.g[i];
    norm = Math.sqrt(norm);
    const k = norm > clip ? clip / norm : 1;

    // Adam
    this.t++;
    const bc1 = 1 - Math.pow(beta1, this.t), bc2 = 1 - Math.pow(beta2, this.t);
    for (const p of this.params) {
      const { w, g, m, v } = p;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * k;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        w[i] -= (lr * m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
    return loss / batchSize;
  }

  /** Autoregressive sampling: append `length` tokens after `context`. */
  generate(context, length, { temperature = 0.8, rng = Math.random } = {}) {
    const out = context.slice();
    const { V } = this;
    const p = new Float64Array(V);
    for (let s = 0; s < length; s++) {
      const ctx = out.slice(-this.T);
      const logits = this.forward(ctx), o = (ctx.length - 1) * V;
      let max = -Infinity;
      for (let k = 0; k < V; k++) if (logits[o + k] > max) max = logits[o + k];
      let sum = 0;
      for (let k = 0; k < V; k++) { p[k] = Math.exp((logits[o + k] - max) / temperature); sum += p[k]; }
      let r = rng() * sum, next = V - 1;
      for (let k = 0; k < V; k++) { r -= p[k]; if (r <= 0) { next = k; break; } }
      out.push(next);
    }
    return out.slice(context.length);
  }

  /** Attention weights for a sequence: Float32Array [layer][head][t][u] (n × n per head). */
  attentionMaps(idx) {
    this.forward(idx);
    const { L, H, T } = this, n = idx.length;
    const out = new Float32Array(L * H * n * n);
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) for (let t = 0; t < n; t++) for (let u = 0; u < n; u++) {
      out[((l * H + h) * n + t) * n + u] = this.cache[l].att[(h * T + t) * T + u];
    }
    return out;
  }
}

// ───────────────────────────── building blocks ─────────────────────────────

/** out[t, j] = b[j] + Σ_i x[t, i] · W[i, j]   (x: n×m, W: m×k) */
function matmul(out, x, W, b, n, m, k) {
  for (let t = 0; t < n; t++) {
    const oo = t * k, xo = t * m;
    for (let j = 0; j < k; j++) out[oo + j] = b[j];
    for (let i = 0; i < m; i++) {
      const xv = x[xo + i], wo = i * k;
      for (let j = 0; j < k; j++) out[oo + j] += xv * W[wo + j];
    }
  }
}

/** Gradients of matmul: dW += xᵀ·dout, db += Σ dout, dx += dout·Wᵀ. */
function matmulBack(dx, dW, db, dout, x, W, n, m, k) {
  for (let t = 0; t < n; t++) {
    const oo = t * k, xo = t * m;
    for (let j = 0; j < k; j++) db[j] += dout[oo + j];
    for (let i = 0; i < m; i++) {
      const xv = x[xo + i], wo = i * k;
      let s = 0;
      for (let j = 0; j < k; j++) { const g = dout[oo + j]; dW[wo + j] += xv * g; s += W[wo + j] * g; }
      dx[xo + i] += s;
    }
  }
}

function layerNorm(out, xhat, rstd, x, g, b, n, d) {
  for (let t = 0; t < n; t++) {
    const o = t * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[o + i];
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) { const c = x[o + i] - mean; v += c * c; }
    const r = 1 / Math.sqrt(v / d + 1e-5);
    rstd[t] = r;
    for (let i = 0; i < d; i++) { const xh = (x[o + i] - mean) * r; xhat[o + i] = xh; out[o + i] = xh * g[i] + b[i]; }
  }
}

/** dx += rstd · (dx̂ − mean(dx̂) − x̂ · mean(dx̂ · x̂)),  with dx̂ = dy · γ. */
function layerNormBack(dx, dy, xhat, rstd, g, dg, db, n, d) {
  for (let t = 0; t < n; t++) {
    const o = t * d;
    let m1 = 0, m2 = 0;
    for (let i = 0; i < d; i++) {
      const dyi = dy[o + i], dxh = dyi * g[i];
      dg[i] += dyi * xhat[o + i]; db[i] += dyi;
      m1 += dxh; m2 += dxh * xhat[o + i];
    }
    m1 /= d; m2 /= d;
    for (let i = 0; i < d; i++) dx[o + i] += rstd[t] * (dy[o + i] * g[i] - m1 - xhat[o + i] * m2);
  }
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

function geluGrad(x) {
  const th = Math.tanh(GELU_C * (x + 0.044715 * x * x * x));
  return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GELU_C * (1 + 3 * 0.044715 * x * x);
}
