/**
 * Lab.js — the interactive training playground.
 *
 * Trains the from-scratch Network a few mini-batches per animation frame,
 * paints the decision boundary, plots loss/accuracy, mirrors the network in 3D,
 * and shows the real backward() source code that is doing the learning.
 */
import { Network } from '../nn/Network.js';
import { DATASETS } from '../nn/datasets.js';
import { Network3D, CLASS_HEX } from './Network3D.js';

const DOMAIN = 1.15;  // the plane shows [-DOMAIN, DOMAIN]²
const RES = 72;       // decision-boundary grid resolution (upscaled smoothly)
const MAX_POINTS = 800;
const SPEEDS = { '1×': 2, '4×': 8, '16×': 32 }; // mini-batches per frame
const ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'sin'];
const CLASS_RGB = CLASS_HEX.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
const BG_RGB = [10, 12, 19];
const CLASS_NAMES = ['A', 'B', 'C'];

// Log-scale slider: 0 → 0.0003, 50 → 0.01, 100 → 0.32 (high enough to watch it diverge).
const lrFromSlider = (v) => Math.pow(10, -3.5 + v * 0.03);
const sliderFromLr = (lr) => Math.round((Math.log10(lr) + 3.5) / 0.03);

export class Lab {
  constructor(root, { onParams } = {}) {
    this.root = root;
    this.onParams = onParams;
    const $ = (s) => root.querySelector(s);
    this.el = {
      plane: $('#lab-plane'), readout: $('#lab-readout'), net: $('#lab-net'), arch: $('#lab-arch'),
      epoch: $('#stat-epoch'), loss: $('#stat-loss'), acc: $('#stat-acc'), params: $('#stat-params'),
      dataset: $('#ctrl-dataset'), classes: $('#ctrl-class'), activation: $('#ctrl-activation'),
      lr: $('#ctrl-lr'), lrOut: $('#lr-out'), speed: $('#ctrl-speed'),
      play: $('#btn-play'), step: $('#btn-step'), reset: $('#btn-reset'),
      chart: $('#lab-chart'), code: $('#lab-code'),
      layers: $('[data-stepper="layers"]'), neurons: $('[data-stepper="neurons"]'),
    };

    this.state = { dataset: 'spiral', layers: 2, neurons: 12, activation: 'tanh', lr: 0.01, speed: '4×', playing: true, cls: 0 };
    this.seed = 1;
    this.probe = { x: 0, y: 0, hover: false };
    this.visible = false;
    this.frameCount = 0;
    this.input = new Float64Array(2);

    this.ctx = this.el.plane.getContext('2d');
    this.grid = document.createElement('canvas');
    this.grid.width = this.grid.height = RES;
    this.gridCtx = this.grid.getContext('2d');
    this.gridImg = this.gridCtx.createImageData(RES, RES);
    this.chartCtx = this.el.chart.getContext('2d');

    this.net3d = new Network3D(this.el.net);

    this._buildControls();
    this._bindPlane();
    this._renderCode();
    this.loadDataset('spiral');

    new ResizeObserver(() => this._resizeCanvases()).observe(this.el.plane);
    new ResizeObserver(() => this._resizeCanvases()).observe(this.el.chart);
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }, { rootMargin: '100px' }).observe(root);
  }

  // ───────────────────────────── data & model ─────────────────────────────

  loadDataset(key) {
    const d = DATASETS[key].make();
    this.state.dataset = key;
    this.X = d.X; this.Y = d.Y; this.classes = d.classes;
    this.births = new Array(this.X.length).fill(-1);
    this.state.cls = 0;
    this._renderClassChips();
    this._setPressed(this.el.dataset, key);
    this.resetNetwork();
  }

  resetNetwork() {
    const { layers, neurons, activation } = this.state;
    const sizes = [2, ...Array(layers).fill(neurons), this.classes];
    this.net = new Network({ layers: sizes, activation, seed: this.seed++ });
    this.epoch = 0;
    this.history = []; this.stride = 1;
    this.lastStats = null;
    this.net3d.setArchitecture(sizes);
    this.el.arch.textContent = sizes.join(' → ');
    this.el.params.textContent = this.net.paramCount.toLocaleString();
    this.el.layers.querySelector('output').textContent = layers;
    this.el.neurons.querySelector('output').textContent = neurons;
    this._updateStats(true);
    this.onParams?.(this.net.paramCount);
  }

  trainOnce(steps) {
    if (!this.X.length) return;
    const epochs = this.net.trainSteps(this.X, this.Y, steps, { lr: this.state.lr, batchSize: 16 });
    for (let e = 0; e < epochs; e++) {
      this.epoch++;
      if (this.epoch % this.stride === 0) {
        this.history.push({ ...this.net.lastEpoch });
        if (this.history.length > 360) { this.history = this.history.filter((_, i) => i % 2 === 0); this.stride *= 2; }
      }
    }
  }

  // ───────────────────────────── frame loop ─────────────────────────────

  frame(time, dt) {
    if (!this.visible || this.enabled === false) return;
    this.frameCount++;
    this.time = time;

    if (this.state.playing) this.trainOnce(SPEEDS[this.state.speed]);

    if (!this.probe.hover) {
      // Auto-probe wanders along a Lissajous curve so the 3D view is always alive.
      this.probe.x = Math.sin(time * 0.37) * 0.75;
      this.probe.y = Math.sin(time * 0.53 + 1.2) * 0.75;
    }

    this._drawPlane();

    // Forward the probe last, so net.a holds *its* activations for the 3D view.
    this.input[0] = this.probe.x; this.input[1] = this.probe.y;
    const p = this.net.forward(this.input);
    this.net3d.sync(this.net);
    this.net3d.frame(time, dt);

    if (this.frameCount % 4 === 0) { this._updateReadout(p); this._updateStats(); }
    if (this.frameCount % 3 === 0) this._drawChart();
  }

  // ───────────────────────────── rendering ─────────────────────────────

  _resizeCanvases() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const c of [this.el.plane, this.el.chart]) {
      const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
      if (w && h && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; }
    }
  }

  _toPx(x, y) {
    const W = this.el.plane.width, H = this.el.plane.height;
    return [((x + DOMAIN) / (2 * DOMAIN)) * W, ((DOMAIN - y) / (2 * DOMAIN)) * H];
  }

  _drawPlane() {
    const { net, input, gridImg } = this;
    const K = this.classes, data = gridImg.data;

    // 1. Decision boundary: classify every grid cell, colour by class, brightness by confidence.
    for (let gy = 0; gy < RES; gy++) {
      for (let gx = 0; gx < RES; gx++) {
        input[0] = -DOMAIN + ((gx + 0.5) / RES) * 2 * DOMAIN;
        input[1] = DOMAIN - ((gy + 0.5) / RES) * 2 * DOMAIN;
        const p = net.forward(input);
        let r = 0, g = 0, b = 0, max = 0;
        for (let k = 0; k < K; k++) {
          r += p[k] * CLASS_RGB[k][0]; g += p[k] * CLASS_RGB[k][1]; b += p[k] * CLASS_RGB[k][2];
          if (p[k] > max) max = p[k];
        }
        const conf = (max - 1 / K) / (1 - 1 / K);
        const m = 0.06 + 0.24 * conf * Math.sqrt(conf);
        const o = (gy * RES + gx) * 4;
        data[o] = BG_RGB[0] + (r - BG_RGB[0]) * m;
        data[o + 1] = BG_RGB[1] + (g - BG_RGB[1]) * m;
        data[o + 2] = BG_RGB[2] + (b - BG_RGB[2]) * m;
        data[o + 3] = 255;
      }
    }
    this.gridCtx.putImageData(gridImg, 0, 0);

    const ctx = this.ctx, W = this.el.plane.width, H = this.el.plane.height;
    if (!W || !H) return;
    const dpr = W / (this.el.plane.clientWidth || W);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    // A soft blur hides the grid cells; draw slightly oversized so edges stay filled.
    const bleed = 8 * dpr;
    ctx.filter = `blur(${3 * dpr}px)`;
    ctx.drawImage(this.grid, -bleed, -bleed, W + 2 * bleed, H + 2 * bleed);
    ctx.filter = 'none';

    // 2. Axes / grid.
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let v = -1; v <= 1; v += 0.5) {
      const [x] = this._toPx(v, 0), [, y] = this._toPx(0, v);
      ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.moveTo(0, y); ctx.lineTo(W, y);
    }
    ctx.stroke();

    // 3. Data points (misclassified ones get a white ring; new ones pop in).
    const now = this.time || 0;
    for (let s = 0; s < this.X.length; s++) {
      const [px, py] = this._toPx(this.X[s][0], this.X[s][1]);
      let r = 4.2 * dpr;
      const born = this.births[s];
      if (born >= 0 && now - born < 0.6) {
        const t = (now - born) / 0.6;
        r *= 1 + Math.sin(t * Math.PI) * 0.9;
      }
      const wrong = net.predict(this.X[s]) !== this.Y[s];
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_HEX[this.Y[s]];
      ctx.fill();
      ctx.lineWidth = (wrong ? 2 : 1.2) * dpr;
      ctx.strokeStyle = wrong ? 'rgba(255,255,255,0.95)' : 'rgba(5,6,10,0.9)';
      ctx.stroke();
    }

    // 4. Probe crosshair.
    const [qx, qy] = this._toPx(this.probe.x, this.probe.y);
    ctx.strokeStyle = this.probe.hover ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.2 * dpr;
    ctx.setLineDash(this.probe.hover ? [] : [3 * dpr, 3 * dpr]);
    ctx.beginPath();
    ctx.arc(qx, qy, 10 * dpr, 0, Math.PI * 2);
    ctx.moveTo(qx - 16 * dpr, qy); ctx.lineTo(qx - 6 * dpr, qy);
    ctx.moveTo(qx + 6 * dpr, qy); ctx.lineTo(qx + 16 * dpr, qy);
    ctx.moveTo(qx, qy - 16 * dpr); ctx.lineTo(qx, qy - 6 * dpr);
    ctx.moveTo(qx, qy + 6 * dpr); ctx.lineTo(qx, qy + 16 * dpr);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _drawChart() {
    const ctx = this.chartCtx, c = this.el.chart, W = c.width, H = c.height;
    if (!W || !H) return;
    const dpr = W / (c.clientWidth || W);
    ctx.clearRect(0, 0, W, H);
    const pad = 10 * dpr, h = this.history;

    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) { const y = pad + ((H - 2 * pad) * i) / 4; ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); }
    ctx.stroke();
    if (h.length < 2) return;

    const maxLoss = Math.max(...h.map((d) => d.loss), 1e-6);
    const xOf = (i) => pad + (i / (h.length - 1)) * (W - 2 * pad);
    const line = (key, scale, color, fill) => {
      ctx.beginPath();
      h.forEach((d, i) => { const y = H - pad - (d[key] / scale) * (H - 2 * pad); i ? ctx.lineTo(xOf(i), y) : ctx.moveTo(xOf(i), y); });
      ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.lineJoin = 'round'; ctx.stroke();
      if (fill) {
        ctx.lineTo(xOf(h.length - 1), H - pad); ctx.lineTo(xOf(0), H - pad); ctx.closePath();
        const g = ctx.createLinearGradient(0, pad, 0, H);
        g.addColorStop(0, fill); g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g; ctx.fill();
      }
    };
    line('loss', maxLoss, CLASS_HEX[0], 'rgba(255,106,61,0.18)');
    line('acc', 1, CLASS_HEX[1]);
  }

  _updateStats(force) {
    const s = this.net.lastEpoch;
    if (!force && s === this.lastStats) return;
    this.lastStats = s;
    this.el.epoch.textContent = this.epoch.toLocaleString();
    this.el.loss.textContent = s ? s.loss.toFixed(3) : '—';
    this.el.acc.textContent = s ? `${(s.acc * 100).toFixed(1)}%` : '—';
  }

  _updateReadout(p) {
    const rows = [];
    for (let k = 0; k < this.classes; k++) {
      rows.push(`<span class="ro-row"><i style="background:${CLASS_HEX[k]}"></i>${CLASS_NAMES[k]}<b style="--w:${(p[k] * 100).toFixed(1)}%"></b><em>${p[k].toFixed(2)}</em></span>`);
    }
    this.el.readout.innerHTML =
      `<span class="ro-head">${this.probe.hover ? 'Your input' : 'Auto probe'} · (${this.probe.x.toFixed(2)}, ${this.probe.y.toFixed(2)})</span>${rows.join('')}`;
  }

  // ───────────────────────────── controls ─────────────────────────────

  _setPressed(group, value) {
    group.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.value === value)));
  }

  _seg(group, values, current, onPick, labels = values) {
    group.innerHTML = values.map((v, i) => `<button type="button" data-value="${v}" aria-pressed="${v === current}">${labels[i]}</button>`).join('');
    group.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      this._setPressed(group, b.dataset.value);
      onPick(b.dataset.value);
    });
  }

  _buildControls() {
    const keys = Object.keys(DATASETS);
    this._seg(this.el.dataset, keys, this.state.dataset, (k) => this.loadDataset(k), keys.map((k) => DATASETS[k].label));
    this._seg(this.el.activation, ACTIVATIONS, this.state.activation, (a) => { this.state.activation = a; this.resetNetwork(); });
    this._seg(this.el.speed, Object.keys(SPEEDS), this.state.speed, (s) => { this.state.speed = s; });

    const stepper = (el, key, min, max) => el.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-d]');
      if (!b) return;
      const v = Math.max(min, Math.min(max, this.state[key] + Number(b.dataset.d)));
      if (v !== this.state[key]) { this.state[key] = v; this.resetNetwork(); }
    });
    stepper(this.el.layers, 'layers', 1, 4);
    stepper(this.el.neurons, 'neurons', 2, 16);

    this.el.lr.value = sliderFromLr(this.state.lr);
    const onLr = () => { this.state.lr = lrFromSlider(Number(this.el.lr.value)); this.el.lrOut.textContent = this.state.lr.toFixed(4); };
    this.el.lr.addEventListener('input', onLr);
    onLr();

    this.el.play.addEventListener('click', () => {
      this.state.playing = !this.state.playing;
      this.el.play.textContent = this.state.playing ? 'Pause' : 'Train';
      this.el.play.setAttribute('aria-pressed', String(this.state.playing));
    });
    this.el.step.addEventListener('click', () => {
      this.state.playing = false;
      this.el.play.textContent = 'Train';
      this.trainOnce(Math.max(1, Math.ceil(this.X.length / 16)));
      this._updateStats();
    });
    this.el.reset.addEventListener('click', () => this.resetNetwork());
  }

  _renderClassChips() {
    const n = this.classes;
    this.el.classes.innerHTML = Array.from({ length: n }, (_, k) =>
      `<button type="button" data-value="${k}" aria-pressed="${k === this.state.cls}" style="--c:${CLASS_HEX[k]}"><i></i>Class ${CLASS_NAMES[k]}</button>`).join('');
    this.el.classes.onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      this.state.cls = Number(b.dataset.value);
      this._setPressed(this.el.classes, b.dataset.value);
    };
  }

  _bindPlane() {
    const c = this.el.plane;
    const toData = (e) => {
      const r = c.getBoundingClientRect();
      return [((e.clientX - r.left) / r.width) * 2 * DOMAIN - DOMAIN, DOMAIN - ((e.clientY - r.top) / r.height) * 2 * DOMAIN];
    };
    c.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch') return;
      [this.probe.x, this.probe.y] = toData(e);
      this.probe.hover = true;
    });
    c.addEventListener('pointerleave', () => { this.probe.hover = false; });
    c.addEventListener('click', (e) => {
      if (this.X.length >= MAX_POINTS) return;
      const [x, y] = toData(e);
      if (Math.abs(x) > 1.1 || Math.abs(y) > 1.1) return;
      this.X.push([x, y]); this.Y.push(this.state.cls); this.births.push(this.time || 0);
      [this.probe.x, this.probe.y] = [x, y];
      if (!this.state.playing) this.el.play.click();
    });
  }

  _renderCode() {
    const src = dedent(Network.prototype.backward.toString());
    this.el.code.innerHTML = highlight(`// network.backward(label), read at runtime via Function.prototype.toString()\n${src}`);
  }
}

// ───────────────────────────── helpers ─────────────────────────────

function dedent(src) {
  const lines = src.split('\n');
  const indents = lines.slice(1).filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length);
  const min = indents.length ? Math.min(...indents) : 0;
  return [lines[0], ...lines.slice(1).map((l) => l.slice(min))].join('\n');
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Tiny JS syntax highlighter (comments, strings, keywords, numbers, calls). */
function highlight(src) {
  const re = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|('(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(const|let|var|for|if|else|return|new|this|function|of|while)\b|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|([A-Za-z_$][\w$]*)(?=\()/g;
  let out = '', last = 0;
  for (const m of src.matchAll(re)) {
    out += esc(src.slice(last, m.index));
    const cls = m[1] ? 'c' : m[2] ? 's' : m[3] ? 'k' : m[4] ? 'n' : 'f';
    out += `<span class="tk-${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(src.slice(last));
}
