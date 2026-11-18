/**
 * LanguageLab.js — UI for the tiny GPT: live samples while it trains, prompt
 * completion, an attention inspector and the training curve. The model itself
 * trains in gpt.worker.js.
 */
import { Transformer } from '../nn/Transformer.js';
import { COLORS, fitCanvas, segmented, drawLines, pushCapped, showSource } from '../lab/ui.js';

const HEAD_COLORS = [COLORS.ember, COLORS.ice, COLORS.lime, COLORS.violet];
const PROBE_TEXT = 'attention lets a model look back';

export class LanguageLab {
  constructor(root) {
    this.root = root;
    const $ = (s) => root.querySelector(s);
    this.el = {
      sample: $('#lm-sample'), sampleStep: $('#lm-sample-step'), live: $('#lm-live'),
      form: $('#lm-form'), prompt: $('#lm-prompt'), temp: $('#lm-temp'), tempOut: $('#lm-temp-out'), output: $('#lm-output'),
      attn: $('#lm-attn'), layer: $('#lm-layer'), head: $('#lm-head'),
      step: $('#lm-step'), loss: $('#lm-loss'), tps: $('#lm-tps'), params: $('#lm-params'),
      arch: $('#lm-arch'), chart: $('#lm-chart'), play: $('#lm-play'), reset: $('#lm-reset'), code: $('#lm-code'),
    };
    this.userPlaying = true;
    this.active = false;
    this.visible = false;
    this.history = [];
    this.typer = { el: this.el.sample, text: '', shown: 0 };
    this.outTyper = null;
    this.attnState = { layer: 1, head: 'all', focus: null, hover: false, data: null };
    this.genId = 0;

    showSource(this.el.code, Transformer.prototype.attentionBackward, 'transformer.attentionBackward()');
    this._bind();

    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; this._sync(); }, { rootMargin: '80px' }).observe(root);
  }

  _startWorker() {
    if (this.worker) return;
    this.worker = new Worker(new URL('./gpt.worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data }) => this._onMessage(data);
    this.worker.onerror = (e) => { this.el.sample.textContent = `The training worker could not start (${e.message || 'unknown error'}).`; };
  }

  setActive(on) {
    this.active = on;
    if (on) this._startWorker();
    this._sync();
  }

  _sync() {
    if (!this.worker) return;
    const run = this.active && this.visible && this.userPlaying;
    if (run !== this.running) { this.running = run; this.worker.postMessage({ type: run ? 'play' : 'pause' }); }
    this.el.live.classList.toggle('is-on', !!run);
  }

  _onMessage(m) {
    if (m.type === 'ready') {
      this.history = [];
      this.el.params.textContent = m.params.toLocaleString();
      const c = m.config;
      this.el.arch.innerHTML = [
        ['Vocabulary', `${m.vocab} characters`], ['Context', `${c.blockSize} characters`],
        ['Embedding', `${c.dModel} dimensions`], ['Blocks', `${c.nLayer} × (attention + MLP)`],
        ['Heads', `${c.nHead} per block`], ['Trained on', `${m.corpusChars.toLocaleString()} characters of text from this studio`],
      ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
      this._type(this.typer, 'waiting for the first sample… right now the network only knows random noise.');
      this.worker.postMessage({ type: 'attention', text: PROBE_TEXT });
    } else if (m.type === 'progress') {
      this.step = m.step;
      pushCapped(this.history, m.loss);
      this.el.step.textContent = m.step.toLocaleString();
      this.el.loss.textContent = m.loss.toFixed(3);
      this.el.tps.textContent = m.tokensPerSec.toLocaleString();
      if (m.sample) {
        this.el.sampleStep.textContent = m.step.toLocaleString();
        this._type(this.typer, m.sample);
        if (!this.attnState.custom) this.worker.postMessage({ type: 'attention', text: PROBE_TEXT });
      }
    } else if (m.type === 'generated' && m.id === this.genId) {
      this.outTyper = { el: this.el.output, prefix: m.prompt, text: m.text, shown: 0 };
      this.el.form.querySelector('button').disabled = false;
      this.attnState.custom = true;
      this.worker.postMessage({ type: 'attention', text: m.prompt + m.text.slice(0, 32 - Math.min(m.prompt.length, 24)) });
    } else if (m.type === 'attention') {
      this.attnState.data = m;
      if (this.attnState.focus === null || this.attnState.focus >= m.text.length) this.attnState.focus = m.text.length - 1;
    }
  }

  _type(t, text) {
    t.text = text; t.shown = 0;
  }

  _bind() {
    this.el.form.addEventListener('submit', (e) => {
      e.preventDefault();
      if (!this.worker) return;
      this.genId++;
      this.el.form.querySelector('button').disabled = true;
      this.el.output.innerHTML = '<span class="lm__muted">generating…</span>';
      this.worker.postMessage({ type: 'generate', id: this.genId, prompt: this.el.prompt.value, temperature: Number(this.el.temp.value) / 100, length: 180 });
    });
    const onTemp = () => { this.el.tempOut.textContent = (Number(this.el.temp.value) / 100).toFixed(2); };
    this.el.temp.addEventListener('input', onTemp);
    onTemp();

    this.el.play.addEventListener('click', () => {
      this.userPlaying = !this.userPlaying;
      this.el.play.textContent = this.userPlaying ? 'Pause' : 'Train';
      this.el.play.setAttribute('aria-pressed', String(this.userPlaying));
      this._sync();
    });
    this.el.reset.addEventListener('click', () => {
      this.worker?.postMessage({ type: 'reset' });
      this.attnState.custom = false;
      this.el.output.textContent = '';
    });

    segmented(this.el.layer, ['0', '1'], '1', (v) => { this.attnState.layer = Number(v); }, ['Block 1', 'Block 2']);
    segmented(this.el.head, ['all', '0', '1', '2', '3'], 'all', (v) => { this.attnState.head = v; },
      ['All heads', ...HEAD_COLORS.map((c, i) => `<i style="background:${c}"></i>H${i + 1}`)]);

    const c = this.el.attn;
    c.addEventListener('pointermove', (e) => {
      const d = this.attnState.data;
      if (!d || !this._layout) return;
      const r = c.getBoundingClientRect(), x = (e.clientX - r.left) * (c.width / r.width);
      const { x0, cw } = this._layout;
      this.attnState.focus = Math.max(1, Math.min(d.text.length - 1, Math.round((x - x0) / cw - 0.5)));
      this.attnState.hover = true;
    });
    c.addEventListener('pointerleave', () => { this.attnState.hover = false; });
  }

  frame(time) {
    if (!this.active || !this.visible) return;
    this._advanceTyper(this.typer, 4);
    if (this.outTyper) this._advanceTyper(this.outTyper, 3);

    // When nobody is hovering, the focus sweeps along the text.
    const a = this.attnState;
    if (a.data && !a.hover) a.focus = 1 + (Math.floor(time / 0.55) % (a.data.text.length - 1));
    this._drawAttention();
    if (Math.floor(time * 10) % 3 === 0) {
      fitCanvas(this.el.chart);
      drawLines(this.el.chart, [{ values: this.history, max: Math.max(...this.history, 0.1), color: COLORS.ember, fill: 'rgba(255,106,61,0.18)' }]);
    }
  }

  _advanceTyper(t, speed) {
    if (t.shown >= t.text.length) return;
    t.shown = Math.min(t.text.length, t.shown + speed);
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const body = esc(t.text.slice(0, t.shown)).replace(/\n+/g, '<br>');
    const caret = t.shown < t.text.length ? '<span class="lm__caret"></span>' : '';
    t.el.innerHTML = (t.prefix ? `<span class="lm__muted">${esc(t.prefix)}</span>` : '') + body + caret;
  }

  _drawAttention() {
    const c = this.el.attn, d = this.attnState.data;
    const dpr = fitCanvas(c), ctx = c.getContext('2d'), W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    if (!d) return;
    const n = d.text.length, { layer, head } = this.attnState, focus = this.attnState.focus;
    const cw = Math.min(26 * dpr, (W - 32 * dpr) / n), x0 = (W - cw * n) / 2, base = H - 22 * dpr;
    this._layout = { x0, cw };
    const cx = (i) => x0 + (i + 0.5) * cw;
    const heads = head === 'all' ? [0, 1, 2, 3].slice(0, d.heads) : [Number(head)];
    const at = (h, t, u) => d.att[((layer * d.heads + h) * n + t) * n + u];

    // Arcs from every earlier character u to the focus character t.
    ctx.lineCap = 'round';
    for (const h of heads) {
      for (let u = 0; u <= focus; u++) {
        const w = at(h, focus, u);
        if (w < 0.02) continue;
        const x1 = cx(u), x2 = cx(focus), lift = Math.min(H - 60 * dpr, (18 + Math.abs(focus - u) * 7) * dpr);
        ctx.beginPath();
        ctx.moveTo(x1, base - 18 * dpr);
        ctx.bezierCurveTo(x1, base - 18 * dpr - lift, x2, base - 18 * dpr - lift, x2, base - 18 * dpr);
        ctx.strokeStyle = HEAD_COLORS[h];
        ctx.globalAlpha = Math.min(1, 0.15 + w * 1.1);
        ctx.lineWidth = (0.8 + w * 5) * dpr;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Characters, tinted by how much attention they receive.
    ctx.font = `${Math.round(Math.min(18 * dpr, cw * 0.9))}px "JetBrains Mono", ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    for (let u = 0; u < n; u++) {
      let w = 0;
      for (const h of heads) if (u <= focus) w = Math.max(w, at(h, focus, u));
      const ch = d.text[u] === ' ' ? '·' : d.text[u] === '\n' ? '↵' : d.text[u];
      if (u === focus) {
        ctx.fillStyle = 'rgba(255,106,61,0.18)';
        ctx.fillRect(cx(u) - cw / 2 + dpr, base - 18 * dpr, cw - 2 * dpr, 24 * dpr);
        ctx.fillStyle = '#ffffff';
      } else {
        ctx.fillStyle = u > focus ? 'rgba(238,240,246,0.18)' : `rgba(238,240,246,${0.35 + 0.65 * Math.min(1, w * 2)})`;
      }
      ctx.fillText(ch, cx(u), base);
    }
  }
}
