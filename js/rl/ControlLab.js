/**
 * ControlLab.js — UI for the reinforcement-learning balancer.
 *
 * Training episodes run headless in the background (a few per second, so you
 * can watch progress); the stage shows the *current* policy playing in real
 * time. Click beside the cart to push it and test how robust the policy is.
 */
import { CartPole, ReinforceAgent } from './CartPole.js';
import { COLORS, fitCanvas, segmented, showSource } from '../lab/ui.js';

const EPISODES_PER_SECOND = { '1×': 3, '4×': 12, '16×': 48 };
const SOLVED = 475; // average steps over the last 20 episodes (max 500)

export class ControlLab {
  constructor(root) {
    this.root = root;
    const $ = (s) => root.querySelector(s);
    this.el = {
      canvas: $('#rl-canvas'), status: $('#rl-status'), run: $('#rl-run'),
      episodes: $('#rl-episodes'), last: $('#rl-last'), best: $('#rl-best'), avg: $('#rl-avg'),
      chart: $('#rl-chart'), left: $('#rl-left'), right: $('#rl-right'), pl: $('#rl-pl'), pr: $('#rl-pr'),
      speed: $('#rl-speed'), play: $('#rl-play'), reset: $('#rl-reset'), code: $('#rl-code'),
    };
    this.speed = '4×';
    this.seed = 1;
    this.active = false;
    this.visible = false;
    this.frames = 0;

    showSource(this.el.code, ReinforceAgent.prototype.update, 'agent.update(episode)');
    segmented(this.el.speed, Object.keys(EPISODES_PER_SECOND), this.speed, (s) => { this.speed = s; });
    this.el.play.addEventListener('click', () => this._setTraining(!this.training));
    this.el.reset.addEventListener('click', () => this.reset());
    this.el.canvas.addEventListener('pointerdown', (e) => this._push(e));

    this.reset();
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }, { rootMargin: '80px' }).observe(root);
  }

  reset() {
    const s = this.seed++;
    this.agent = new ReinforceAgent({ hidden: 32, lr: 0.01, seed: s });
    this.trainEnv = new CartPole(s);
    this.stage = new CartPole(s + 1000);
    this.lengths = [];
    this.best = 0;
    this.solvedAt = null;
    this.trainAcc = 0;
    this.simAcc = 0;
    this.push = { t: 0, force: 0 };
    this.flash = 0;
    this.pRight = 0.5;
    this._setTraining(true);
    this._stats();
  }

  setActive(on) { this.active = on; }

  _setTraining(on) {
    this.training = on;
    this.el.play.textContent = on ? 'Pause' : this.solvedAt ? 'Keep training' : 'Train';
    this.el.play.setAttribute('aria-pressed', String(on));
  }

  _push(e) {
    const r = this.el.canvas.getBoundingClientRect();
    const cartX = this._toPx ? this._toPx(this.stage.state[0]) / (this.el.canvas.width / r.width) : r.width / 2;
    const dir = e.clientX - r.left < cartX ? -1 : 1;
    this.push = { t: 0.12, force: dir * 30 };
  }

  frame(time, dt) {
    if (!this.active || !this.visible) return;
    this.frames++;

    // 1. Learn: run whole training episodes in the background at the chosen pace.
    if (this.training) {
      this.trainAcc += dt * EPISODES_PER_SECOND[this.speed];
      let budget = 64;
      while (this.trainAcc >= 1 && budget-- > 0) {
        this.trainAcc -= 1;
        const ep = this.agent.runEpisode(this.trainEnv);
        this.agent.update(ep);
        this.lengths.push(ep.length);
        this.best = Math.max(this.best, ep.length);
        if (this.solvedAt === null && this._avg() >= SOLVED && this.lengths.length >= 20) {
          this.solvedAt = this.lengths.length;
          this._setTraining(false); // stop while it's good: plain REINFORCE can drift after solving
        }
      }
    }

    // 2. Act: the current policy balances the stage pole in real time (50 Hz physics).
    this.simAcc += Math.min(dt, 0.05);
    while (this.simAcc >= this.stage.tau) {
      this.simAcc -= this.stage.tau;
      const extra = this.push.t > 0 ? this.push.force : 0;
      this.push.t -= this.stage.tau;
      const { action, pRight } = this.agent.act(this.stage.state);
      this.pRight = pRight;
      this.stage.step(action, extra);
      if (this.stage.done) { this.flash = 1; this.lastRun = this.stage.steps * this.stage.tau; this.stage.reset(); }
    }
    this.flash = Math.max(0, this.flash - dt * 2.5);

    this._draw();
    if (this.frames % 6 === 0) { this._stats(); this._chart(); }
  }

  _avg() {
    const last = this.lengths.slice(-20);
    return last.length ? last.reduce((a, b) => a + b, 0) / last.length : 0;
  }

  _stats() {
    const n = this.lengths.length;
    this.el.episodes.textContent = n.toLocaleString();
    this.el.last.textContent = n ? this.lengths[n - 1] : '—';
    this.el.best.textContent = n ? this.best : '—';
    this.el.avg.textContent = n ? Math.round(this._avg()) : '—';
    this.el.status.textContent = this.solvedAt
      ? `Solved after ${this.solvedAt} episodes`
      : n ? `Learning · episode ${n}` : 'Untrained policy';
    this.el.status.classList.toggle('is-solved', !!this.solvedAt);
    this.el.run.textContent = `Balancing ${(this.stage.steps * this.stage.tau).toFixed(1)} s`;
    const pl = 1 - this.pRight;
    this.el.left.style.setProperty('--w', `${(pl * 100).toFixed(1)}%`);
    this.el.right.style.setProperty('--w', `${(this.pRight * 100).toFixed(1)}%`);
    this.el.pl.textContent = pl.toFixed(2);
    this.el.pr.textContent = this.pRight.toFixed(2);
  }

  _chart() {
    const c = this.el.chart, dpr = fitCanvas(c), ctx = c.getContext('2d'), W = c.width, H = c.height;
    if (!W || !H) return;
    ctx.clearRect(0, 0, W, H);
    const pad = 10 * dpr, max = 500, y = (v) => H - pad - (v / max) * (H - 2 * pad);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) { const yy = pad + ((H - 2 * pad) * i) / 4; ctx.moveTo(pad, yy); ctx.lineTo(W - pad, yy); }
    ctx.stroke();
    const vals = this.lengths.slice(-160), n = Math.max(vals.length, 60), bw = (W - 2 * pad) / n;
    vals.forEach((v, i) => {
      ctx.fillStyle = v >= SOLVED ? COLORS.lime : v > 200 ? COLORS.ice : COLORS.ember;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(pad + i * bw, y(v), Math.max(1, bw - dpr), H - pad - y(v));
    });
    ctx.globalAlpha = 1;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.strokeStyle = 'rgba(198,243,107,0.7)';
    ctx.beginPath(); ctx.moveTo(pad, y(SOLVED)); ctx.lineTo(W - pad, y(SOLVED)); ctx.stroke();
    ctx.setLineDash([]);
  }

  _draw() {
    const c = this.el.canvas, dpr = fitCanvas(c), ctx = c.getContext('2d'), W = c.width, H = c.height;
    if (!W || !H) return;
    const env = this.stage, [x, , theta] = env.state;
    const margin = 20 * dpr, scale = (W - 2 * margin) / (2 * env.xLimit + 0.8);
    const toPx = (m) => W / 2 + m * scale;
    this._toPx = toPx;
    const ground = H * 0.74, poleLen = 2 * env.halfLength * scale * 1.15;
    const cartW = 0.62 * scale, cartH = 0.28 * scale;

    ctx.clearRect(0, 0, W, H);

    // Track with end stops.
    const left = toPx(-env.xLimit), right = toPx(env.xLimit);
    ctx.fillStyle = 'rgba(255,106,61,0.08)';
    ctx.fillRect(0, ground - 4 * dpr, left, 8 * dpr);
    ctx.fillRect(right, ground - 4 * dpr, W - right, 8 * dpr);
    ctx.strokeStyle = 'rgba(238,240,246,0.22)';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(left, ground); ctx.lineTo(right, ground); ctx.stroke();
    ctx.lineWidth = dpr;
    for (let m = -2; m <= 2; m++) {
      ctx.beginPath(); ctx.moveTo(toPx(m), ground + 8 * dpr); ctx.lineTo(toPx(m), ground + 14 * dpr); ctx.stroke();
    }
    ctx.fillStyle = COLORS.ember;
    for (const px of [left, right]) ctx.fillRect(px - dpr, ground - 16 * dpr, 2 * dpr, 20 * dpr);

    // Cart.
    const wheelR = Math.max(3 * dpr, cartH * 0.24);
    const cx = toPx(x), top = ground - cartH - wheelR * 1.5;
    ctx.fillStyle = '#121622';
    ctx.strokeStyle = 'rgba(92,200,255,0.8)';
    ctx.lineWidth = 1.5 * dpr;
    roundRect(ctx, cx - cartW / 2, top, cartW, cartH, Math.min(6 * dpr, cartH / 3));
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#05060a';
    for (const o of [-0.28, 0.28]) {
      ctx.beginPath(); ctx.arc(cx + o * cartW, ground - wheelR, wheelR, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }

    // Allowed angle (±12°).
    const pivotY = top + 2 * dpr;
    ctx.strokeStyle = 'rgba(238,240,246,0.12)';
    ctx.setLineDash([3 * dpr, 5 * dpr]);
    for (const a of [-env.thetaLimit, env.thetaLimit]) {
      ctx.beginPath(); ctx.moveTo(cx, pivotY); ctx.lineTo(cx + Math.sin(a) * poleLen, pivotY - Math.cos(a) * poleLen); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Pole: ice when upright, ember near the limit.
    const danger = Math.min(1, Math.abs(theta) / env.thetaLimit);
    const tipX = cx + Math.sin(theta) * poleLen, tipY = pivotY - Math.cos(theta) * poleLen;
    const col = mix([92, 200, 255], [255, 106, 61], danger);
    ctx.strokeStyle = `rgb(${col})`;
    ctx.lineWidth = 7 * dpr;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx, pivotY); ctx.lineTo(tipX, tipY); ctx.stroke();
    const glow = ctx.createRadialGradient(tipX, tipY, 0, tipX, tipY, 28 * dpr);
    glow.addColorStop(0, `rgba(${col},0.9)`); glow.addColorStop(1, `rgba(${col},0)`);
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(tipX, tipY, 28 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(cx, pivotY, 4 * dpr, 0, Math.PI * 2); ctx.fill();

    // Your push.
    if (this.push.t > 0) {
      const dir = Math.sign(this.push.force), ax = cx - dir * (cartW / 2 + 14 * dpr), ay = top + cartH / 2;
      ctx.strokeStyle = COLORS.lime; ctx.fillStyle = COLORS.lime; ctx.lineWidth = 3 * dpr;
      ctx.beginPath(); ctx.moveTo(ax - dir * 34 * dpr, ay); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ax + dir * 4 * dpr, ay); ctx.lineTo(ax - dir * 8 * dpr, ay - 7 * dpr); ctx.lineTo(ax - dir * 8 * dpr, ay + 7 * dpr); ctx.fill();
    }

    if (this.flash > 0) {
      ctx.fillStyle = `rgba(255,106,61,${0.14 * this.flash})`;
      ctx.fillRect(0, 0, W, H);
    }
  }
}

function mix(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',');
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
