/** ui.js — small helpers shared by the three model labs. */

export const COLORS = { ember: '#ff6a3d', ice: '#5cc8ff', lime: '#c6f36b', violet: '#c9a7ff', muted: '#8a8fa3' };

/** Keep a canvas's backing store at CSS size × device pixel ratio. Returns the ratio. */
export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
  return dpr;
}

/** Segmented button group: renders buttons and calls onPick(value). */
export function segmented(group, values, current, onPick, labels = values) {
  group.innerHTML = values.map((v, i) => `<button type="button" data-value="${v}" aria-pressed="${String(v) === String(current)}">${labels[i]}</button>`).join('');
  group.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    group.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    onPick(b.dataset.value);
  });
}

/** Line chart with an optional gradient fill; series = [{ values, max, color, fill }]. */
export function drawLines(canvas, series, { guide } = {}) {
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  if (!W || !H) return;
  const dpr = W / (canvas.clientWidth || W), pad = 10 * dpr;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = dpr;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) { const y = pad + ((H - 2 * pad) * i) / 4; ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); }
  ctx.stroke();
  if (guide) {
    const y = H - pad - guide.value * (H - 2 * pad);
    ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.strokeStyle = guide.color; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke(); ctx.setLineDash([]);
  }
  for (const s of series) {
    const v = s.values;
    if (v.length < 2) continue;
    const x = (i) => pad + (i / (v.length - 1)) * (W - 2 * pad);
    const y = (val) => H - pad - Math.min(1, val / s.max) * (H - 2 * pad);
    ctx.beginPath();
    v.forEach((val, i) => (i ? ctx.lineTo(x(i), y(val)) : ctx.moveTo(x(i), y(val))));
    ctx.strokeStyle = s.color; ctx.lineWidth = 2 * dpr; ctx.lineJoin = 'round'; ctx.stroke();
    if (s.fill) {
      ctx.lineTo(x(v.length - 1), H - pad); ctx.lineTo(x(0), H - pad); ctx.closePath();
      const g = ctx.createLinearGradient(0, pad, 0, H);
      g.addColorStop(0, s.fill); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g; ctx.fill();
    }
  }
}

/** Keep at most `max` points by dropping every other one (for long training curves). */
export function pushCapped(arr, value, max = 360) {
  arr.push(value);
  if (arr.length > max) for (let i = 1; i < arr.length; i++) arr.splice(i, 1);
  return arr;
}

export function dedent(src) {
  const lines = src.split('\n');
  const indents = lines.slice(1).filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length);
  const min = indents.length ? Math.min(...indents) : 0;
  return [lines[0], ...lines.slice(1).map((l) => l.slice(min))].join('\n');
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Tiny JS syntax highlighter (comments, strings, keywords, numbers, calls). */
export function highlight(src) {
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

/** Render a function's live source into a <code> element. */
export function showSource(el, fn, caption) {
  el.innerHTML = highlight(`// ${caption}, read at runtime via Function.prototype.toString()\n${dedent(fn.toString())}`);
}
