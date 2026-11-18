/**
 * main.js — boots the page and wires every motion system together:
 * preloader → intro, Lenis smooth scroll, scroll-linked 3D morphs, pinned
 * horizontal process, text reveals, custom cursor, magnetic buttons, tilt
 * cards, velocity marquee, counters and the live neural-network lab.
 */
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import { ParticleField } from './hero/ParticleField.js';
import { SHAPES } from './hero/shapes.js';
import { Lab } from './lab/Lab.js';

gsap.registerPlugin(ScrollTrigger);

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = matchMedia('(hover: hover) and (pointer: fine)').matches;
const lowPower = !finePointer || matchMedia('(max-width: 760px)').matches;

if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
window.scrollTo(0, 0);
$('#year').textContent = new Date().getFullYear();

/* ─────────────────────────── 1 · WebGL particle field ─────────────────────────── */
const particleCount = lowPower ? 14000 : 30000;
let field = null;
try {
  field = new ParticleField($('#webgl'), { count: particleCount, bloom: !lowPower, reducedMotion });
} catch (err) {
  console.warn('WebGL unavailable, using the static background.', err);
  document.documentElement.classList.add('no-webgl');
}
$('#stat-particles').dataset.count = String(particleCount);

window.addEventListener('pointermove', (e) => {
  field?.setPointer((e.clientX / innerWidth) * 2 - 1, -((e.clientY / innerHeight) * 2 - 1), true);
}, { passive: true });
document.documentElement.addEventListener('mouseleave', () => field?.setPointer(0, 0, false));

/* ─────────────────────────── 2 · Smooth scroll ─────────────────────────── */
const lenis = reducedMotion ? null : new Lenis({ duration: 1.15, easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
if (lenis) {
  lenis.on('scroll', ScrollTrigger.update);
  lenis.stop();
}
gsap.ticker.lagSmoothing(0);

$$('a[href^="#"]').forEach((a) => a.addEventListener('click', (e) => {
  const target = $(a.getAttribute('href'));
  if (!target) return;
  e.preventDefault();
  if (lenis) lenis.scrollTo(target, { duration: 1.8 });
  else target.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
}));

/* ─────────────────────────── 3 · Text splitting ─────────────────────────── */
function splitText(el, { chars = false } = {}) {
  const walk = (node) => {
    [...node.childNodes].forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((part) => {
          if (!part) return;
          if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(' ')); return; }
          const word = document.createElement('span');
          word.className = 'word';
          if (chars) {
            for (const ch of part) {
              const c = document.createElement('span');
              c.className = 'char'; c.textContent = ch;
              word.appendChild(c);
            }
          } else {
            const inner = document.createElement('span');
            inner.className = 'word__inner'; inner.textContent = part;
            word.appendChild(inner);
          }
          frag.appendChild(word);
        });
        child.replaceWith(frag);
      } else if (child.nodeType === Node.ELEMENT_NODE && child.tagName !== 'BR') {
        walk(child);
      }
    });
  };
  walk(el);
}

$$('[data-split]').forEach((el) => splitText(el, { chars: true }));
$$('[data-reveal]').forEach((el) => splitText(el));
$$('[data-scrub-words]').forEach((el) => splitText(el));

/* ─────────────────────────── 4 · Scroll choreography ─────────────────────────── */
// Created in page order: the pinned section first, so later triggers include its spacing.

// 4a · Process: pinned horizontal scroll
const track = $('.process__track');
const processDistance = () => track.scrollWidth - window.innerWidth;
const processTween = gsap.to(track, {
  x: () => -processDistance(),
  ease: 'none',
  scrollTrigger: {
    trigger: '#process',
    pin: true,
    start: 'top top',
    end: () => `+=${processDistance()}`,
    scrub: 0.8,
    invalidateOnRefresh: true,
    onUpdate: (self) => { $('#process-bar').style.transform = `scaleX(${self.progress})`; },
  },
});
const processST = processTween.scrollTrigger;

$$('.step__card').forEach((card, i) => {
  if (i === 0) return;
  gsap.from(card.children, {
    y: 60, opacity: 0, stagger: 0.06, duration: 0.9, ease: 'power3.out',
    scrollTrigger: { trigger: card, containerAnimation: processTween, start: 'left 75%', toggleActions: 'play none none reverse' },
  });
});

// 4b · Hero content drifts away as you leave
gsap.to('.hero__inner', {
  yPercent: -18, opacity: 0, ease: 'none',
  scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom 15%', scrub: true },
});

// 4c · Manifesto: words light up with scroll
gsap.fromTo('.manifesto__text .word', { opacity: 0.12 }, {
  opacity: 1, ease: 'none', stagger: 0.1,
  scrollTrigger: { trigger: '.manifesto__text', start: 'top 78%', end: 'bottom 42%', scrub: true },
});

// 4d · Headline reveals
$$('[data-reveal]').forEach((el) => {
  gsap.from(el.querySelectorAll('.word__inner'), {
    yPercent: 110, duration: 1.1, ease: 'expo.out', stagger: 0.035,
    scrollTrigger: { trigger: el, start: 'top 86%' },
  });
});
gsap.from('.lab__grid > *, .lab__bottom > *', {
  y: 70, opacity: 0, duration: 1.1, ease: 'power3.out', stagger: 0.1,
  scrollTrigger: { trigger: '.lab__grid', start: 'top 82%' },
});
gsap.from('.card', {
  y: 90, opacity: 0, duration: 1.1, ease: 'power3.out', stagger: 0.1,
  scrollTrigger: { trigger: '.cards', start: 'top 82%' },
});
gsap.from('.contact__cta, .contact__mail', {
  scale: 0.6, opacity: 0, duration: 1.2, ease: 'expo.out', stagger: 0.12,
  scrollTrigger: { trigger: '.contact__title', start: 'top 80%' },
});

// 4e · Counters (all numbers are true for this page)
const fmt = (v, f) => (f === 'k' && v >= 1000 ? `${Math.round(v / 1000)}k` : Math.round(v).toLocaleString());
$$('[data-count]').forEach((el) => {
  const target = Number(el.dataset.count), obj = { v: 0 };
  el.textContent = fmt(0, el.dataset.format);
  ScrollTrigger.create({
    trigger: el, start: 'top 90%', once: true,
    onEnter: () => gsap.to(obj, { v: target, duration: 1.8, ease: 'power3.out', onUpdate: () => { el.textContent = fmt(obj.v, el.dataset.format); } }),
  });
});

// 4f · Section HUD + progress bar
$$('[data-section]').forEach((sec) => ScrollTrigger.create({
  trigger: sec, start: 'top 55%', end: 'bottom 55%',
  onToggle: (self) => {
    if (!self.isActive) return;
    $('#hud-index').textContent = sec.dataset.section;
    $('#hud-title').textContent = sec.dataset.title;
  },
}));
ScrollTrigger.create({ start: 0, end: 'max', onUpdate: (self) => { $('#progress-bar').style.transform = `scaleX(${self.progress})`; } });

/* ─────────────────────────── 5 · Scroll → 3D morph mapping ─────────────────────────── */
const LOOKS = {
  hero:         { shape: SHAPES.sphere,    dim: 1,    x: 1.4, scale: 1 },
  manifesto:    { shape: SHAPES.galaxy,    dim: 0.7,  x: 0,   scale: 1.05 },
  data:         { shape: SHAPES.clusters,  dim: 1,    x: 1.8, scale: 0.85 },
  arch:         { shape: SHAPES.layers,    dim: 1,    x: 1.8, scale: 0.85 },
  train:        { shape: SHAPES.landscape, dim: 1,    x: 1.8, scale: 0.8 },
  deploy:       { shape: SHAPES.ring,      dim: 1,    x: 1.8, scale: 0.72 },
  lab:          { shape: SHAPES.galaxy,    dim: 0,    x: 0,   scale: 1.25 }, // hidden: keeps the lab readable
  capabilities: { shape: SHAPES.sphere,    dim: 0.35, x: 0,   scale: 1.1 },
  contact:      { shape: SHAPES.ring,      dim: 1,    x: 0,   scale: 1 },
};

const zones = [];
const zone = (from, to, trigger, start, end) => {
  const st = ScrollTrigger.create({ trigger, start, end });
  zones.push({ from, to, get start() { return st.start; }, get end() { return st.end; } });
};
zone(LOOKS.hero, LOOKS.manifesto, '#manifesto', 'top bottom', 'top 25%');
zone(LOOKS.manifesto, LOOKS.data, '#process', 'top bottom', 'top top');
const stepLooks = [LOOKS.data, LOOKS.arch, LOOKS.train, LOOKS.deploy];
for (let k = 1; k < stepLooks.length; k++) {
  const span = () => processST.end - processST.start;
  zones.push({
    from: stepLooks[k - 1], to: stepLooks[k],
    get start() { return processST.start + (span() * (k - 0.8)) / 3; },
    get end() { return processST.start + (span() * (k - 0.2)) / 3; },
  });
}
zone(LOOKS.deploy, LOOKS.lab, '#lab', 'top bottom', 'top 35%');
zone(LOOKS.lab, LOOKS.capabilities, '#capabilities', 'top 85%', 'top 25%');
zone(LOOKS.capabilities, LOOKS.contact, '#contact', 'top 65%', 'top 10%');

function updateMorph() {
  if (!field) return;
  const y = window.scrollY;
  let z = null;
  for (const candidate of zones) { if (y >= candidate.start) z = candidate; else break; }
  if (!z) { const h = LOOKS.hero; field.setLook({ from: h.shape, to: h.shape, t: 0, dim: h.dim, x: h.x, scale: h.scale }); return; }
  const t = clamp((y - z.start) / Math.max(1, z.end - z.start), 0, 1);
  const e = t * t * (3 - 2 * t);
  field.setLook({
    from: z.from.shape, to: z.to.shape, t: e,
    dim: lerp(z.from.dim, z.to.dim, e), x: lerp(z.from.x, z.to.x, e), scale: lerp(z.from.scale, z.to.scale, e),
  });
}

/* ─────────────────────────── 6 · Cursor, magnetic, tilt ─────────────────────────── */
const cursor = { x: -100, y: -100, rx: -100, ry: -100, el: $('.cursor') };
if (finePointer && !reducedMotion) {
  document.documentElement.classList.add('has-cursor');
  const dot = $('.cursor__dot'), ring = $('.cursor__ring'), label = $('.cursor__label');
  window.addEventListener('pointermove', (e) => {
    cursor.x = e.clientX; cursor.y = e.clientY;
    if (!cursor.shown) { cursor.rx = cursor.x; cursor.ry = cursor.y; cursor.shown = true; cursor.el.classList.add('is-visible'); }
  }, { passive: true });
  document.documentElement.addEventListener('mouseleave', () => { cursor.el.classList.remove('is-visible'); cursor.shown = false; });
  document.addEventListener('pointerover', (e) => {
    const hit = e.target.closest('[data-cursor], a, button, input, canvas');
    const text = hit?.closest('[data-cursor]')?.dataset.cursor || '';
    cursor.el.classList.toggle('is-hover', !!hit);
    cursor.el.classList.toggle('has-label', !!text && !e.target.closest('button, input'));
    if (text) label.textContent = text;
  });
  window.addEventListener('pointerdown', () => cursor.el.classList.add('is-down'));
  window.addEventListener('pointerup', () => cursor.el.classList.remove('is-down'));
  cursor.update = (dt) => {
    const k = 1 - Math.pow(0.0001, dt);
    cursor.rx = lerp(cursor.rx, cursor.x, k); cursor.ry = lerp(cursor.ry, cursor.y, k);
    dot.style.transform = `translate3d(${cursor.x}px, ${cursor.y}px, 0)`;
    ring.style.transform = `translate3d(${cursor.rx}px, ${cursor.ry}px, 0)`;
  };
}

if (finePointer && !reducedMotion) {
  $$('[data-magnetic]').forEach((el) => {
    const strength = Number(el.dataset.magneticStrength || 0.3);
    const inner = el.querySelector(':scope > span');
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      gsap.to(el, { x: dx * strength, y: dy * strength, duration: 0.5, ease: 'power3.out', overwrite: 'auto' });
      if (inner) gsap.to(inner, { x: dx * strength * 0.5, y: dy * strength * 0.5, duration: 0.5, ease: 'power3.out', overwrite: 'auto' });
    });
    el.addEventListener('pointerleave', () => {
      gsap.to(el, { x: 0, y: 0, duration: 1, ease: 'elastic.out(1, 0.35)', overwrite: 'auto' });
      if (inner) gsap.to(inner, { x: 0, y: 0, duration: 1, ease: 'elastic.out(1, 0.35)', overwrite: 'auto' });
    });
  });

  $$('[data-tilt]').forEach((card) => {
    card.addEventListener('pointermove', (e) => {
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      card.style.setProperty('--mx', `${px * 100}%`);
      card.style.setProperty('--my', `${py * 100}%`);
      gsap.to(card, { rotateY: (px - 0.5) * 14, rotateX: (0.5 - py) * 12, transformPerspective: 900, duration: 0.5, ease: 'power3.out', overwrite: 'auto' });
    });
    card.addEventListener('pointerleave', () => gsap.to(card, { rotateX: 0, rotateY: 0, duration: 1.1, ease: 'elastic.out(1, 0.45)', overwrite: 'auto' }));
  });
}

/* ─────────────────────────── 7 · Velocity marquee ─────────────────────────── */
const marquee = { el: $('.marquee__inner'), x: 0, dir: -1, skew: 0, visible: false };
new IntersectionObserver(([e]) => { marquee.visible = e.isIntersecting; }).observe($('.marquee'));
function updateMarquee(dt) {
  if (!marquee.visible) return;
  const v = lenis ? lenis.velocity : 0;
  if (Math.abs(v) > 0.5) marquee.dir = v > 0 ? -1 : 1;
  const speed = reducedMotion ? 0 : 50 + Math.min(Math.abs(v) * 28, 1400);
  marquee.x += marquee.dir * speed * dt;
  const half = marquee.el.scrollWidth / 2;
  if (marquee.x <= -half) marquee.x += half;
  if (marquee.x > 0) marquee.x -= half;
  marquee.skew = lerp(marquee.skew, reducedMotion ? 0 : clamp(-v * 0.5, -14, 14), 0.1);
  marquee.el.style.transform = `translate3d(${marquee.x}px, 0, 0) skewX(${marquee.skew}deg)`;
}

/* ─────────────────────────── 8 · The lab (from-scratch neural network) ─────────────────────────── */
const liveParams = $('#stat-live-params');
const lab = new Lab($('#lab'), {
  onParams: (n) => {
    liveParams.textContent = n.toLocaleString();
    gsap.fromTo(liveParams, { color: '#ffffff' }, { color: '#ff6a3d', duration: 0.8 });
  },
});

if (new URLSearchParams(location.search).has('debug')) window.__latentform = { field, lab, lenis, gsap, ScrollTrigger };

/* ─────────────────────────── 9 · Main loop ─────────────────────────── */
let last = 0;
gsap.ticker.add((time) => {
  const dt = Math.min(Math.max(time - last, 0), 0.1);
  last = time;
  lenis?.raf(time * 1000);
  updateMorph();
  if (field) {
    field.setVelocity(lenis ? lenis.velocity : 0);
    field.render(time);
  }
  lab.frame(time, dt);
  cursor.update?.(dt);
  updateMarquee(dt);
});

/* ─────────────────────────── 10 · Preloader → intro ─────────────────────────── */
function runPreloader() {
  const matrix = $('#preloader-matrix');
  const cells = Array.from({ length: 48 }, () => matrix.appendChild(document.createElement('span')));
  const rnd = () => (Math.random() * 2 - 1).toFixed(2);
  cells.forEach((c) => { c.textContent = rnd(); });
  const flicker = setInterval(() => cells.forEach((c) => {
    if (Math.random() < 0.3) { c.textContent = rnd(); c.classList.toggle('hot', Math.random() < 0.08); }
  }), 70);
  const counter = { v: 0 };
  return new Promise((resolve) => {
    gsap.to(counter, {
      v: 100, duration: reducedMotion ? 0.4 : 1.8, ease: 'power2.inOut',
      onUpdate: () => {
        $('#preloader-count').textContent = String(Math.round(counter.v)).padStart(3, '0');
        $('#preloader-bar').style.transform = `scaleX(${counter.v / 100})`;
      },
      onComplete: () => { clearInterval(flicker); resolve(); },
    });
  });
}

Promise.all([runPreloader(), document.fonts?.ready ?? Promise.resolve()]).then(() => {
  const tl = gsap.timeline({
    onComplete: () => {
      $('#preloader').remove();
      document.body.classList.remove('is-loading');
      lenis?.start();
      ScrollTrigger.refresh();
    },
  });
  tl.to('#preloader', { clipPath: 'inset(0 0 100% 0)', duration: 1.1, ease: 'expo.inOut' })
    .add(() => field?.startIntro(), 0.15)
    .from('.hero__title .char', { yPercent: 115, rotate: 7, duration: 1.3, ease: 'expo.out', stagger: 0.028 }, 0.5)
    .from('.hero [data-fade], .nav', { opacity: 0, y: 24, duration: 1, ease: 'power3.out', stagger: 0.07 }, 0.85);
});
