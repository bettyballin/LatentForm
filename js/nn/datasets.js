/**
 * datasets.js — toy 2D classification problems, all in the square [-1, 1]².
 * Every generator is seeded, so the same dataset looks the same on every visit.
 */
import { createRng, randn } from './Network.js';

function spiral(rng, n = 300, classes = 3) {
  const X = [], Y = [];
  const per = Math.floor(n / classes);
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < per; i++) {
      const r = (i / per) * 0.95 + 0.05;
      const t = (c * 2 * Math.PI) / classes + r * 8 + randn(rng) * 0.12;
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      Y.push(c);
    }
  }
  return { X, Y, classes };
}

function rings(rng, n = 260) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const inner = i % 2 === 0;
    const r = inner ? Math.sqrt(rng()) * 0.42 : 0.62 + rng() * 0.3;
    const t = rng() * Math.PI * 2;
    X.push([r * Math.cos(t) + randn(rng) * 0.03, r * Math.sin(t) + randn(rng) * 0.03]);
    Y.push(inner ? 0 : 1);
  }
  return { X, Y, classes: 2 };
}

function xor(rng, n = 260) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    let x = rng() * 2 - 1, y = rng() * 2 - 1;
    // keep a small margin around the axes so the four quadrants read clearly
    x += x > 0 ? 0.06 : -0.06; y += y > 0 ? 0.06 : -0.06;
    X.push([x * 0.92, y * 0.92]);
    Y.push(x * y > 0 ? 0 : 1);
  }
  return { X, Y, classes: 2 };
}

function moons(rng, n = 260) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const upper = i % 2 === 0;
    const t = rng() * Math.PI;
    let x, y;
    if (upper) { x = Math.cos(t); y = Math.sin(t); }
    else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
    // centre and scale into [-1, 1]
    x = (x - 0.5) * 0.62 + randn(rng) * 0.06;
    y = (y - 0.25) * 0.9 + randn(rng) * 0.06;
    X.push([x, y]);
    Y.push(upper ? 0 : 1);
  }
  return { X, Y, classes: 2 };
}

function blank() {
  return { X: [], Y: [], classes: 2 };
}

export const DATASETS = {
  spiral: { label: 'Spiral', make: (seed = 3) => spiral(createRng(seed)) },
  rings:  { label: 'Rings',  make: (seed = 5) => rings(createRng(seed)) },
  xor:    { label: 'XOR',    make: (seed = 7) => xor(createRng(seed)) },
  moons:  { label: 'Moons',  make: (seed = 9) => moons(createRng(seed)) },
  blank:  { label: 'Blank',  make: () => blank() },
};
