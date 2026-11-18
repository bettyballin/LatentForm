/**
 * shapes.js — the six morph targets of the hero particle field.
 * Each generator returns a Float32Array of xyz positions (world units; the
 * visible half-height at z = 0 is ~3.7). Index order is shuffled so that
 * morphs swirl instead of sliding.
 */
import { createRng, randn } from '../nn/Network.js';

export const SHAPES = { sphere: 0, galaxy: 1, clusters: 2, layers: 3, landscape: 4, ring: 5 };

export function buildShapes(count, seed = 7) {
  const rng = createRng(seed);
  return [sphere, galaxy, clusters, layers, landscape, ring].map((fn) => shuffle(fn(count, rng), rng));
}

function shuffle(arr, rng) {
  for (let i = arr.length / 3 - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    for (let k = 0; k < 3; k++) {
      const t = arr[i * 3 + k]; arr[i * 3 + k] = arr[j * 3 + k]; arr[j * 3 + k] = t;
    }
  }
  return arr;
}

const put = (out, i, x, y, z) => { out[i * 3] = x; out[i * 3 + 1] = y; out[i * 3 + 2] = z; };

function rotX(x, y, z, a) { const c = Math.cos(a), s = Math.sin(a); return [x, y * c - z * s, y * s + z * c]; }
function rotY(x, y, z, a) { const c = Math.cos(a), s = Math.sin(a); return [x * c + z * s, y, -x * s + z * c]; }
function rotZ(x, y, z, a) { const c = Math.cos(a), s = Math.sin(a); return [x * c - y * s, x * s + y * c, z]; }

/** 0 · Sphere core: Fibonacci shell + a sparse interior. */
function sphere(n, rng) {
  const out = new Float32Array(n * 3);
  const R = 2.3, shell = Math.floor(n * 0.86), golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    if (i < shell) {
      const y = 1 - (2 * (i + 0.5)) / shell, r = Math.sqrt(1 - y * y), th = i * golden;
      const rr = R + randn(rng) * 0.035;
      put(out, i, Math.cos(th) * r * rr, y * rr, Math.sin(th) * r * rr);
    } else {
      const u = rng() * 2 - 1, th = rng() * Math.PI * 2, rr = R * 0.88 * Math.cbrt(rng()), s = Math.sqrt(1 - u * u);
      put(out, i, s * Math.cos(th) * rr, u * rr, s * Math.sin(th) * rr);
    }
  }
  return out;
}

/** 1 · Three-armed spiral galaxy, tilted toward the viewer. */
function galaxy(n, rng) {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (rng() < 0.08) {
      const r = Math.abs(randn(rng)) * 0.3, th = rng() * Math.PI * 2, u = rng() * 2 - 1, s = Math.sqrt(1 - u * u);
      x = s * Math.cos(th) * r; y = u * r * 0.6; z = s * Math.sin(th) * r;
    } else {
      const r = 0.15 + Math.pow(rng(), 1.4) * 3.3;
      const ang = ((i % 3) / 3) * Math.PI * 2 + r * 1.25;
      const spread = 0.05 + r * 0.16;
      const jitter = () => Math.pow(rng(), 2.2) * (rng() < 0.5 ? -1 : 1) * spread * 2;
      x = Math.cos(ang) * r + jitter();
      z = Math.sin(ang) * r + jitter();
      y = randn(rng) * 0.07 * (1 + r * 0.15);
    }
    [x, y, z] = rotX(x, y, z, 1.05);
    [x, y, z] = rotZ(x, y, z, 0.35);
    put(out, i, x, y, z);
  }
  return out;
}

/** 2 · Data: labelled Gaussian clusters in a sparse cloud. */
function clusters(n, rng) {
  const out = new Float32Array(n * 3);
  const centers = [[-1.6, 0.9, 0.3], [0.2, 1.4, -0.6], [1.5, 0.6, 0.5], [-1.0, -0.9, -0.4], [0.6, -0.5, 0.9], [1.8, -1.3, -0.5], [-0.3, 0.1, -1.4]];
  const sizes = [0.3, 0.25, 0.27, 0.32, 0.22, 0.24, 0.28];
  for (let i = 0; i < n; i++) {
    if (rng() < 0.1) {
      put(out, i, (rng() * 2 - 1) * 2.6, (rng() * 2 - 1) * 2.0, (rng() * 2 - 1) * 1.5);
    } else {
      const c = i % centers.length, s = sizes[c], [cx, cy, cz] = centers[c];
      put(out, i, cx + randn(rng) * s * 1.15, cy + randn(rng) * s, cz + randn(rng) * s * 0.9);
    }
  }
  return out;
}

/** 3 · Architecture: five neuron layers, with particles strung along the connections. */
function layers(n, rng) {
  const out = new Float32Array(n * 3);
  const grids = [[2, 2], [4, 4], [5, 5], [4, 4], [3, 1]];
  const xs = [-2.6, -1.3, 0, 1.3, 2.6], sp = 0.52;
  const nodes = [], byLayer = [];
  grids.forEach(([rows, cols], l) => {
    byLayer.push([]);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      byLayer[l].push(nodes.length);
      nodes.push([xs[l], (r - (rows - 1) / 2) * sp, (c - (cols - 1) / 2) * sp]);
    }
  });
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (rng() < 0.62) {
      const [nx, ny, nz] = nodes[Math.floor(rng() * nodes.length)];
      x = nx + randn(rng) * 0.055; y = ny + randn(rng) * 0.055; z = nz + randn(rng) * 0.055;
    } else {
      const l = Math.floor(rng() * (grids.length - 1));
      const a = nodes[pick(byLayer[l])], b = nodes[pick(byLayer[l + 1])], t = rng();
      x = a[0] + (b[0] - a[0]) * t; y = a[1] + (b[1] - a[1]) * t + randn(rng) * 0.008; z = a[2] + (b[2] - a[2]) * t;
    }
    [x, y, z] = rotY(x, y, z, -0.55);
    [x, y, z] = rotX(x, y, z, 0.18);
    put(out, i, x, y, z);
  }
  return out;
}

/** Height of the toy loss surface: a bowl with two valleys, a peak and ripples. */
export function lossHeight(x, z) {
  return 0.06 * (x * x + z * z)
    - 1.1 * Math.exp(-((x - 0.9) ** 2 + (z + 0.5) ** 2) / 0.6)
    - 0.7 * Math.exp(-((x + 1.3) ** 2 + (z - 0.9) ** 2) / 0.4)
    + 0.8 * Math.exp(-((x + 0.6) ** 2 + (z + 1.2) ** 2) / 0.5)
    + 0.12 * Math.sin(x * 2.2) * Math.cos(z * 2.0);
}

/** 4 · Training: a loss landscape, plus the trail of a gradient-descent run. */
function landscape(n, rng) {
  const out = new Float32Array(n * 3);
  const pathCount = Math.floor(n * 0.05), gridN = n - pathCount;
  const side = Math.ceil(Math.sqrt(gridN)), S = 3.3, Y = 1.25, tilt = 0.55;

  // Simulate gradient descent with momentum on the same surface.
  const path = [];
  let px = -2.6, pz = 2.4, vx = 0, vz = 0;
  for (let k = 0; k < 220; k++) {
    const e = 1e-3;
    const gx = (lossHeight(px + e, pz) - lossHeight(px - e, pz)) / (2 * e);
    const gz = (lossHeight(px, pz + e) - lossHeight(px, pz - e)) / (2 * e);
    vx = 0.86 * vx - 0.05 * gx; vz = 0.86 * vz - 0.05 * gz;
    px += vx; pz += vz;
    path.push([px, lossHeight(px, pz), pz]);
  }

  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (i < gridN) {
      x = ((i % side) / (side - 1) * 2 - 1) * S + (rng() - 0.5) * 0.02;
      z = (Math.floor(i / side) / (side - 1) * 2 - 1) * S + (rng() - 0.5) * 0.02;
      y = lossHeight(x, z);
    } else {
      const [qx, qy, qz] = path[Math.floor(rng() * path.length)];
      x = qx + randn(rng) * 0.03; y = qy + 0.07 + Math.abs(randn(rng)) * 0.03; z = qz + randn(rng) * 0.03;
    }
    [x, y, z] = rotX(x, (y - 0.15) * Y, z, tilt);
    put(out, i, x, y - 0.2, z);
  }
  return out;
}

/** 5 · Deploy / contact: a glowing torus portal with a sparse halo. */
function ring(n, rng) {
  const out = new Float32Array(n * 3);
  const R = 2.75;
  for (let i = 0; i < n; i++) {
    const u = rng() * Math.PI * 2;
    let x, y, z;
    if (rng() < 0.78) {
      const v = rng() * Math.PI * 2, r = 0.1 + Math.abs(randn(rng)) * 0.05, rr = R + r * Math.cos(v);
      x = rr * Math.cos(u); y = rr * Math.sin(u); z = r * Math.sin(v);
    } else {
      const rr = R + randn(rng) * 0.45;
      x = rr * Math.cos(u); y = rr * Math.sin(u); z = randn(rng) * 0.25;
    }
    [x, y, z] = rotX(x, y, z, 0.28);
    put(out, i, x, y, z);
  }
  return out;
}
