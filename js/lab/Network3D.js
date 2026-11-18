/**
 * Network3D.js — renders the *live* network from Network.js in 3D.
 *
 *   neurons      glow with their activation (ember = positive, ice = negative)
 *   connections  brighten with the signal they carry, |w · a|
 *   pulses       travel along the strongest connections → the forward pass, visible
 *
 * Each layer is a ring of neurons; drag to rotate.
 */
import * as THREE from 'three';

export const CLASS_HEX = ['#ff6a3d', '#5cc8ff', '#c6f36b'];
const WARM = new THREE.Color('#ff6a3d');
const COOL = new THREE.Color('#5cc8ff');
const CLASS_COLORS = CLASS_HEX.map((h) => new THREE.Color(h));
const BG = 0x080a10;
const clampPitch = (p) => Math.max(-0.7, Math.min(0.7, p));

const glowVertex = /* glsl */ `
attribute float aSize;
attribute vec3 aColor;
uniform float uPR;
varying vec3 vColor;
void main() {
  vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = aSize * uPR * (10.0 / -mv.z);
}`;

const glowFragment = /* glsl */ `
varying vec3 vColor;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float halo = 1.0 - smoothstep(0.0, 0.5, d);
  float core = 1.0 - smoothstep(0.08, 0.2, d);
  vec3 col = vColor * (halo * halo * 0.9 + core * 1.1) + vec3(core * 0.18);
  float a = clamp(halo * halo + core, 0.0, 1.0);
  if (a < 0.01) discard;
  gl_FragColor = vec4(col, a);
  #include <colorspace_fragment>
}`;

export class Network3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.pr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(this.pr);
    this.renderer.setClearColor(BG, 1);
    this.renderer.domElement.setAttribute('aria-label', '3D view of the live neural network');
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.glowMaterial = () => new THREE.ShaderMaterial({
      vertexShader: glowVertex, fragmentShader: glowFragment,
      uniforms: { uPR: { value: this.pr } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });

    // Drag-to-rotate with inertia (touch-friendly: vertical page scroll still works).
    this.yaw = 0; this.pitch = 0; this.vYaw = 0; this.vPitch = 0; this.dragging = false;
    const el = this.renderer.domElement;
    el.style.touchAction = 'pan-y';
    el.addEventListener('pointerdown', (e) => { this.dragging = true; this.lx = e.clientX; this.ly = e.clientY; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      this.vYaw = (e.clientX - this.lx) * 0.006; this.vPitch = (e.clientY - this.ly) * 0.004;
      this.yaw += this.vYaw; this.pitch = clampPitch(this.pitch + this.vPitch);
      this.lx = e.clientX; this.ly = e.clientY;
    });
    const end = () => { this.dragging = false; };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this._fitCamera();
  }

  _fitCamera() {
    if (!this.width) return;
    const fitW = (this.width + 1.2) / (2 * Math.tan((this.camera.fov * Math.PI) / 360) * this.camera.aspect);
    const fitH = (this.height + 1.4) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    const z = Math.max(fitW, fitH, 5);
    this.camera.position.set(0, z * 0.12, z);
    this.camera.lookAt(0, 0, 0);
  }

  /** (Re)build geometry for a list of layer sizes, e.g. [2, 12, 12, 3]. */
  setArchitecture(sizes) {
    for (const obj of [this.neurons, this.lines, this.pulses]) {
      if (!obj) continue;
      this.group.remove(obj); obj.geometry.dispose(); obj.material.dispose();
    }
    this.sizes = sizes.slice();
    const nL = sizes.length;
    const spacing = Math.min(1.7, 7.2 / (nL - 1));
    const radius = (n) => (n <= 1 ? 0 : Math.max(0.36, (n * 0.42) / (2 * Math.PI)));

    // Neurons
    const total = sizes.reduce((a, b) => a + b, 0);
    this.offsets = [];
    this.pos = new Float32Array(total * 3);
    let idx = 0, maxR = 0;
    sizes.forEach((n, l) => {
      this.offsets.push(idx);
      const r = radius(n), x = (l - (nL - 1) / 2) * spacing;
      maxR = Math.max(maxR, r);
      for (let j = 0; j < n; j++, idx++) {
        const th = (2 * Math.PI * j) / n + l * 0.35;
        this.pos[idx * 3] = x;
        this.pos[idx * 3 + 1] = r * Math.cos(th);
        this.pos[idx * 3 + 2] = r * Math.sin(th);
      }
    });
    this.width = (nL - 1) * spacing;
    this.height = maxR * 2;
    this._fitCamera();

    const nGeo = new THREE.BufferGeometry();
    nGeo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.nColor = new Float32Array(total * 3);
    this.nSize = new Float32Array(total).fill(14);
    nGeo.setAttribute('aColor', new THREE.BufferAttribute(this.nColor, 3));
    nGeo.setAttribute('aSize', new THREE.BufferAttribute(this.nSize, 1));
    this.neurons = new THREE.Points(nGeo, this.glowMaterial());
    this.neurons.renderOrder = 2;

    // Connections (one line segment per weight)
    let E = 0;
    for (let l = 0; l < nL - 1; l++) E += sizes[l] * sizes[l + 1];
    this.E = E;
    this.eSrc = new Uint32Array(E); this.eDst = new Uint32Array(E);
    this.eLayer = new Uint8Array(E); this.eW = new Uint32Array(E);
    this.eStrength = new Float32Array(E);
    const lPos = new Float32Array(E * 6);
    this.lColor = new Float32Array(E * 6);
    let e = 0;
    for (let l = 0; l < nL - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      for (let j = 0; j < nOut; j++) for (let i = 0; i < nIn; i++, e++) {
        const s = this.offsets[l] + i, d = this.offsets[l + 1] + j;
        this.eSrc[e] = s; this.eDst[e] = d; this.eLayer[e] = l; this.eW[e] = j * nIn + i;
        for (let k = 0; k < 3; k++) { lPos[e * 6 + k] = this.pos[s * 3 + k]; lPos[e * 6 + 3 + k] = this.pos[d * 3 + k]; }
      }
    }
    const lGeo = new THREE.BufferGeometry();
    lGeo.setAttribute('position', new THREE.BufferAttribute(lPos, 3));
    lGeo.setAttribute('color', new THREE.BufferAttribute(this.lColor, 3));
    this.lines = new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));

    // Pulses travelling along strong connections
    this.P = Math.min(160, Math.max(40, Math.floor(E * 0.35)));
    this.pPos = new Float32Array(this.P * 3);
    this.pColor = new Float32Array(this.P * 3);
    this.pSize = new Float32Array(this.P);
    this.pEdge = new Uint32Array(this.P);
    this.pT = new Float32Array(this.P);
    this.pSpeed = new Float32Array(this.P);
    for (let p = 0; p < this.P; p++) { this.pEdge[p] = Math.floor(Math.random() * E); this.pT[p] = Math.random(); this.pSpeed[p] = 0.6 + Math.random() * 0.8; }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(this.pPos, 3));
    pGeo.setAttribute('aColor', new THREE.BufferAttribute(this.pColor, 3));
    pGeo.setAttribute('aSize', new THREE.BufferAttribute(this.pSize, 1));
    this.pulses = new THREE.Points(pGeo, this.glowMaterial());
    this.pulses.renderOrder = 3;

    this.group.add(this.lines, this.neurons, this.pulses);
    this.disp = new Float32Array(total);
  }

  /** Read weights + the activations of the last forward pass from a Network. */
  sync(net) {
    if (!this.sizes || net.sizes.join() !== this.sizes.join()) this.setArchitecture(net.sizes);
    const { L, sizes } = net;
    const c = new THREE.Color();

    for (let l = 0; l <= L; l++) {
      const a = net.a[l], off = this.offsets[l];
      let norm = 1;
      if (l === 0) norm = 1.2;
      else if (l < L) { let m = 0; for (let j = 0; j < sizes[l]; j++) m = Math.max(m, Math.abs(a[j])); norm = Math.max(1, m); }
      for (let j = 0; j < sizes[l]; j++) {
        const v = Math.max(-1, Math.min(1, a[j] / norm));
        this.disp[off + j] = v;
        const i = off + j;
        if (l === L) {
          c.copy(CLASS_COLORS[j % CLASS_COLORS.length]).multiplyScalar(0.12 + 0.88 * a[j]);
          this.nSize[i] = 18 + 30 * a[j];
        } else if (l === 0) {
          c.setRGB(1, 1, 1).multiplyScalar(0.3 + 0.7 * Math.abs(v));
          this.nSize[i] = 16 + 14 * Math.abs(v);
        } else {
          const m = Math.abs(v);
          c.copy(v >= 0 ? WARM : COOL).multiplyScalar(0.08 + 0.92 * m);
          c.r += 0.05; c.g += 0.05; c.b += 0.06;
          this.nSize[i] = 11 + 24 * m;
        }
        this.nColor[i * 3] = c.r; this.nColor[i * 3 + 1] = c.g; this.nColor[i * 3 + 2] = c.b;
      }
    }

    // Edge brightness = |w · a_in|, normalised per layer.
    const maxPerLayer = new Float32Array(L);
    for (let e = 0; e < this.E; e++) {
      const s = Math.abs(net.W[this.eLayer[e]][this.eW[e]] * this.disp[this.eSrc[e]]);
      this.eStrength[e] = s;
      if (s > maxPerLayer[this.eLayer[e]]) maxPerLayer[this.eLayer[e]] = s;
    }
    for (let e = 0; e < this.E; e++) {
      const w = net.W[this.eLayer[e]][this.eW[e]];
      const b = Math.pow(this.eStrength[e] / (maxPerLayer[this.eLayer[e]] || 1), 0.8);
      this.eStrength[e] = b;
      const base = w >= 0 ? WARM : COOL, k = 0.025 + 0.8 * b;
      const o = e * 6;
      this.lColor[o] = base.r * k; this.lColor[o + 1] = base.g * k; this.lColor[o + 2] = base.b * k;
      this.lColor[o + 3] = base.r * k * 0.5; this.lColor[o + 4] = base.g * k * 0.5; this.lColor[o + 5] = base.b * k * 0.5;
    }

    this.neurons.geometry.attributes.aColor.needsUpdate = true;
    this.neurons.geometry.attributes.aSize.needsUpdate = true;
    this.lines.geometry.attributes.color.needsUpdate = true;
  }

  frame(time, dt) {
    if (!this.pulses) return;

    // Pulses: advance, and respawn on a connection picked in proportion to its strength.
    for (let p = 0; p < this.P; p++) {
      this.pT[p] += dt * this.pSpeed[p];
      if (this.pT[p] >= 1) {
        let e = Math.floor(Math.random() * this.E);
        for (let tries = 0; tries < 12 && Math.random() > this.eStrength[e]; tries++) e = Math.floor(Math.random() * this.E);
        this.pEdge[p] = e; this.pT[p] = 0; this.pSpeed[p] = 0.6 + Math.random() * 0.8;
      }
      const e = this.pEdge[p], t = this.pT[p], s = this.eSrc[e] * 3, d = this.eDst[e] * 3;
      for (let k = 0; k < 3; k++) this.pPos[p * 3 + k] = this.pos[s + k] + (this.pos[d + k] - this.pos[s + k]) * t;
      const o = e * 6, glow = 1.8;
      this.pColor[p * 3] = this.lColor[o] * glow + 0.04;
      this.pColor[p * 3 + 1] = this.lColor[o + 1] * glow + 0.04;
      this.pColor[p * 3 + 2] = this.lColor[o + 2] * glow + 0.04;
      this.pSize[p] = (5 + 9 * this.eStrength[e]) * Math.sin(Math.PI * t);
    }
    const pa = this.pulses.geometry.attributes;
    pa.position.needsUpdate = true; pa.aColor.needsUpdate = true; pa.aSize.needsUpdate = true;

    // Rotation: idle sway + inertial drag.
    if (!this.dragging) {
      this.yaw += this.vYaw; this.pitch = clampPitch(this.pitch + this.vPitch);
      this.vYaw *= 0.92; this.vPitch *= 0.92;
    }
    this.group.rotation.set(0.12 + this.pitch, -0.55 + this.yaw + Math.sin(time * 0.25) * 0.3, 0);

    this.renderer.render(this.scene, this.camera);
  }
}
