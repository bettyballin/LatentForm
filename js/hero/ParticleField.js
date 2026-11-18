/**
 * ParticleField.js — the fixed WebGL background behind the whole page.
 *
 * 30k particles (fewer on mobile) with six morph targets, simplex flow, mouse
 * repulsion and an UnrealBloom pass. The page tells it what to look like via
 * setLook(); everything is eased here so motion always feels physical.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { buildShapes } from './shapes.js';
import { particleVertex, particleFragment } from './shaders.js';

const SHAPE_COUNT = 6;
const lerp = (a, b, t) => a + (b - a) * t;

export class ParticleField {
  constructor(canvas, { count = 30000, bloom = true, reducedMotion = false } = {}) {
    this.canvas = canvas;
    this.count = count;
    this.reducedMotion = reducedMotion;

    const renderer = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' }));
    // three r160 writes the clear colour into the (linear) composer target already
    // sRGB-encoded, and OutputPass encodes again. Hand it the linear values tagged
    // as sRGB so the final pixel is exactly #05060a.
    const bg = new THREE.Color(0x05060a);
    renderer.setClearColor(new THREE.Color().setRGB(bg.r, bg.g, bg.b, THREE.SRGBColorSpace), 1);
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 1.75);
    renderer.setPixelRatio(this.pixelRatio);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.set(0, 0, 9);

    // Geometry: shape 0 lives in `position`, shapes 1–5 in aS1…aS5.
    const shapes = buildShapes(count);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(shapes[0], 3));
    for (let s = 1; s < SHAPE_COUNT; s++) geo.setAttribute(`aS${s}`, new THREE.BufferAttribute(shapes[s], 3));
    const rand = new Float32Array(count * 4);
    for (let i = 0; i < rand.length; i++) rand[i] = Math.random();
    geo.setAttribute('aRand', new THREE.BufferAttribute(rand, 4));

    this.uniforms = {
      uTime: { value: 0 },
      uWeights: { value: [1, 0, 0, 0, 0, 0] },
      uMouse: { value: new THREE.Vector3(99, 99, 0) },
      uMouseForce: { value: 0 },
      uSize: { value: count > 20000 ? 4.2 : 5.5 },
      uPixelRatio: { value: this.pixelRatio },
      uIntro: { value: 0 },
      uNoiseAmp: { value: 0.1 },
      uDim: { value: 1 },
      uColorA: { value: new THREE.Color('#5cc8ff') },
      uColorB: { value: new THREE.Color('#ff6a3d') },
      uColorC: { value: new THREE.Color('#c6f36b') },
    };

    const material = new THREE.ShaderMaterial({
      vertexShader: particleVertex,
      fragmentShader: particleFragment,
      uniforms: this.uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(geo, material);
    this.points.frustumCulled = false; // positions are computed in the shader
    this.group = new THREE.Group();
    this.group.add(this.points);
    this.scene.add(this.group);

    // Post-processing: bloom makes dense regions glow.
    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    if (bloom) {
      this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.55, 0.4, 0.32);
      this.composer.addPass(this.bloom);
    }
    this.composer.addPass(new OutputPass());

    // Pointer, look and timing state.
    this.ndc = new THREE.Vector2(0, 0);
    this.ndcSmooth = new THREE.Vector2(0, 0);
    this.pointerActive = false;
    this._ray = new THREE.Vector3();
    this.look = { weights: [1, 0, 0, 0, 0, 0], dim: 1, x: 0, y: 0, scale: 1 };
    this.target = { weights: [1, 0, 0, 0, 0, 0], dim: 1, x: 0, y: 0, scale: 1 };
    this.velocity = 0;
    this.introStart = null;
    this.lastTime = 0;

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    if (!w || !h) { this.hidden = true; return; } // e.g. loaded in a hidden tab; avoid zero-size framebuffers
    this.hidden = false;
    this.aspect = w / h;
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.pixelRatio);
    this.composer.setSize(w, h);
    this.camera.aspect = this.aspect;
    this.camera.updateProjectionMatrix();
  }

  /** Normalised device coords (-1..1). */
  setPointer(nx, ny, active = true) {
    this.ndc.set(nx, ny);
    this.pointerActive = active;
  }

  /** Scroll velocity (px/frame from Lenis): shakes the field a little. */
  setVelocity(v) {
    this.velocity = v;
  }

  /**
   * @param {{from:number, to:number, t:number, dim:number, x:number, scale:number}} look
   * from/to are shape indices; t is the blend between them.
   */
  setLook({ from, to, t, dim, x, scale }) {
    const w = this.target.weights;
    w.fill(0);
    w[from] += 1 - t;
    w[to] += t;
    // On portrait screens, move "side" layouts above the text instead.
    const portrait = this.aspect < 0.9;
    this.target.x = portrait ? 0 : x;
    this.target.y = portrait && x !== 0 ? 1.5 : 0;
    this.target.dim = dim;
    this.target.scale = scale * (portrait ? Math.max(0.5, this.aspect * 0.95) : 1);
  }

  startIntro() {
    this.introStart = performance.now();
  }

  render(time) {
    if (this.hidden) { this.resize(); if (this.hidden) return; }
    const dt = Math.min(Math.max(time - this.lastTime, 0), 0.1);
    this.lastTime = time;
    const u = this.uniforms;
    const k = 1 - Math.pow(0.0015, dt); // frame-rate independent easing

    // Ease the look toward its target.
    const L = this.look, T = this.target;
    for (let i = 0; i < SHAPE_COUNT; i++) L.weights[i] = lerp(L.weights[i], T.weights[i], k);
    L.dim = lerp(L.dim, T.dim, k);
    L.x = lerp(L.x, T.x, k * 0.8);
    L.y = lerp(L.y, T.y, k * 0.8);
    L.scale = lerp(L.scale, T.scale, k * 0.8);
    u.uWeights.value = L.weights;
    u.uDim.value = L.dim;
    this.group.position.set(L.x, L.y, 0);
    this.group.scale.setScalar(L.scale);

    // Intro burst.
    if (this.introStart !== null) {
      const p = this.reducedMotion ? 1 : (performance.now() - this.introStart) / 2600;
      u.uIntro.value = Math.min(1, p);
    }

    // Pointer → world position on the z = 0 plane.
    this.ndcSmooth.lerp(this.ndc, 1 - Math.pow(0.002, dt));
    this._ray.set(this.ndcSmooth.x, this.ndcSmooth.y, 0.5).unproject(this.camera).sub(this.camera.position).normalize();
    const dist = -this.camera.position.z / this._ray.z;
    u.uMouse.value.copy(this.camera.position).addScaledVector(this._ray, dist);
    u.uMouseForce.value = lerp(u.uMouseForce.value, this.pointerActive && !this.reducedMotion ? 1 : 0, k * 0.6);

    // Gentle sway + parallax toward the cursor.
    const sway = this.reducedMotion ? 0 : Math.sin(time * 0.12) * 0.22;
    this.group.rotation.y = lerp(this.group.rotation.y, sway + this.ndcSmooth.x * 0.35, k);
    this.group.rotation.x = lerp(this.group.rotation.x, -this.ndcSmooth.y * 0.18, k);

    // Scroll speed makes the flow more turbulent.
    const base = this.reducedMotion ? 0.03 : 0.1;
    u.uNoiseAmp.value = lerp(u.uNoiseAmp.value, base + Math.min(Math.abs(this.velocity) * 0.012, 0.45), k);
    u.uTime.value = this.reducedMotion ? time * 0.25 : time;

    this.composer.render(dt);
  }
}
