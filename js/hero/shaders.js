/**
 * GLSL for the hero particle field.
 *
 * Every particle stores its position in six different shapes (morph targets).
 * The vertex shader blends them with weights driven by scroll position, adds
 * simplex-noise flow (stronger mid-transition), and pushes particles away from
 * the mouse. The fragment shader draws a soft additive glow.
 */

// 3D simplex noise — Ian McEwan, Ashima Arts / Stefan Gustavson (MIT).
const SIMPLEX_NOISE = /* glsl */ `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 105.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
`;

export const particleVertex = /* glsl */ `
uniform float uTime;
uniform float uWeights[6];
uniform vec3  uMouse;
uniform float uMouseForce;
uniform float uSize;
uniform float uPixelRatio;
uniform float uIntro;
uniform float uNoiseAmp;
uniform float uDim;
uniform vec3  uColorA;
uniform vec3  uColorB;
uniform vec3  uColorC;

attribute vec3 aS1;
attribute vec3 aS2;
attribute vec3 aS3;
attribute vec3 aS4;
attribute vec3 aS5;
attribute vec4 aRand;

varying vec3  vColor;
varying float vAlpha;

${SIMPLEX_NOISE}

void main() {
  // 1. Blend the six morph targets (position = shape 0).
  vec3 p = position * uWeights[0] + aS1 * uWeights[1] + aS2 * uWeights[2]
         + aS3 * uWeights[3] + aS4 * uWeights[4] + aS5 * uWeights[5];

  // 0 when settled on one shape, 0.5 halfway between two.
  float wMax = max(max(max(uWeights[0], uWeights[1]), max(uWeights[2], uWeights[3])),
                   max(uWeights[4], uWeights[5]));
  float morphing = 1.0 - wMax;

  // 2. Curl-ish flow from three decorrelated simplex fields.
  float t = uTime * 0.12;
  vec3 q = p * 0.55;
  vec3 flow = vec3(
    snoise(q + vec3(t, 0.0, 0.0)),
    snoise(q + vec3(0.0, t, 0.0) + 19.1),
    snoise(q + vec3(0.0, 0.0, t) + 47.3)
  );
  p += flow * (uNoiseAmp + morphing * 1.5 * (0.4 + aRand.x));

  // 3. Intro: burst out from the centre, each particle with its own delay.
  float intro = clamp(uIntro * 1.6 - aRand.w * 0.6, 0.0, 1.0);
  intro = 1.0 - pow(1.0 - intro, 3.0);
  p *= intro;

  vec4 world = modelMatrix * vec4(p, 1.0);

  // 4. Mouse repulsion in the screen plane, with a little depth kick.
  vec2 dm = world.xy - uMouse.xy;
  float dist = length(dm);
  float push = exp(-dist * dist * 0.9) * uMouseForce;
  world.xy += (dm / max(dist, 1e-3)) * push * 0.9;
  world.z += push * 0.6;

  vec4 mv = viewMatrix * world;
  gl_Position = projectionMatrix * mv;

  float size = uSize * (0.35 + aRand.y * 0.9);
  gl_PointSize = size * uPixelRatio * (9.0 / -mv.z);

  // 5. Colour: ice ↔ ember by noise, lime glow where the mouse pushes.
  float k = smoothstep(-0.45, 0.75, flow.x * 0.8 + aRand.z - 0.55);
  vColor = mix(uColorA, uColorB, k);
  vColor = mix(vColor, uColorC, clamp(push * 1.6, 0.0, 1.0));
  vAlpha = (0.1 + 0.4 * aRand.z) * uDim * intro;
}
`;

export const particleFragment = /* glsl */ `
varying vec3  vColor;
varying float vAlpha;

void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = 1.0 - smoothstep(0.0, 0.5, d);
  a = pow(a, 1.6);
  if (a < 0.01) discard;
  gl_FragColor = vec4(vColor, a * vAlpha);
  #include <colorspace_fragment>
}
`;
