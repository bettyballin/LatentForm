/**
 * Proves the hand-written backpropagation is mathematically correct.
 *
 * Gradient checking: for every single weight and bias, compare the analytic
 * gradient from backward() with a numerical estimate
 *     (L(w + ε) − L(w − ε)) / 2ε
 * They must agree to ~1e-7 relative error.
 *
 * Run:  npm test   (or: node --test tests/)
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Network, createRng } from '../js/nn/Network.js';
import { DATASETS } from '../js/nn/datasets.js';

function gradientCheck(activation) {
  const net = new Network({ layers: [2, 5, 4, 3], activation, seed: 42 });
  const rng = createRng(11);
  const X = Array.from({ length: 8 }, () => [rng() * 2 - 1, rng() * 2 - 1]);
  const Y = Array.from({ length: 8 }, () => Math.floor(rng() * 3));

  net.zeroGrad();
  for (let s = 0; s < X.length; s++) net.accumulate(X[s], Y[s]);

  const eps = 1e-5;
  let worst = 0;
  for (let l = 0; l < net.L; l++) {
    for (const [P, G] of [[net.W[l], net.gW[l]], [net.b[l], net.gb[l]]]) {
      for (let k = 0; k < P.length; k++) {
        const orig = P[k];
        P[k] = orig + eps; const lp = net.lossOf(X, Y);
        P[k] = orig - eps; const lm = net.lossOf(X, Y);
        P[k] = orig;
        const numeric = (lp - lm) / (2 * eps);
        const analytic = G[k] / X.length;
        const rel = Math.abs(numeric - analytic) / Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
        worst = Math.max(worst, rel);
      }
    }
  }
  return worst;
}

for (const activation of ['tanh', 'sigmoid', 'sin']) {
  test(`backprop gradients match numerical gradients (${activation})`, () => {
    const worst = gradientCheck(activation);
    assert.ok(worst < 1e-6, `max relative error ${worst.toExponential(2)} too large`);
  });
}

test('network learns the 3-class spiral (>95% accuracy)', () => {
  const { X, Y, classes } = DATASETS.spiral.make();
  const net = new Network({ layers: [2, 16, 16, classes], activation: 'tanh', seed: 1 });
  let last;
  for (let e = 0; e < 400; e++) last = net.trainEpoch(X, Y, { lr: 0.02, batchSize: 16 });
  const acc = net.accuracyOf(X, Y);
  assert.ok(acc > 0.95, `accuracy only ${(acc * 100).toFixed(1)}% (last loss ${last.loss.toFixed(3)})`);
});

test('softmax output is a valid probability distribution', () => {
  const net = new Network({ layers: [2, 6, 4], activation: 'relu', seed: 3 });
  const p = net.forward([0.3, -0.7]);
  const sum = p.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12);
  assert.ok([...p].every((v) => v > 0 && v < 1));
});
