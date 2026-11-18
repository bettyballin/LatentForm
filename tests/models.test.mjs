/**
 * Tests for the tiny GPT and the reinforcement-learning agent.
 *   • every hand-written transformer gradient (attention, LayerNorm, GELU, embeddings)
 *     matches a numerical estimate
 *   • the transformer's loss drops well below chance on the corpus
 *   • REINFORCE learns to balance the pole
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Transformer } from '../js/nn/Transformer.js';
import { CORPUS } from '../js/nn/corpus.js';
import { CartPole, ReinforceAgent } from '../js/rl/CartPole.js';

test('transformer backprop matches numerical gradients for all parameters', () => {
  const m = new Transformer({ vocabSize: 6, blockSize: 5, dModel: 8, nHead: 2, nLayer: 2, seed: 3, dtype: Float64Array });
  const idx = [1, 4, 2, 0, 5], tgt = [4, 2, 0, 5, 3];
  m.zeroGrad();
  m.loss(idx, tgt);
  m.backward(tgt, 1 / idx.length);

  const eps = 1e-5;
  let worst = 0, checked = 0;
  for (const p of m.params) {
    for (let i = 0; i < p.w.length; i++) {
      const orig = p.w[i];
      p.w[i] = orig + eps; const lp = m.loss(idx, tgt);
      p.w[i] = orig - eps; const lm = m.loss(idx, tgt);
      p.w[i] = orig;
      const numeric = (lp - lm) / (2 * eps), analytic = p.g[i];
      worst = Math.max(worst, Math.abs(numeric - analytic) / Math.max(1e-7, Math.abs(numeric) + Math.abs(analytic)));
      checked++;
    }
  }
  assert.ok(checked > 1500, `only ${checked} parameters checked`);
  assert.ok(worst < 1e-4, `max relative error ${worst.toExponential(2)} too large`);
});

test('attention is causal: no position looks at the future', () => {
  const m = new Transformer({ vocabSize: 6, blockSize: 6, dModel: 8, nHead: 2, nLayer: 1, seed: 1 });
  const n = 6, att = m.attentionMaps([0, 1, 2, 3, 4, 5]);
  for (let h = 0; h < 2; h++) for (let t = 0; t < n; t++) {
    let sum = 0;
    for (let u = 0; u < n; u++) { const w = att[(h * n + t) * n + u]; if (u > t) assert.equal(w, 0); sum += w; }
    assert.ok(Math.abs(sum - 1) < 1e-5);
  }
});

test('tiny GPT learns the corpus (loss falls far below chance)', () => {
  const chars = [...new Set(CORPUS)].sort();
  const stoi = new Map(chars.map((c, i) => [c, i]));
  const data = Uint8Array.from([...CORPUS].map((c) => stoi.get(c)));
  const m = new Transformer({ vocabSize: chars.length, blockSize: 32, dModel: 32, nHead: 4, nLayer: 2, seed: 1 });
  let ema = null;
  for (let s = 0; s < 600; s++) {
    const l = m.trainStep(data, { batchSize: 4, lr: 6e-3 });
    ema = ema === null ? l : 0.9 * ema + 0.1 * l;
  }
  const chance = Math.log(chars.length); // ≈ 3.4 nats
  assert.ok(ema < 2.0, `loss ${ema.toFixed(2)} vs chance ${chance.toFixed(2)}`);
});

test('REINFORCE agent learns to balance the pole', () => {
  const env = new CartPole(1), agent = new ReinforceAgent({ hidden: 32, lr: 0.01, seed: 1 });
  const lengths = [];
  for (let ep = 0; ep < 600; ep++) {
    const e = agent.runEpisode(env);
    agent.update(e);
    lengths.push(e.length);
    const last = lengths.slice(-20);
    if (last.length === 20 && last.reduce((a, b) => a + b, 0) / 20 >= 475) return;
  }
  assert.fail(`not solved after 600 episodes (last 20 avg ${lengths.slice(-20).reduce((a, b) => a + b, 0) / 20})`);
});
