/**
 * CartPole.js — reinforcement learning from scratch.
 *
 * Environment: the classic cart-pole (Barto, Sutton & Anderson, 1983). Push the
 * cart left or right to keep the pole upright. +1 reward for every step it stays up.
 *
 * Agent: a policy network (Network.js, 4 → 32 → 2) trained with REINFORCE,
 * the original policy-gradient algorithm. No labels: it learns from its own
 * trial and error.
 */
import { Network, createRng } from '../nn/Network.js';

export class CartPole {
  constructor(seed = 1) {
    this.rng = createRng(seed);
    Object.assign(this, {
      gravity: 9.8, massCart: 1, massPole: 0.1, halfLength: 0.5, forceMag: 10, tau: 0.02,
      xLimit: 2.4, thetaLimit: (12 * Math.PI) / 180, maxSteps: 500,
    });
    this.reset();
  }

  reset() {
    const r = () => (this.rng() - 0.5) * 0.1;
    this.state = [r(), r(), r(), r()]; // x, ẋ, θ, θ̇
    this.steps = 0;
    this.done = false;
    return this.state;
  }

  /** Advance one physics step (Euler, 20 ms). `extra` is an outside push in newtons. */
  step(action, extra = 0) {
    const [x, xDot, theta, thetaDot] = this.state;
    const { gravity, massCart, massPole, halfLength, tau } = this;
    const total = massCart + massPole, poleMassLength = massPole * halfLength;
    const force = (action === 1 ? this.forceMag : -this.forceMag) + extra;
    const cos = Math.cos(theta), sin = Math.sin(theta);
    const temp = (force + poleMassLength * thetaDot * thetaDot * sin) / total;
    const thetaAcc = (gravity * sin - cos * temp) / (halfLength * (4 / 3 - (massPole * cos * cos) / total));
    const xAcc = temp - (poleMassLength * thetaAcc * cos) / total;

    this.state = [x + tau * xDot, xDot + tau * xAcc, theta + tau * thetaDot, thetaDot + tau * thetaAcc];
    this.steps++;
    this.done = Math.abs(this.state[0]) > this.xLimit || Math.abs(this.state[2]) > this.thetaLimit;
    return 1;
  }
}

export class ReinforceAgent {
  constructor({ hidden = 32, seed = 1, lr = 0.01, gamma = 0.99 } = {}) {
    this.net = new Network({ layers: [4, hidden, 2], activation: 'tanh', seed });
    this.rng = createRng(seed + 101);
    this.lr = lr;
    this.gamma = gamma;
    this.obs = new Float64Array(4);
  }

  /** Scale the raw state to roughly [-1, 1] for the network. */
  observe([x, xDot, theta, thetaDot]) {
    const o = this.obs;
    o[0] = x / 2.4; o[1] = xDot / 3; o[2] = theta / 0.21; o[3] = thetaDot / 3;
    return o;
  }

  /** Sample an action from the policy π(a | s). Returns { action, pRight }. */
  act(state, greedy = false) {
    const p = this.net.forward(this.observe(state));
    const action = greedy ? (p[1] > p[0] ? 1 : 0) : (this.rng() < p[1] ? 1 : 0);
    return { action, pRight: p[1] };
  }

  /** Play one full episode with the current policy and remember what happened. */
  runEpisode(env) {
    env.reset();
    const states = [], actions = [];
    while (!env.done && env.steps < env.maxSteps) {
      const s = env.state.slice();
      const { action } = this.act(s);
      states.push(s); actions.push(action);
      env.step(action);
    }
    return { states, actions, length: actions.length };
  }

  /**
   * REINFORCE: make every action more likely in proportion to the (normalised)
   * discounted return that followed it.  ∇J = E[ G_t · ∇ log π(a_t | s_t) ]
   */
  update(episode) {
    const { states, actions } = episode, n = actions.length;
    const returns = new Float64Array(n);
    let G = 0;
    for (let t = n - 1; t >= 0; t--) { G = 1 + this.gamma * G; returns[t] = G; }
    let mean = 0;
    for (let t = 0; t < n; t++) mean += returns[t];
    mean /= n;
    let std = 0;
    for (let t = 0; t < n; t++) std += (returns[t] - mean) ** 2;
    std = Math.sqrt(std / n) + 1e-8;

    this.net.zeroGrad();
    for (let t = 0; t < n; t++) {
      this.net.forward(this.observe(states[t]));
      this.net.backward(actions[t], (returns[t] - mean) / std); // weight −log π(a) by the advantage
    }
    this.net.step(n, { lr: this.lr });
  }
}
