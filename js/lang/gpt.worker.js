/**
 * gpt.worker.js — trains the tiny GPT off the main thread so the page stays at 60 fps.
 *
 * in:  { type: 'play' | 'pause' | 'reset' }
 *      { type: 'generate', prompt, temperature, length, id }
 *      { type: 'attention', text }
 * out: { type: 'ready', vocab, params, config, corpusChars }
 *      { type: 'progress', step, loss, tokensPerSec, sample? }
 *      { type: 'generated', id, prompt, text }
 *      { type: 'attention', text, layers, heads, att }
 */
import { Transformer } from '../nn/Transformer.js';
import { CORPUS } from '../nn/corpus.js';
import { createRng } from '../nn/Network.js';

const CONFIG = { blockSize: 32, dModel: 32, nHead: 4, nLayer: 2 };
const TRAIN = { batchSize: 4, lr: 6e-3 };
const SAMPLE_PROMPTS = ['the ', 'a model ', 'learning ', 'attention ', 'data '];

const chars = [...new Set(CORPUS)].sort();
const stoi = new Map(chars.map((c, i) => [c, i]));
const encode = (s) => [...s.toLowerCase()].filter((c) => stoi.has(c)).map((c) => stoi.get(c));
const decode = (ids) => ids.map((i) => chars[i]).join('');
const data = Uint8Array.from(encode(CORPUS));

let model, step, playing = false, seed = 1, lastSample = 0, promptIndex = 0;
const rng = createRng(42);

function reset() {
  model = new Transformer({ vocabSize: chars.length, ...CONFIG, seed: seed++ });
  step = 0; lastSample = 0;
  postMessage({ type: 'ready', vocab: chars.length, params: model.paramCount, config: CONFIG, corpusChars: CORPUS.length, chars: chars.join('') });
}

function sample() {
  const prompt = SAMPLE_PROMPTS[promptIndex++ % SAMPLE_PROMPTS.length];
  return prompt + decode(model.generate(encode(prompt), 140, { temperature: 0.8, rng }));
}

function loop() {
  if (playing) {
    const t0 = performance.now();
    let n = 0, loss = 0;
    while (performance.now() - t0 < 50) { loss += model.trainStep(data, TRAIN); n++; step++; }
    const dt = (performance.now() - t0) / 1000;
    const msg = { type: 'progress', step, loss: loss / n, tokensPerSec: Math.round((n * TRAIN.batchSize * CONFIG.blockSize) / dt) };
    if (performance.now() - lastSample > 2200) { msg.sample = sample(); lastSample = performance.now(); }
    postMessage(msg);
  }
  setTimeout(loop, playing ? 0 : 120);
}

onmessage = ({ data: m }) => {
  if (m.type === 'play') playing = true;
  else if (m.type === 'pause') playing = false;
  else if (m.type === 'reset') reset();
  else if (m.type === 'generate') {
    const ctx = encode(m.prompt);
    const start = ctx.length ? ctx : encode('the ');
    const text = decode(model.generate(start, m.length || 200, { temperature: m.temperature || 0.8, rng }));
    postMessage({ type: 'generated', id: m.id, prompt: decode(start), text });
  } else if (m.type === 'attention') {
    const ids = encode(m.text).slice(-CONFIG.blockSize);
    if (ids.length < 2) return;
    const att = model.attentionMaps(ids);
    postMessage({ type: 'attention', text: decode(ids), layers: CONFIG.nLayer, heads: CONFIG.nHead, att }, [att.buffer]);
  }
};

reset();
loop();
