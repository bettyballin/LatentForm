# Latentform: neural engineering × immersive 3D

A one-page showcase site to show clients two things:

1. **You can build neural networks from scratch.** Three models from three fields of machine learning train **live in the
   visitor's browser**. Each one has hand-written backpropagation, and none uses TensorFlow, PyTorch or any other ML library:
   - **Classifier** (supervised learning): a multi-layer perceptron. Visitors change the architecture, add data points and
     watch the decision boundary and a 3D view of the network update in real time.
   - **Tiny GPT** (natural language): a character-level transformer with causal multi-head self-attention, LayerNorm, GELU
     and Adam. It trains in a Web Worker, writes text as it learns, completes prompts and shows its attention patterns.
   - **Balancer** (reinforcement learning): a policy network trained with REINFORCE learns to balance a pole on a cart by
     trial and error. Visitors can push the cart to test it.
2. **You can build advanced real-time 3D and motion.** The page has:
   - 30,000 GPU particles with custom GLSL shaders and bloom
   - six 3D shapes that morph into each other as you scroll
   - mouse repulsion, smooth scrolling and a pinned horizontal section
   - a magnetic cursor, 3D tilt cards and a marquee that reacts to scroll speed

The scene-by-scene plan, and a 90-second script for presenting the site to a client, are in **[STORYBOARD.md](STORYBOARD.md)**.

## Run it

```bash
npm start          # → http://localhost:5173   (zero dependencies, just Node ≥ 18)
```

ES modules don't load from `file://`, so open it through a server. Any static server works (`python3 -m http.server`, too).

## Prove the math

```bash
npm test
```

This runs **numerical gradient checks**. For every weight and bias in the perceptron and in the transformer (attention,
LayerNorm, GELU, embeddings), the analytic gradient from `backward()` is compared with `(L(w+ε) − L(w−ε)) / 2ε`. Other
tests confirm that attention is causal and that each model actually learns: the perceptron classifies the spiral, the GPT's
loss falls far below chance, and the agent balances the pole. It's useful when a technical client asks "did you really
write the backprop?"

## Where things live

| File | What it does |
|---|---|
| `js/nn/Network.js` | ★ The from-scratch neural network: init, forward, backward, softmax/cross-entropy, Adam/SGD, mini-batches |
| `js/nn/datasets.js` | Seeded toy datasets: spiral, rings, XOR, moons |
| `js/nn/Transformer.js` | ★ The tiny GPT: embeddings, causal multi-head attention, LayerNorm, GELU MLP, all backward passes, Adam, sampling |
| `js/nn/corpus.js` | The text the GPT learns from (original writing; replace it to change the model's voice) |
| `js/rl/CartPole.js` | ★ Cart-pole physics and the REINFORCE agent (reuses `Network.js` as the policy) |
| `js/lab/Lab.js` | Classifier: live training loop, decision-boundary renderer, controls, loss chart, live source-code panel |
| `js/lab/Network3D.js` | 3D view of the classifier network: activations, weight signals, travelling pulses |
| `js/lang/gpt.worker.js` | Trains the GPT in a Web Worker so the page stays smooth |
| `js/lang/LanguageLab.js` | Language tab: live samples, prompt completion, attention inspector, loss curve |
| `js/rl/ControlLab.js` | Balancer tab: real-time stage, push interaction, policy readout, episode chart |
| `js/lab/ui.js` | Small shared helpers: charts, segmented buttons, the live source-code highlighter |
| `js/hero/shaders.js` | GLSL: simplex noise, 6-way morph blending, mouse repulsion, soft additive points |
| `js/hero/shapes.js` | Morph targets: sphere, galaxy, data clusters, network layers, loss landscape (+ a gradient-descent trail), ring |
| `js/hero/ParticleField.js` | Three.js renderer + bloom composer, eased "look" state |
| `js/main.js` | Wiring: preloader, Lenis, ScrollTrigger choreography, scroll → morph mapping, cursor, magnetic, tilt, marquee |
| `css/style.css` | Design tokens at the top (`:root`), then one block per section |

Three.js 0.160, GSAP 3.12.5 and Lenis 1.1.13 load from jsDelivr at pinned versions (see the import map in `index.html`).

## Make it yours

- **Name and copy:** "Latentform", the headlines and the section text are plain HTML in `index.html`.
- **Contact:** replace `betty.ballin@yahoo.de` in `index.html` (it appears twice).
- **Colours and fonts:** the variables at the top of `css/style.css`. The particle colours are `uColorA/B/C` in
  `js/hero/ParticleField.js`, and the lab class colours are `CLASS_HEX` in `js/lab/Network3D.js`.
- **3D story:** `LOOKS` in `js/main.js` maps each section to a shape, brightness, x-offset and scale.
  To add a shape, write a generator in `shapes.js` (a 7th shape also needs another attribute and weight in `shaders.js`).
- **Lab defaults:** `this.state` in `js/lab/Lab.js` sets the classifier's starting dataset, layers, neurons, activation,
  learning rate and speed. `CONFIG` and `TRAIN` in `js/lang/gpt.worker.js` size the GPT, and `ControlLab.js` sets the
  balancer's pace.
- **What the GPT writes:** edit `js/nn/corpus.js`. More text gives more variety, but it takes longer to learn.

## Deploy

It's a static site, so upload the folder to Netlify, Vercel, GitHub Pages, Cloudflare Pages or any web host.
`server.mjs`, `package.json` and `tests/` aren't needed in production.

## Debugging

Open the page with `?debug` to get `window.__latentform` (`field`, `lab`, `models`, `selectModel`, `lenis`, `gsap`,
`ScrollTrigger`) in the console. For example, `__latentform.lab.trainOnce(1900)` runs 100 classifier epochs instantly, and
`__latentform.selectModel('language')` switches tabs.

## Performance and accessibility

- On phones and touch devices it uses 14k particles and no bloom. Device pixel ratio is capped. Only the selected model trains
  and renders, and only while it's on screen. The language and balancer models start the first time their tab is opened.
- `prefers-reduced-motion` turns off smooth scroll, the custom cursor, the magnetic and tilt effects and the marquee motion, and calms the shader.
- If WebGL isn't available, the page falls back to a static gradient background, and all the content still works.
