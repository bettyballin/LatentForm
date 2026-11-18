# Latentform: neural engineering × immersive 3D

A one-page showcase site to show clients two things:

1. **You can build neural networks from scratch.** A multi-layer perceptron with hand-written backpropagation and the Adam
   optimiser trains **live in the visitor's browser**. Visitors can change the architecture, add data points and watch the
   decision boundary and a 3D view of the network update in real time. No TensorFlow, no PyTorch, no ML libraries.
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

This runs a **numerical gradient check**. For every weight and bias, the analytic gradient from `backward()` is compared with
`(L(w+ε) − L(w−ε)) / 2ε`, and they must agree to a relative error below 1e-6. A second test confirms the network learns
the 3-class spiral to more than 95% accuracy. It's useful when a technical client asks "did you really write the backprop?"

## Where things live

| File | What it does |
|---|---|
| `js/nn/Network.js` | ★ The from-scratch neural network: init, forward, backward, softmax/cross-entropy, Adam/SGD, mini-batches |
| `js/nn/datasets.js` | Seeded toy datasets: spiral, rings, XOR, moons |
| `js/lab/Lab.js` | Live training loop, decision-boundary renderer, controls, loss chart, live source-code panel |
| `js/lab/Network3D.js` | 3D view of the *same* network: activations, weight signals, travelling pulses |
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
- **Lab defaults:** `this.state` in `js/lab/Lab.js` sets the starting dataset, layers, neurons, activation, learning rate and speed.

## Deploy

It's a static site, so upload the folder to Netlify, Vercel, GitHub Pages, Cloudflare Pages or any web host.
`server.mjs`, `package.json` and `tests/` aren't needed in production.

## Debugging

Open the page with `?debug` to get `window.__latentform` (`field`, `lab`, `lenis`, `gsap`, `ScrollTrigger`) in the console. For example,
`__latentform.lab.trainOnce(1900)` runs 100 epochs instantly.

## Performance and accessibility

- On phones and touch devices it uses 14k particles and no bloom. Device pixel ratio is capped. The lab only trains and
  renders while it's on screen.
- `prefers-reduced-motion` turns off smooth scroll, the custom cursor, the magnetic and tilt effects and the marquee motion, and calms the shader.
- If WebGL isn't available, the page falls back to a static gradient background, and all the content still works.
