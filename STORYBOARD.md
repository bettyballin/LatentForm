# LATENTFORM — Storyboard

**What it is:** A one-page showcase site for a fictional studio/freelancer, *Latentform* ("neural engineering × immersive 3D").
It has one job: in about 90 seconds, convince a client that the person behind it can **build neural networks from scratch**
and **make advanced real-time 3D and motion work**.

**How it proves it:**

1. **A neural network written from scratch**, with no TensorFlow, PyTorch or ML libraries. It trains live in the visitor's browser,
   and they can change the network, add data and watch it learn.
2. **A GPU particle system** (30,000 particles, custom GLSL shaders, bloom). It changes shape as you scroll
   and reacts to the mouse.
3. **Motion throughout:** smooth scrolling, pinned horizontal scrolling, scroll-scrubbed text, a magnetic cursor, 3D card tilt
   and a marquee that reacts to scroll speed.

**Visual language:** near-black (`#05060A`) background, *ember* orange (`#FF6A3D`) and *ice* blue (`#5CC8FF`) accents,
*lime* (`#C6F36B`) as a third data colour. Fonts: Space Grotesk (headings), Instrument Serif italic (accent words),
JetBrains Mono (labels, numbers, code). Fine film grain over everything.

---

## Scene map

```
 scroll ─────────────────────────────────────────────────────────────────────────────▶
 ┌────────┬────────┬───────────┬────────────────────────────┬──────────┬────────┬──────────┬────────┐
 │  00    │  01    │   02      │   03  PROCESS (pinned)     │   04     │  05    │ marquee  │  06    │
 │ LOADER │  HERO  │ MANIFESTO │ data→arch→train→deploy     │   LAB    │ CAPAB. │ + stats  │CONTACT │
 ├────────┼────────┼───────────┼────────────────────────────┼──────────┼────────┼──────────┼────────┤
 │3D:  –  │ sphere │  galaxy   │ clusters→layers→landscape→ │ galaxy   │ sphere │ sphere   │  ring  │
 │        │  core  │           │ ring                       │ (dimmed) │ (dim)  │          │ portal │
 └────────┴────────┴───────────┴────────────────────────────┴──────────┴────────┴──────────┴────────┘
```

The 3D particle field is **one fixed canvas behind the whole page**. Sections scroll over it. Each scroll range blends
the particles between two of six **morph targets**. The blend is tied directly to scroll position, so scrolling back
plays it in reverse.

---

## 00 · Preloader — "Initializing weights"

```
┌──────────────────────────────────────────────┐
│  LATENTFORM                                  │
│                                              │
│   -0.42  0.17 -1.03  0.88  0.05 -0.61        │
│    0.33 -0.94  0.12  0.70 -0.27  0.49        │   ← a "weight matrix" of random numbers
│   -0.08  0.56 -0.71  0.21  0.93 -0.15        │      that keeps changing
│                                              │
│  initializing weights ............... 064%   │
└──────────────────────────────────────────────┘
```

- **Motion:** the counter runs 000 → 100% over about 1.6 s (eased). The matrix numbers flicker. The overlay then wipes upward (`clip-path`).
- **Purpose:** sets the ML tone in the first second and hides font and shader compilation.

## 01 · Hero — "Intelligence, *hand-built.*"

```
┌──────────────────────────────────────────────┐
│ LATENTFORM      Work  Model  Capabilities   [Let's talk]
│                                              │
│              . · ˙ ·  ·  ˙ · .               │
│          ·  ˙   ◉ particle core ◉  ˙  ·      │   ← 30k particles on a sphere,
│   Intelligence,    ·  ˙    ·                 │      shifting with noise, glowing (bloom)
│   hand-built.   ˙  ·  ˙  ·   ˙               │
│              ˙ ·  ·  ˙ ·                     │
│  I design neural networks from first         │
│  principles and turn them into real-time...  │
│  [ Train the model ↓ ]  [ Start a project ]  │
│                                  scroll ┃    │
└──────────────────────────────────────────────┘
```

- **3D:** shape 0, the **sphere core**: a Fibonacci-distributed shell plus a few particles inside. Simplex noise in the vertex shader keeps it shifting. The intro tween grows it outward from the centre.
- **Mouse:** particles are **pushed away from the cursor** (projected into world space), glowing lime near it. The whole field **tilts toward the mouse** with inertia.
- **Motion:** headline letters rise from a mask, one after another. Subtext and buttons fade up. The scroll cue line loops.
  As the hero scrolls away, its content drifts up and fades (scrubbed to scroll).
- **Copy:** *Intelligence, hand-built.* / "I design neural networks from first principles and turn them into real-time,
  interactive experiences: the math, the model and the motion, under one roof."

## 02 · Manifesto — words light up as you read

```
┌──────────────────────────────────────────────┐
│ 02 — Manifesto                               │
│                                              │
│  Most products rent their intelligence.      │   ← words fade from 15% to 100%
│  I write mine — from the first matrix        │      opacity in step with scroll
│  multiply to the last pixel on screen. ░░░░  │
│  ░░░░ ░░░░░ ░░░░ ░░ ░░░ ░░░░ ░░░░░ ░░░░░     │
│                      ✺ galaxy behind (dim)   │
└──────────────────────────────────────────────┘
```

- **3D:** sphere → shape 1, a **three-armed spiral galaxy**, tilted. Extra turbulence is added *during* the change, so it looks like a transformation rather than a straight cross-fade.
- **Motion:** each word's opacity is scrubbed to scroll. Faster scrolling also makes the particle noise stronger (Lenis velocity sets `uNoiseAmp`).

## 03 · Process — pinned horizontal scroll, four steps

```
┌──────────────────────────────────────────────┐
│ 03 — How a model is born        ▬▬▬▬▬░░░░░░  │  ← progress bar
│ ┌──────────────┐                             │
│ │ 01           │            ∴∵∴  ∴∵          │  01 DATA  → particles form labelled clusters
│ │ Data         │          ∵∴∵∴  ∵∴∵∴         │  02 ARCH  → five neuron layers + connections
│ │ Collect &    │            ∴∵     ∴∵∴       │  03 TRAIN → a 3D loss landscape (valleys/peaks)
│ │ shape the    │                             │  04 DEPLOY→ a glowing ring/portal
│ │ signal…      │   ◀── vertical scroll moves the track sideways ──
│ └──────────────┘                             │
└──────────────────────────────────────────────┘
```

- **Mechanics:** the section **pins** while vertical scrolling moves a 4-panel track horizontally (GSAP ScrollTrigger, `scrub`).
- **3D:** the particle group moves **to the right half** (text stays on the left). Each panel blends to its own shape:
  - 01 **Data**: six Gaussian clusters ("labelled examples")
  - 02 **Architecture**: five parallel neuron planes, with particles strung along the connections between them
  - 03 **Training**: a **loss landscape** surface with a global valley, local minima and ripples
  - 04 **Deploy**: a torus ring
- **Purpose:** explains the craft to a non-technical client, and every step is drawn in 3D.

## 04 · The Lab — a live neural network written from scratch ★ centrepiece

```
┌───────────────────────────────────────────────────────────────────┐
│ 04 — The Model                                                    │
│ A neural network, written from scratch. Train it live.            │
│ ┌──────────────────────────┐  ┌─────────────────────────────────┐ │
│ │ ░░▓▓▓▓░░  ● ● ●          │  │   ◉        ◉        ◉           │ │
│ │ ░▓▓▓▓░░ ●  decision   ●  │  │  ◉ ╲╱╲╱╲╱ ◉ ╲╱╲╱╲╱ ◉   ◉        │ │ ← 3D view of the SAME
│ │ ░░▓▓░  ● boundary heat●  │  │   ◉ ╱╲╱╲╱ ◉ ╱╲╱╲╱ ◉   ◉        │ │   network: neurons glow
│ │ ░░░  ●  map, retrains ●  │  │  pulses run along strong weights│ │   with activations
│ │ (hover = probe, click =  │  └─────────────────────────────────┘ │
│ │  add a point)            │  EPOCH 412  LOSS 0.083  ACC 98.7%    │
│ └──────────────────────────┘  [Spiral][Rings][XOR][Moons][Blank]  │
│                               layers − 2 +   neurons − 12 +       │
│                               tanh relu sigmoid sin   lr ━━●━━    │
│                               [❚❚ Pause] [Step] [Reset]           │
│ ┌ loss / accuracy curve ─────────────────┐ ┌ backward() source ─┐ │
│ │ ╲___________                           │ │ // the actual code │ │
│ └────────────────────────────────────────┘ └────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

- **The model:** a multi-layer perceptron written by hand. It has Xavier/He initialisation, tanh/ReLU/sigmoid/sine activations,
  softmax and cross-entropy, backpropagation, the Adam optimiser and mini-batches. **No ML libraries.** A Node
  test suite checks the gradients numerically (gradient checking), which proves the backprop math is correct.
- **Interaction:**
  - Pick a dataset (3-class spiral, rings, XOR, moons, or blank to draw your own).
  - **Click** the plane to add points of the selected class. The network adapts in real time.
  - Change depth (1–4 hidden layers), width (2–16 neurons), activation and learning rate. The network rebuilds instantly.
  - **Hover** the plane to see that exact input pass through the 3D network. When the mouse is away, a probe moves on its own along a path.
  - **Drag** the 3D view to rotate it.
- **Visuals:**
  - The decision-boundary heat map is coloured by class and brightens with confidence. Misclassified points get a white ring.
  - In the 3D view, neurons glow ember (positive) or ice (negative) with their activation. Connections brighten with signal strength (|w·a|).
  - Light pulses travel along the strongest connections, so you can see the forward pass.
  - The live loss/accuracy chart and the readouts show epoch, loss, accuracy, parameter count and epochs/s.
- **Code panel:** shows the **actual `backward()` source** running on the page (via `Function.prototype.toString`),
  with syntax highlighting. Nothing in it is a mock-up.
- **3D background:** the particles dim to 18% so they don't distract.

## 05 · Capabilities — 3D tilt cards with spotlight

```
┌──────────────────────────────────────────────┐
│ 05 — What I build                            │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐│
│ │ ◎       │ │ ◇       │ │ ↯       │ │ ⇪     ││ ← cards tilt toward the cursor
│ │ Neural  │ │ Real-   │ │ Inter-  │ │ Edge &││   (perspective rotateX/Y), and a
│ │ network │ │ time 3D │ │ active  │ │ deploy││   radial spotlight follows the mouse
│ └─────────┘ └─────────┘ └─────────┘ └───────┘│
└──────────────────────────────────────────────┘
```

- **3D:** galaxy → sphere, still dimmed.
- **Motion:** cards stagger up when they come into view. Tilt and spotlight respond to the mouse. Tag chips sit in a slightly different depth layer (`translateZ`).

## Marquee + stats strip

- A huge outlined marquee: *NEURAL NETWORKS ✦ WEBGL ✦ GLSL ✦ BACKPROPAGATION ✦ …*. It speeds up and **skews with scroll speed**, and reverses direction when you scroll up.
- Number counters run up when the strip comes into view. Every number is **true for this page**:
  `0` ML libraries in the model · `30k` GPU particles · live `parameter count` of the lab network · `6` morph targets.

## 06 · Contact — the portal

```
┌──────────────────────────────────────────────┐
│               ◯ ◯ ◯ ◯ ◯ ◯ ◯                  │
│           ◯                   ◯              │
│         ◯   Have a problem      ◯            │  ← particles form a ring that frames the CTA
│         ◯   worth a neural      ◯            │
│           ◯   network?        ◯              │
│               (( Start a project ))          │  ← big magnetic round button
│ © 2026 Latentform          Back to top ↑     │
└──────────────────────────────────────────────┘
```

- **3D:** sphere → **ring/portal** at full brightness, centred.
- **Motion:** headline words rise in. The CTA is strongly **magnetic** (it follows the cursor within ~120 px and springs back elastically).

---

## Systems that run on every scene

| System | Behaviour |
|---|---|
| **Smooth scroll** | Lenis inertial scrolling, kept in sync with GSAP ScrollTrigger. Anchor links glide to their section. |
| **Custom cursor** | A dot plus a lagging ring. Over interactive elements the ring grows and shows a label (*Drag*, *Add point*, *Play*). Hidden on touch devices. |
| **Magnetic UI** | Buttons marked `data-magnetic` lean toward the cursor. |
| **Section HUD** | The nav shows `03 / 06 · Process`, and a scroll progress bar runs along the top. |
| **Scroll-linked 3D** | A single function maps scroll position to (morph A, morph B, blend, dim, x-offset). Pinned sections are handled correctly, and scrolling backwards works. |
| **Performance** | Fewer particles and no bloom on mobile. Device pixel ratio capped. The lab only trains and renders while visible. |
| **Accessibility** | `prefers-reduced-motion` turns off smooth scroll, marquee motion and intro effects. Semantic HTML, labelled controls and real buttons. |

## Tech architecture

```
latentform-showcase/
├── index.html              structure + content, import map for three / gsap / lenis (CDN, pinned versions)
├── css/style.css           design tokens, layout, components, responsive rules
├── js/main.js              boot: preloader, Lenis + ScrollTrigger, reveals, cursor, magnetic, tilt, marquee, HUD
├── js/hero/
│   ├── ParticleField.js    Three.js renderer, bloom composer, mouse projection, scroll-zone morph state
│   ├── shapes.js           6 morph-target generators (sphere, galaxy, clusters, layers, landscape, ring)
│   └── shaders.js          GLSL: simplex noise, 6-way morph blend, mouse repulsion, soft additive points
├── js/nn/
│   ├── Network.js          ★ the from-scratch neural network (forward, backward, Adam)
│   └── datasets.js         spiral / rings / xor / moons generators (seeded)
├── js/lab/
│   ├── Lab.js              training loop, decision-boundary renderer, controls, loss chart, code panel
│   └── Network3D.js        3D neuron/weight/pulse visualisation of the live network
├── tests/network.test.mjs  numerical gradient check + "learns the spiral" test (node --test)
└── server.mjs              zero-dependency static server (npm start)
```

## 90-second client demo script

1. **(0:00)** Load the page. The "initializing weights" loader shows → *"everything you'll see is generated, not video."*
2. **(0:10)** Move the mouse through the particle core. It pulls apart around the cursor → *"30,000 particles, custom shaders, on the GPU."*
3. **(0:20)** Scroll slowly through the process. The particles become data, then layers, then a loss landscape → *"the story of a model, drawn in 3D."*
4. **(0:40)** Stop at the Lab. Watch the spiral get learned. Switch to **Blank** and click a few points of each class. It retrains instantly →
   *"this network has no ML library behind it. I wrote the backprop. Here's the actual code."*
5. **(1:10)** Hover the plane and point at the 3D network lighting up → *"that's your input moving through every neuron."*
6. **(1:20)** Scroll to contact and pull the magnetic button → *"let's build yours."*
