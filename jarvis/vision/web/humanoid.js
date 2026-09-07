// Stage 6 (humanoid edition): an alternative hologram visualization — a
// particle-built humanoid bust with a boot-up "assembling" sequence, a
// glowing core that brightens with speech, and particle "mountain range"
// flourishes flanking the figure. Reads exactly the same broadcast state as
// hologram.js (rotation/zoom from gestures, mouthOpenness, processing,
// agents, statusLabel/assistantState) — Python doesn't know or care which
// page is open; this file just draws the shared state differently.
//
// Everything that animates per-frame (the boot assembly, the idle shimmer,
// the breathing drift) runs in the vertex shader off a couple of uniforms,
// so the CPU never touches — or re-uploads — the particle buffers after
// setup. That's what keeps it smooth with ~6k particles; an earlier version
// rebuilt the color buffer on the CPU every frame and stuttered for it.
//
// This file can't be exercised by an automated test (it needs a real
// browser + GPU) — verify it by running `python hologram.py --style
// humanoid` (or `main.py --hologram --hologram-style humanoid`) and looking
// at it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const bootLabelEl = document.getElementById('bootLabel');
const statusLabelEl = document.getElementById('statusLabel');
const agentDashboardEl = document.getElementById('agentDashboard');

// ---------------------------------------------------------------------------
// Bust silhouette: a handful of (heightFraction, radius) keyframes swept as
// horizontal rings of particles from scalp (t=0) down through the neck and
// out across the shoulders/chest (t=1). Smoothstep interpolation between
// keyframes keeps it a single continuous curve, no explicit head/torso seam.
// ---------------------------------------------------------------------------

const BUST_TOP_Y = 1.15;
const BUST_BOTTOM_Y = -1.55;

const RADIUS_KEYFRAMES = [
  [0.0, 0.015],
  [0.13, 0.29],
  [0.27, 0.40],
  [0.45, 0.24],
  [0.53, 0.135],
  [0.62, 0.165],
  [0.7, 0.34],
  [0.86, 0.95],
  [1.0, 0.82],
];

function smootherstep(edge0, edge1, x) {
  const t = THREE.MathUtils.clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function bustRadius(t) {
  for (let i = 0; i < RADIUS_KEYFRAMES.length - 1; i++) {
    const [t0, r0] = RADIUS_KEYFRAMES[i];
    const [t1, r1] = RADIUS_KEYFRAMES[i + 1];
    if (t >= t0 && t <= t1) {
      return THREE.MathUtils.lerp(r0, r1, smootherstep(t0, t1, t));
    }
  }
  return RADIUS_KEYFRAMES[RADIUS_KEYFRAMES.length - 1][1];
}

function depthScale(t) {
  // Head reads rounder (closer to circular cross-section); shoulders/chest
  // read flatter front-to-back, like an actual torso.
  return THREE.MathUtils.lerp(0.78, 0.5, smootherstep(0.55, 0.85, t));
}

// ---------------------------------------------------------------------------
// Shared particle shader. Handles the boot assembly (mix from a scattered
// start position to the real one), a per-particle idle shimmer, and a soft
// round sprite — GL points are square by default, which is most of what made
// an earlier version read as "pixels" rather than "glow".
// ---------------------------------------------------------------------------

const PARTICLE_VERTEX_SHADER = /* glsl */ `
  uniform float uTime;
  uniform float uBoot;
  uniform float uProcessing;
  uniform float uPixelRatio;
  uniform float uSizeScale;

  attribute vec3 aStart;
  attribute vec3 aColor;
  attribute float aPhase;
  attribute float aSize;

  varying vec3 vColor;
  varying float vGlow;

  void main() {
    // Per-particle stagger: each one starts easing into place at a slightly
    // different point in the boot, so the figure assembles in a sweep rather
    // than snapping into existence all at once.
    float staggered = clamp((uBoot - aPhase * 0.32) / 0.68, 0.0, 1.0);
    float eased = 1.0 - pow(1.0 - staggered, 3.0);
    vec3 assembled = mix(aStart, position, eased);

    // Gentle "alive" motion so the figure never looks like a frozen model:
    // a slow breath plus a tiny per-particle drift.
    float breath = 1.0 + 0.007 * sin(uTime * 0.9);
    assembled *= breath;
    assembled.x += 0.005 * sin(uTime * 0.7 + aPhase * 13.0);
    assembled.y += 0.004 * sin(uTime * 0.55 + aPhase * 9.0);

    vec4 mvPosition = modelViewMatrix * vec4(assembled, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = aSize * uSizeScale * uPixelRatio * (260.0 / max(-mvPosition.z, 0.001));

    float shimmerSpeed = 1.2 + 2.6 * uProcessing;
    float shimmerAmp = 0.16 + 0.26 * uProcessing;
    float shimmer = 0.74 + shimmerAmp * (0.5 + 0.5 * sin(uTime * shimmerSpeed + aPhase * 21.0));

    vColor = aColor;
    vGlow = shimmer * eased;
  }
`;

const PARTICLE_FRAGMENT_SHADER = /* glsl */ `
  uniform float uOpacity;

  varying vec3 vColor;
  varying float vGlow;

  void main() {
    // Round sprite with a soft falloff — cheap stand-in for a glow texture,
    // and it plays well with the bloom pass on top.
    float r = length(gl_PointCoord - vec2(0.5));
    if (r > 0.5) discard;
    float alpha = pow(smoothstep(0.5, 0.0, r), 1.6);
    gl_FragColor = vec4(vColor * vGlow, alpha * uOpacity);
  }
`;

function makeParticleMaterial(opacity, sizeScale) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uBoot: { value: 0 },
      uProcessing: { value: 0 },
      uOpacity: { value: opacity },
      uPixelRatio: { value: Math.min(window.devicePixelRatio, 1.5) },
      uSizeScale: { value: sizeScale },
    },
    vertexShader: PARTICLE_VERTEX_SHADER,
    fragmentShader: PARTICLE_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
}

// A particle's "start" position for the boot sweep: a loose swirling spiral,
// matching the reference's comet-trail intro.
function bootStartPosition(index, count) {
  const a = (index / count) * Math.PI * 6 + Math.random() * 0.4;
  const radius = 0.4 + (index / count) * 1.7;
  return [
    Math.cos(a) * radius + (Math.random() - 0.5) * 0.2,
    Math.sin(a * 0.6) * radius * 0.8 - 0.3 + (Math.random() - 0.5) * 0.2,
    (Math.random() - 0.5) * 0.6,
  ];
}

function finishParticleGeometry(positions, colors, sizes) {
  const count = positions.length / 3;
  const starts = new Float32Array(count * 3);
  const phases = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const [sx, sy, sz] = bootStartPosition(i, count);
    starts[i * 3 + 0] = sx;
    starts[i * 3 + 1] = sy;
    starts[i * 3 + 2] = sz;
    phases[i] = Math.random();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));
  geometry.setAttribute('aStart', new THREE.BufferAttribute(starts, 3));
  geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  return geometry;
}

function buildBust() {
  const positions = [];
  const colors = [];
  const sizes = [];
  const color = new THREE.Color();

  const headEnd = 0.56; // fraction of total height where the "dense head" band ends
  const totalSpan = BUST_TOP_Y - BUST_BOTTOM_Y;

  let t = 0.0;
  while (t <= 1.0001) {
    const inHead = t < headEnd;
    const ringPoints = inHead ? 72 : 48;
    const y = BUST_TOP_Y - t * totalSpan;
    const r = bustRadius(t);
    const dz = depthScale(t);

    for (let i = 0; i < ringPoints; i++) {
      const theta = (i / ringPoints) * Math.PI * 2;
      const jitter = (Math.random() - 0.5) * 0.008;
      const profileFront = Math.max(0, Math.cos(theta));
      const nose = inHead ? profileFront * Math.max(0, 0.12 - Math.abs(t - 0.29) * 0.35) : 0;
      const x = Math.cos(theta) * (r + jitter) + nose;
      const z = Math.sin(theta) * (r + jitter) * dz;
      positions.push(x, y, z);

      // Rim-lit look: particles facing the camera burn brighter cyan, ones
      // wrapping around the back fall away into deep blue.
      const frontness = THREE.MathUtils.clamp((z + r * dz) / (2 * r * dz || 1), 0, 1);
      const faceWarmth = inHead ? profileFront * 0.22 : 0;
      color.setHSL(THREE.MathUtils.lerp(0.55, 0.09, faceWarmth), 0.9, THREE.MathUtils.lerp(0.3, 0.68, frontness));
      colors.push(color.r, color.g, color.b);
      sizes.push(THREE.MathUtils.lerp(0.011, 0.021, frontness + faceWarmth * 0.4));
    }

    t += inHead ? 0.009 : 0.028;
  }

  return finishParticleGeometry(positions, colors, sizes);
}

// ---------------------------------------------------------------------------
// Glowing core: horizontal "waveform" strokes inside the face, brightening
// and rippling with speech. This one stays on the CPU — it's ~360 vertices,
// and the ripple needs the live mouthOpenness value anyway.
// ---------------------------------------------------------------------------

const CORE_ROWS = 11;
const CORE_SEGMENTS = 44;

function buildCoreLines(group) {
  const lines = [];
  for (let row = 0; row < CORE_ROWS; row++) {
    const positions = new Float32Array(CORE_SEGMENTS * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({
      color: 0xffa23c,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const line = new THREE.Line(geometry, material);
    group.add(line);
    lines.push({ line, row, positions });
  }
  return lines;
}

function buildChestEmitter(group) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0, -1.04, 0.43,
    -0.04, -1.04, 0.42,
    0.04, -1.04, 0.42,
  ], 3));
  const material = new THREE.PointsMaterial({
    color: 0xffd08a,
    size: 0.12,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const emitter = new THREE.Points(geometry, material);
  group.add(emitter);
  return emitter;
}

function updateCoreLines(coreLines, elapsedSeconds, mouthOpenness, bootAmount) {
  const faceTopT = 0.08;
  const faceBottomT = 0.5;
  const totalSpan = BUST_TOP_Y - BUST_BOTTOM_Y;

  for (const { row, positions, line } of coreLines) {
    const t = THREE.MathUtils.lerp(faceTopT, faceBottomT, row / (CORE_ROWS - 1));
    const y = BUST_TOP_Y - t * totalSpan;
    const halfWidth = bustRadius(t) * 0.72;
    const z = bustRadius(t) * depthScale(t) * 0.6;

    for (let i = 0; i < CORE_SEGMENTS; i++) {
      const u = i / (CORE_SEGMENTS - 1);
      const x = THREE.MathUtils.lerp(-halfWidth, halfWidth, u);
      const idleWobble = Math.sin(elapsedSeconds * 1.3 + row * 0.7 + u * 6) * 0.008;
      const speechWobble = Math.sin(elapsedSeconds * 9 + row * 1.1 + u * 14) * 0.05 * mouthOpenness;
      positions[i * 3 + 0] = x;
      positions[i * 3 + 1] = y + idleWobble + speechWobble;
      positions[i * 3 + 2] = z;
    }
    line.geometry.attributes.position.needsUpdate = true;
    line.material.opacity = bootAmount * (0.3 + 0.65 * mouthOpenness);
  }
}

// ---------------------------------------------------------------------------
// Mountain-range flourishes: layered particle ridgelines flanking the bust,
// mostly cool blue with a few warm "vein" particles threaded through.
// ---------------------------------------------------------------------------

function ridgeHeight(x, seed) {
  return (
    Math.sin(x * 1.7 + seed) * 0.32 +
    Math.sin(x * 4.1 + seed * 1.7) * 0.14 +
    Math.sin(x * 9.3 + seed * 2.3) * 0.06
  );
}

function buildWings() {
  const positions = [];
  const colors = [];
  const sizes = [];
  const blue = new THREE.Color(0x3aa0ff);
  const amber = new THREE.Color(0xffb84a);
  const layers = 5;
  const pointsPerLayer = 150;

  for (const side of [-1, 1]) {
    for (let layer = 0; layer < layers; layer++) {
      const seed = layer * 3.1 + (side > 0 ? 0.5 : 0);
      const depthFade = 1 - layer / layers;
      for (let i = 0; i < pointsPerLayer; i++) {
        const u = i / (pointsPerLayer - 1);
        const x = side * (0.95 + u * 2.35);
        const baseY = -0.35 - layer * 0.22;
        const y = baseY + ridgeHeight(u * 6, seed) * (0.6 + depthFade * 0.6);
        const z = -0.3 - layer * 0.35 + (Math.random() - 0.5) * 0.1;
        positions.push(x, y, z);

        const isVein = Math.random() < 0.1;
        const c = isVein ? amber : blue;
        const brightness = (isVein ? 0.95 : 0.55) * (0.5 + depthFade * 0.5);
        colors.push(c.r * brightness, c.g * brightness, c.b * brightness);
        sizes.push(isVein ? 0.016 : 0.011);
      }
    }
  }

  return finishParticleGeometry(positions, colors, sizes);
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02020a);
scene.fog = new THREE.FogExp2(0x02020a, 0.05);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.1, 4.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.8;
controls.maxDistance = 12;
controls.target.set(0, -0.1, 0);

function buildStarfield(count = 400) {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3 + 0] = (Math.random() - 0.5) * 40;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 40;
    positions[i * 3 + 2] = -Math.random() * 30 - 4;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x8fc7ff,
    size: 0.04,
    transparent: true,
    opacity: 0.45,
    sizeAttenuation: true,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
}
scene.add(buildStarfield());

const bustGroup = new THREE.Group();
scene.add(bustGroup);

const bustMaterial = makeParticleMaterial(0.95, 1.0);
const bustPoints = new THREE.Points(buildBust(), bustMaterial);
bustPoints.frustumCulled = false; // the vertex shader moves points around, so
                                  // the CPU-side bounding box isn't the truth
bustGroup.add(bustPoints);

const wingsMaterial = makeParticleMaterial(0.85, 1.0);
const wingsPoints = new THREE.Points(buildWings(), wingsMaterial);
wingsPoints.frustumCulled = false;
bustGroup.add(wingsPoints);

const coreGroup = new THREE.Group();
bustGroup.add(coreGroup);
const coreLines = buildCoreLines(coreGroup);
const chestEmitter = buildChestEmitter(coreGroup);

// ---------------------------------------------------------------------------
// Post-processing: bloom is what turns "particles" into "glow". Rendered at
// half resolution — the blur is wide enough that nobody can tell, and it's
// the single most expensive thing on the page.
// ---------------------------------------------------------------------------

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2),
  1.25,  // strength
  0.6,   // radius
  0.22   // threshold
);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------------------
// Adaptive quality: if the machine can't hold a smooth frame rate, step the
// expensive things down rather than stuttering — resolution first, then the
// bloom pass entirely.
// ---------------------------------------------------------------------------

const QUALITY_LEVELS = [
  { pixelRatio: Math.min(window.devicePixelRatio, 1.0), bloom: true },
  { pixelRatio: 1.0, bloom: true },
  { pixelRatio: 1.0, bloom: false },
];
let qualityLevel = 0;
let framesSinceCheck = 0;
let lastCheckTime = performance.now();

function applyQuality() {
  const level = QUALITY_LEVELS[qualityLevel];
  renderer.setPixelRatio(level.pixelRatio);
  bustMaterial.uniforms.uPixelRatio.value = level.pixelRatio;
  wingsMaterial.uniforms.uPixelRatio.value = level.pixelRatio;
  bloomPass.enabled = level.bloom;
}

function checkFrameRate(now) {
  framesSinceCheck++;
  const elapsed = now - lastCheckTime;
  if (elapsed < 2000) return;

  const fps = (framesSinceCheck * 1000) / elapsed;
  framesSinceCheck = 0;
  lastCheckTime = now;

  if (fps < 40 && qualityLevel < QUALITY_LEVELS.length - 1) {
    qualityLevel++;
    applyQuality();
  }
}

// ---------------------------------------------------------------------------
// Agent dashboard (same as hologram.js's — the humanoid view keeps showing
// real background-agent status, just without the region sidebar/labels).
// ---------------------------------------------------------------------------

let lastAgentsJSON = '';
function updateAgentDashboard(agents) {
  const asJSON = JSON.stringify(agents || []);
  if (asJSON === lastAgentsJSON) return;
  lastAgentsJSON = asJSON;

  agentDashboardEl.innerHTML = '';
  for (const agent of agents || []) {
    const card = document.createElement('div');
    card.className = `agentCard agentStatus-${agent.status}`;
    const title = document.createElement('div');
    title.className = 'agentName';
    title.textContent = agent.name;
    const status = document.createElement('div');
    status.className = 'agentDetail';
    status.textContent = agent.detail || agent.status;
    card.appendChild(title);
    card.appendChild(status);
    agentDashboardEl.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Live state from Python (gestures, speech, agent status, assistant state)
// ---------------------------------------------------------------------------

let latestState = {
  rotationX: 0, rotationY: 0, zoom: 1.0,
  mouthOpenness: 0, processing: false, agents: [], assistantState: 'idle',
};
let connected = false;
let lastAppliedZoom = null;
let activeSocket = null;

function connectWebSocket() {
  const params = new URLSearchParams(window.location.search);
  const wsPort = params.get('ws_port') || '8766';
  const ws = new WebSocket(`ws://127.0.0.1:${wsPort}`);

  ws.onopen = () => { activeSocket = ws; connected = true; };
  ws.onclose = () => {
    if (activeSocket === ws) activeSocket = null;
    connected = false;
    setTimeout(connectWebSocket, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    try {
      latestState = JSON.parse(event.data);
    } catch (err) {
      // malformed frame; keep the last good state rather than crashing the page
    }
  };
}
connectWebSocket();

const TRACKED_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', ' ', 'Tab', 'Escape']);

function sendKeyEvent(key, down) {
  if (activeSocket && activeSocket.readyState === WebSocket.OPEN) {
    activeSocket.send(JSON.stringify({ type: 'key', key, down }));
  }
}

window.addEventListener('keydown', (e) => {
  if (!TRACKED_KEYS.has(e.key)) return;
  e.preventDefault();
  sendKeyEvent(e.key, true);
});
window.addEventListener('keyup', (e) => {
  if (!TRACKED_KEYS.has(e.key)) return;
  e.preventDefault();
  sendKeyEvent(e.key, false);
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

const BOOT_DURATION_SECONDS = 2.6;
const clock = new THREE.Clock();
let bootComplete = false;

applyQuality();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const elapsedSeconds = clock.getElapsedTime();

  const bootAmount = Math.min(1, elapsedSeconds / BOOT_DURATION_SECONDS);
  if (!bootComplete) {
    bootLabelEl.textContent = `ASSEMBLING... ${Math.round(bootAmount * 100)}%`;
    if (bootAmount >= 1) {
      bootComplete = true;
      bootLabelEl.style.display = 'none';
      statusLabelEl.style.display = 'block';
    }
  }

  const mouthOpenness = latestState.mouthOpenness || 0;
  const processing = latestState.processing ? 1 : 0;

  for (const material of [bustMaterial, wingsMaterial]) {
    material.uniforms.uTime.value = elapsedSeconds;
    material.uniforms.uBoot.value = bootAmount;
    material.uniforms.uProcessing.value = processing;
  }

  updateCoreLines(coreLines, elapsedSeconds, mouthOpenness, bootAmount);
  chestEmitter.material.size = 0.09 + 0.035 * Math.sin(elapsedSeconds * 2.4) + 0.05 * mouthOpenness;
  chestEmitter.material.opacity = bootAmount * (0.78 + 0.2 * processing);
  coreGroup.visible = bootAmount > 0.15;

  bustGroup.rotation.x = THREE.MathUtils.degToRad(latestState.rotationX || 0);
  bustGroup.rotation.y = THREE.MathUtils.degToRad(latestState.rotationY || 0);

  if (bootComplete) {
    const state = connected ? (latestState.assistantState || 'idle') : 'offline';
    statusLabelEl.textContent = `STATUS: ${state.toUpperCase()}`;
  }
  updateAgentDashboard(latestState.agents);

  controls.update();
  if (latestState.zoom !== lastAppliedZoom) {
    camera.position.setLength(4.2 / Math.max(latestState.zoom || 1.0, 0.1));
    lastAppliedZoom = latestState.zoom;
  }

  composer.render();
  checkFrameRate(now);
}

animate();
