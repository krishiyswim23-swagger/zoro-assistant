// Stage 6 (brain-graph edition): visualizes JARVIS's neural-activity
// pipeline (jarvis/voice/brain_activity.py's per-region firing levels) and
// live background-agent status (jarvis/agents/) in the browser via
// Three.js/WebGL, replacing the earlier face-based hologram. Python still
// owns camera capture, hand tracking, and gesture interpretation (rotation/
// zoom of the graph), plus all of the region-firing/agent-status logic —
// this file only draws state streamed to it over a local WebSocket.
//
// This file can't be exercised by an automated test (it needs a real
// browser + GPU) — verify it by actually running `python hologram.py` (or
// `main.py --hologram`) and looking at it.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const statusEl = document.getElementById('statusLabel');
const cortexListEl = document.getElementById('cortexList');
const agentDashboardEl = document.getElementById('agentDashboard');
const regionLabelsEl = document.getElementById('regionLabels');

// ---------------------------------------------------------------------------
// Region metadata — the keys and the "planned" flag must match
// jarvis/voice/brain_activity.py's REGIONS/PLANNED_REGIONS exactly (that's
// the source of truth for which regions exist and which are real); only
// display label/color/layout live here, since that's presentation, not
// pipeline logic.
// ---------------------------------------------------------------------------

const REGIONS = [
  { key: 'prefrontal', label: 'PREFRONTAL', color: 0x8a6bff, planned: false },
  { key: 'association', label: 'ASSOCIATION', color: 0xe0a83a, planned: false },
  { key: 'reflex_arc', label: 'REFLEX ARC', color: 0x2ec4b6, planned: false },
  { key: 'sensory', label: 'SENSORY CORTEX', color: 0x33c9dd, planned: false },
  { key: 'language', label: 'LANGUAGE', color: 0x52e08a, planned: false },
  { key: 'motor', label: 'MOTOR CORTEX', color: 0xff5c5c, planned: false },
  { key: 'memory', label: 'HIPPOCAMPUS', color: 0xa3d63a, planned: false },
  { key: 'predictive', label: 'PREDICTIVE', color: 0xe05cd9, planned: false },
  { key: 'brainstem', label: 'BRAINSTEM', color: 0x4a90e0, planned: false },
  { key: 'cerebellum', label: 'CEREBELLUM', color: 0x667788, planned: true },
];

const NODES_PER_REGION = 26;
const CLUSTER_RADIUS = 0.34;
const LAYOUT_RADIUS = 1.35;

// Deterministic anchor position per region, spread around a sphere — the
// same Fibonacci-sphere distribution as holographic_scene.py's
// generate_sphere_points, just inlined here since this is a one-off layout,
// not something Python needs to know about.
function regionAnchor(index, total) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = total <= 1 ? 0 : 1 - (index / (total - 1)) * 2;
  const ringRadius = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = golden * index;
  return new THREE.Vector3(
    Math.cos(theta) * ringRadius * LAYOUT_RADIUS,
    y * LAYOUT_RADIUS,
    Math.sin(theta) * ringRadius * LAYOUT_RADIUS
  );
}

const regionAnchors = REGIONS.map((_, i) => regionAnchor(i, REGIONS.length));

function buildBrainGraph() {
  const positions = [];
  const colors = [];
  const regionIndexPerNode = [];
  const baseColor = new THREE.Color();

  REGIONS.forEach((region, regionIndex) => {
    const anchor = regionAnchors[regionIndex];
    baseColor.set(region.color);
    for (let i = 0; i < NODES_PER_REGION; i++) {
      // Sum of three uniforms approximates a Gaussian scatter around the
      // anchor — cheap, and no Box-Muller needed for a purely visual jitter.
      const jitter = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;
      positions.push(
        anchor.x + jitter() * CLUSTER_RADIUS,
        anchor.y + jitter() * CLUSTER_RADIUS,
        anchor.z + jitter() * CLUSTER_RADIUS
      );
      colors.push(baseColor.r, baseColor.g, baseColor.b);
      regionIndexPerNode.push(regionIndex);
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  return { geometry, regionIndexPerNode, basePositions: positions, baseColors: colors.slice() };
}

function buildEdges(basePositions, regionIndexPerNode) {
  const linePositions = [];
  const lineColors = [];
  const color = new THREE.Color();

  // Within-cluster edges: connect each node to its 2 nearest neighbors in
  // the same region (cheap — each cluster is only ~26 nodes).
  REGIONS.forEach((region, regionIndex) => {
    const indices = [];
    for (let i = 0; i < regionIndexPerNode.length; i++) {
      if (regionIndexPerNode[i] === regionIndex) indices.push(i);
    }
    color.set(region.color);
    for (const i of indices) {
      const xi = basePositions[i * 3], yi = basePositions[i * 3 + 1], zi = basePositions[i * 3 + 2];
      const nearest = indices
        .filter((j) => j !== i)
        .map((j) => {
          const dx = basePositions[j * 3] - xi, dy = basePositions[j * 3 + 1] - yi, dz = basePositions[j * 3 + 2] - zi;
          return { j, d2: dx * dx + dy * dy + dz * dz };
        })
        .sort((a, b) => a.d2 - b.d2)
        .slice(0, 2);
      for (const { j } of nearest) {
        linePositions.push(xi, yi, zi, basePositions[j * 3], basePositions[j * 3 + 1], basePositions[j * 3 + 2]);
        lineColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    }
  });

  // A handful of long cross-region edges between anchors, for the
  // "everything's connected" look the reference has.
  const dim = new THREE.Color(0x3a4a5a);
  for (let i = 0; i < regionAnchors.length; i++) {
    const next = regionAnchors[(i + 1) % regionAnchors.length];
    const a = regionAnchors[i];
    linePositions.push(a.x, a.y, a.z, next.x, next.y, next.z);
    lineColors.push(dim.r, dim.g, dim.b, dim.r, dim.g, dim.b);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
  return geometry;
}

function updateFiring(pointsGeometry, regionIndexPerNode, baseColors, regionFiring) {
  const colorAttr = pointsGeometry.getAttribute('color');
  for (let i = 0; i < regionIndexPerNode.length; i++) {
    const region = REGIONS[regionIndexPerNode[i]];
    const level = (regionFiring && regionFiring[region.key]) || 0;
    const brightness = 0.35 + 0.65 * level;
    colorAttr.array[i * 3 + 0] = baseColors[i * 3 + 0] * brightness;
    colorAttr.array[i * 3 + 1] = baseColors[i * 3 + 1] * brightness;
    colorAttr.array[i * 3 + 2] = baseColors[i * 3 + 2] * brightness;
  }
  colorAttr.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02020a);
scene.fog = new THREE.FogExp2(0x02020a, 0.035);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 4.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 2.0;
controls.maxDistance = 14;

scene.add(new THREE.AmbientLight(0x16202a, 0.9));
const keyLight = new THREE.DirectionalLight(0xcfe8ff, 1.0);
keyLight.position.set(1.4, 1.8, 2.6);
scene.add(keyLight);

// -- background: a sparse starfield, nothing else — the graph floats in open space --

function buildStarfield(count = 600) {
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
    size: 0.045,
    transparent: true,
    opacity: 0.65,
    sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}
scene.add(buildStarfield());

// -- the brain graph -------------------------------------------------------

const { geometry: pointsGeometry, regionIndexPerNode, basePositions, baseColors } = buildBrainGraph();
const pointsMaterial = new THREE.PointsMaterial({
  size: 0.05,
  vertexColors: true,
  transparent: true,
  opacity: 0.9,
  sizeAttenuation: true,
});
const points = new THREE.Points(pointsGeometry, pointsMaterial);

const edgesGeometry = buildEdges(basePositions, regionIndexPerNode);
const edgesMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 });
const edges = new THREE.LineSegments(edgesGeometry, edgesMaterial);

const brainGroup = new THREE.Group();
brainGroup.add(points);
brainGroup.add(edges);
scene.add(brainGroup);

// ---------------------------------------------------------------------------
// Post-processing: bloom is what actually makes a "glow" look like a glow.
// ---------------------------------------------------------------------------

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.9,   // strength
  0.5,   // radius
  0.35   // threshold
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
// HUD overlays: sidebar cortex-status list (static — built once), region
// labels (camera-projected each frame), and the live agent dashboard.
// ---------------------------------------------------------------------------

REGIONS.forEach((region) => {
  const row = document.createElement('div');
  row.className = 'cortexRow';
  const dot = document.createElement('span');
  dot.className = 'cortexDot';
  dot.style.background = `#${region.color.toString(16).padStart(6, '0')}`;
  const text = document.createElement('span');
  text.textContent = `${region.label}  ${region.planned ? 'PLANNED' : 'LIVE'}`;
  row.appendChild(dot);
  row.appendChild(text);
  cortexListEl.appendChild(row);
});

const regionLabelEls = REGIONS.map((region) => {
  const el = document.createElement('div');
  el.className = 'regionLabel';
  el.style.borderColor = `#${region.color.toString(16).padStart(6, '0')}`;
  regionLabelsEl.appendChild(el);
  return el;
});

function updateRegionLabels(regionFiring) {
  const projected = new THREE.Vector3();
  REGIONS.forEach((region, i) => {
    projected.copy(regionAnchors[i]).applyEuler(brainGroup.rotation).project(camera);
    const behindCamera = projected.z > 1;
    const el = regionLabelEls[i];
    if (behindCamera) {
      el.style.display = 'none';
      return;
    }
    el.style.display = 'block';
    el.style.left = `${((projected.x + 1) / 2) * window.innerWidth}px`;
    el.style.top = `${((1 - projected.y) / 2) * window.innerHeight}px`;
    const level = (regionFiring && regionFiring[region.key]) || 0;
    el.textContent = region.planned
      ? `${region.label}\n${NODES_PER_REGION} nodes · planned`
      : `${region.label}\n${NODES_PER_REGION} nodes · firing ${(level * 100).toFixed(0)}%`;
  });
}

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
// Live state from Python (gestures, brain activity, agent status, speech)
// ---------------------------------------------------------------------------

let latestState = {
  rotationX: 0, rotationY: 0, zoom: 1.0,
  regions: {}, agents: [], statusLabel: 'CONNECTING...',
};
let lastAppliedZoom = null; // re-set camera distance only when Python's gesture
                            // zoom actually changes, so it doesn't fight the
                            // user's own scroll-to-zoom every frame

let activeSocket = null; // the live WebSocket, if any — keydown/keyup handlers send through this

function connectWebSocket() {
  const params = new URLSearchParams(window.location.search);
  const wsPort = params.get('ws_port') || '8766';
  const ws = new WebSocket(`ws://127.0.0.1:${wsPort}`);

  ws.onopen = () => { activeSocket = ws; };
  ws.onclose = () => {
    statusEl.textContent = 'disconnected — retrying...';
    if (activeSocket === ws) activeSocket = null;
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

// -- keyboard controls: only meaningful to Python's run_keyboard_demo() (which
// reads these back over the same socket to drive the scene when there's no
// camera), but harmless to always send since run()'s camera/gesture mode
// simply never looks at them. --------------------------------------------

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

function animate() {
  requestAnimationFrame(animate);

  updateFiring(pointsGeometry, regionIndexPerNode, baseColors, latestState.regions);

  // Spins the whole graph; OrbitControls separately orbits the *camera*
  // around the scene, so mouse-drag and gesture-driven rotation don't
  // fight each other.
  brainGroup.rotation.x = THREE.MathUtils.degToRad(latestState.rotationX || 0);
  brainGroup.rotation.y = THREE.MathUtils.degToRad(latestState.rotationY || 0);

  statusEl.textContent = latestState.statusLabel || 'CONNECTING...';
  updateRegionLabels(latestState.regions);
  updateAgentDashboard(latestState.agents);

  // Let the user's own scroll-zoom (handled internally by OrbitControls)
  // apply first; only override the camera distance when Python's gesture
  // zoom actually changes, so the two controls don't fight every frame.
  controls.update();
  if (latestState.zoom !== lastAppliedZoom) {
    camera.position.setLength(4.5 / Math.max(latestState.zoom || 1.0, 0.1));
    lastAppliedZoom = latestState.zoom;
  }

  composer.render();
}

animate();
