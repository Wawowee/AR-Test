// Paper Drum — 6 Pads (Acoustic) — FIXED MAPPING + Y-FLIP FOR LABELS

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const btnCam = document.getElementById('btnCam');
const btnCal = document.getElementById('btnCal');
const cbMirror = document.getElementById('cbMirror'); // unchecked by default

// --- Coordinate space for the sheet overlay ---
const SHEET_W = 384, SHEET_H = 288;

// Base pad layout (defined for the PDF; origin effectively bottom-left)
const basePads = [
  { name: "Kick",    x:  64, y:  64, r: 34, sound: "sounds/kick.wav" },
  { name: "Snare",   x: 192, y:  64, r: 34, sound: "sounds/snare.wav" },
  { name: "HiHat C", x: 320, y:  64, r: 30, sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     x:  64, y: 180, r: 32, sound: "sounds/tom.wav" },
  { name: "Clap",    x: 192, y: 180, r: 32, sound: "sounds/clap.wav" },
  { name: "HiHat O", x: 320, y: 180, r: 30, sound: "sounds/hihat_open.wav" },
];

// Use this for all rendering & hit tests: invert Y once to match screen (top-left origin)
function padsForScreen() {
  return basePads.map(p => ({ ...p, y: (SHEET_H - p.y) }));
}

let audioCtx;
const samples = new Map();
let handLandmarker = null;
let lastTip = null;
let lastTime = 0;
const cooldown = new Map();
const COOLDOWN_MS = 120;
const VELOCITY_THRESH = 2.0;
let wasInside = new Map();


function resizeCanvas() {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

async function initAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  for (const p of basePads) {
    const ab = await fetch(p.sound).then(r => r.arrayBuffer());
    const buf = await audioCtx.decodeAudioData(ab);
    samples.set(p.name, buf);
  }
}

function play(name, gain=1.0) {
  if (!audioCtx) return;
  const now = performance.now();
  if (cooldown.get(name) && now - cooldown.get(name) < COOLDOWN_MS) return;
  const buf = samples.get(name);
  if (!buf) return;
  const src = audioCtx.createBufferSource();
  const g = audioCtx.createGain();
  g.gain.value = Math.max(0.1, Math.min(1.0, gain));
  src.buffer = buf;
  src.connect(g).connect(audioCtx.destination);
  src.start();
  cooldown.set(name, now);
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
  video.srcObject = stream;
  await video.play();
  resizeCanvas();
}

async function initHands() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    numHands: 1,
    runningMode: "VIDEO"
  });
}

btnCam.onclick = async () => {
  await initCamera();
  if (!audioCtx) await initAudio();
  if (!handLandmarker) await initHands();
  statusEl.textContent = "Camera on — show the printed sheet and tap a pad.";
  requestAnimationFrame(loop);
};

btnCal.onclick = () => {
  statusEl.textContent = "Calibration set (identity). Keep phone square to the paper.";
};

// --- Video → overlay mapping (accounts for object-fit: cover) ---
function getCoverMapping(overlayW, overlayH, videoW, videoH) {
  const scale = Math.max(overlayW / videoW, overlayH / videoH);
  const displayW = videoW * scale;
  const displayH = videoH * scale;
  const offsetX = (overlayW - displayW) / 2;
  const offsetY = (overlayH - displayH) / 2;
  return { displayW, displayH, offsetX, offsetY };
}

function tipToOverlayPx(tipNormX, tipNormY) {
  const overlayW = overlay.width;
  const overlayH = overlay.height;
  const videoW = video.videoWidth || overlayW;
  const videoH = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - tipNormX) : tipNormX; // mirror only if toggled
  const ny = tipNormY;

  const px = offsetX + nx * displayW;
  const py = offsetY + ny * displayH;
  return { px, py };
}

// Overlay pixels → sheet coords
function overlayPxToSheet(px, py) {
  const u = px / overlay.width;
  const v = py / overlay.height;
  return { x: u * SHEET_W, y: v * SHEET_H };
}

function renderOverlay(tipPx) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const pads = padsForScreen();   // use Y-flipped pads
  const sx = overlay.width / SHEET_W;
  const sy = overlay.height / SHEET_H;

  ctx.lineWidth = 2;
  for (const p of pads) {
    ctx.beginPath();
    ctx.arc(p.x * sx, p.y * sy, p.r * ((sx + sy) / 2), 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.stroke();
    ctx.font = "12px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillText(p.name, p.x * sx - 16, p.y * sy + 4);
  }

  if (tipPx) {
    ctx.beginPath();
    ctx.arc(tipPx.px, tipPx.py, 7, 0, Math.PI*2);
    ctx.fillStyle = "rgba(0,200,255,0.95)";
    ctx.fill();
  }
}

async function loop(ts) {
  if (!video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(loop);
    return;
  }
  const result = handLandmarker.detectForVideo(video, ts);
  let tipSheet = null;
  let tipPx = null;

  if (result && result.landmarks && result.landmarks.length > 0) {
    const lm = result.landmarks[0];
    const tip = lm[8]; // index fingertip
    const p = tipToOverlayPx(tip.x, tip.y);
    tipPx = p;
    tipSheet = overlayPxToSheet(p.px, p.py);
  }

  if (tipSheet) {
    const dt = (ts - lastTime) / 1000;
    let v = 0;
    if (lastTip && dt > 0) {
      const dx = tipSheet.x - lastTip.x;
      const dy = tipSheet.y - lastTip.y;
      v = Math.hypot(dx, dy) / dt;
    }

    // --- Hysteresis-based retriggering inside pads ---
const V_HIT = 220;             // speed to trigger
const V_ARM = 120;             // speed below which we "re-arm"
const MIN_RETRIGGER_MS = 100;  // debounce between hits
const REQUIRE_DOWNWARD = false; // set true if you only want downward strokes

// per-pad state container
if (!window.__padState) window.__padState = new Map(); // name -> { armed, lastTrig, inside }

// compute velocity components for optional direction checks
const dt = (ts - lastTime) / 1000;
let vx = 0, vy = 0;
if (lastTip && dt > 0) {
  vx = (tipSheet.x - lastTip.x) / dt;
  vy = (tipSheet.y - lastTip.y) / dt;
}
const speed = Math.hypot(vx, vy);
const now = performance.now();

const pads = padsForScreen();

for (const p of pads) {
  const d = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
  const inside = d <= p.r;

  let st = window.__padState.get(p.name);
  if (!st) {
    st = { armed: true, lastTrig: 0, inside: false };
    window.__padState.set(p.name, st);
  }

  // Re-arm when you leave the pad OR slow down enough
  if (!inside || speed < V_ARM) {
    st.armed = true;
  }

  // Optional: only count strokes moving downward on screen
  const downOk = REQUIRE_DOWNWARD ? (vy > 0) : true;

  // Fire if armed, inside, fast enough, debounced, and (optionally) downward
  if (st.armed && inside && speed > V_HIT && (now - st.lastTrig) > MIN_RETRIGGER_MS && downOk) {
    const vol = Math.min(1.0, Math.max(0.15, speed / 220));
    play(p.name, vol);
    st.lastTrig = now;
    st.armed = false; // wait until you slow down or exit to re-arm
  }

  st.inside = inside;
}


    lastTip = tipSheet;
  }

  lastTime = ts;
  renderOverlay(tipPx);
  requestAnimationFrame(loop);
}
