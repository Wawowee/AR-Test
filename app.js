// Paper Drum — 6 Pads (Acoustic) — FIXED MAPPING
// - Mirror view OFF by default (toggle if your camera feed looks flipped)
// - Proper conversion from MediaPipe normalized coords -> overlay pixels with object-fit: cover

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const btnCam = document.getElementById('btnCam');
const btnCal = document.getElementById('btnCal');
const cbMirror = document.getElementById('cbMirror'); // now unchecked by default

// Pad layout in sheet coordinates
const SHEET_W = 384, SHEET_H = 288;
const SHEET_W = 384, SHEET_H = 288;

// Base pad layout (as defined for the PDF, origin at bottom-left)
const basePads = [
  { name: "Kick",    x:  64, y:  64, r: 34, sound: "sounds/kick.wav" },
  { name: "Snare",   x: 192, y:  64, r: 34, sound: "sounds/snare.wav" },
  { name: "HiHat C", x: 320, y:  64, r: 30, sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     x:  64, y: 180, r: 32, sound: "sounds/tom.wav" },
  { name: "Clap",    x: 192, y: 180, r: 32, sound: "sounds/clap.wav" },
  { name: "HiHat O", x: 320, y: 180, r: 30, sound: "sounds/hihat_open.wav" },
];

// Use this for all rendering & hit-tests: invert Y once for screen coords
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

// Compute how the video is displayed under the overlay when object-fit: cover is used.
function getCoverMapping(overlayW, overlayH, videoW, videoH) {
  const scale = Math.max(overlayW / videoW, overlayH / videoH);
  const displayW = videoW * scale;
  const displayH = videoH * scale;
  const offsetX = (overlayW - displayW) / 2;
  const offsetY = (overlayH - displayH) / 2;
  return { displayW, displayH, offsetX, offsetY };
}

// Convert MediaPipe normalized tip (0..1) to overlay pixel coords, honoring mirror + cover mapping
function tipToOverlayPx(tipNormX, tipNormY) {
  const overlayW = overlay.width;
  const overlayH = overlay.height;
  const videoW = video.videoWidth || overlayW;
  const videoH = video.videoHeight || overlayH;

  const { displayW, displayH, offsetX, offsetY } =
    getCoverMapping(overlayW, overlayH, videoW, videoH);

  const nx = cbMirror.checked ? (1 - tipNormX) : tipNormX; // mirror only if user toggles it
  const ny = tipNormY;

  const px = offsetX + nx * displayW;
  const py = offsetY + ny * displayH;
  return { px, py };
}

// Map overlay pixels to sheet coordinates
function overlayPxToSheet(px, py) {
  const u = px / overlay.width;   // 0..1 across overlay
  const v = py / overlay.height;  // 0..1 down overlay
  return { x: u * SHEET_W, y: v * SHEET_H };
}

function renderOverlay(tipPx) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const pads = padsForScreen();
  const sx = overlay.width / SHEET_W;
  const sy = overlay.height / SHEET_H;
  ctx.lineWidth = 2;
  for (const p of basePads) {
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
    if (v > VELOCITY_THRESH) {
      const pads = padsForScreen();
      for (const p of basePads) {
        const d = Math.hypot(tipSheet.x - p.x, tipSheet.y - p.y);
        if (d <= p.r) {
          const vol = Math.min(1.0, Math.max(0.2, v / 220.0));
          play(p.name, vol);
        }
      }
    }
    lastTip = tipSheet;
  }

  lastTime = ts;
  renderOverlay(tipPx);
  requestAnimationFrame(loop);
}
