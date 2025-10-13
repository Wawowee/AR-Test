// Paper Drum — 6 Pads (Acoustic)
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const btnCam = document.getElementById('btnCam');
const btnCal = document.getElementById('btnCal');
const cbMirror = document.getElementById('cbMirror');

const pads = [
  { name: "Kick",    x: 64,  y: 64,  r: 34, sound: "sounds/kick.wav" },
  { name: "Snare",   x: 192, y: 64,  r: 34, sound: "sounds/snare.wav" },
  { name: "HiHat C", x: 320, y: 64,  r: 30, sound: "sounds/hihat_closed.wav" },
  { name: "Tom",     x: 64,  y: 180, r: 32, sound: "sounds/tom.wav" },
  { name: "Clap",    x: 192, y: 180, r: 32, sound: "sounds/clap.wav" },
  { name: "HiHat O", x: 320, y: 180, r: 30, sound: "sounds/hihat_open.wav" },
];

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
  for (const p of pads) {
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

function videoToSheet(x, y) {
  const vw = overlay.clientWidth;
  const vh = overlay.clientHeight;
  let nx = x / vw;
  let ny = y / vh;
  if (cbMirror.checked) nx = 1 - nx;
  return { x: nx * 384, y: ny * 288 };
}

function renderOverlay(tip) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const sx = overlay.width / 384;
  const sy = overlay.height / 288;

  ctx.lineWidth = 2;
  for (const p of pads) {
    ctx.beginPath();
    ctx.arc(p.x * sx, p.y * sy, p.r * ((sx+sy)/2), 0, Math.PI*2);
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.stroke();
    ctx.font = "12px system-ui";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.fillText(p.name, p.x * sx - 16, p.y * sy + 4);
  }
  if (tip) {
    ctx.beginPath();
    ctx.arc(tip.x * sx, tip.y * sy, 7, 0, Math.PI*2);
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

  if (result && result.landmarks && result.landmarks.length > 0) {
    const lm = result.landmarks[0];
    const tip = lm[8];
    const x = tip.x * overlay.clientWidth;
    const y = tip.y * overlay.clientHeight;
    tipSheet = videoToSheet(x, y);
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
      for (const p of pads) {
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
  renderOverlay(tipSheet);
  requestAnimationFrame(loop);
}
