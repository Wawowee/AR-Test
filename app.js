// Paper Drum — 6 Pads (Acoustic) — FIXED MAPPING + Y-FLIP FOR LABELS

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// ↓ Declare references; we’ll assign them after DOM is ready
let video, overlay, ctx, statusEl, btnCam, btnCal, cbMirror;
document.addEventListener('DOMContentLoaded', () => {
  // Grab DOM elements after the page has built them
  video    = document.getElementById('video');
  overlay  = document.getElementById('overlay');
  ctx      = overlay.getContext('2d');
  statusEl = document.getElementById('status');
  btnCam   = document.getElementById('btnCam');
  btnCal   = document.getElementById('btnCal');
  cbMirror = document.getElementById('cbMirror');

  // Size the canvas and listen for resizes
  function resizeCanvas() {
    overlay.width  = overlay.clientWidth;
    overlay.height = overlay.clientHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Wire up buttons (handlers already defined below)
  btnCam.onclick = onStartCamera;
  btnCal.onclick = onCalibrate;
});

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

// --- Calibration (homography) state ---
let H = null;             // Float64Array length 9 (3x3) or null
let calibCorners = null;  // debug: the 4 detected corners in overlay pixels


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
// Wait until OpenCV.js is ready
function waitForOpenCV() {
  return new Promise((resolve) => {
    if (window.cv && cv.getBuildInformation) return resolve();
    const check = () => (window.cv && cv.getBuildInformation) ? resolve() : setTimeout(check, 50);
    check();
  });
}

// Grab a video frame into an OpenCV Mat (in the video’s native resolution)
function frameToMat(videoEl) {
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const canvas = document.createElement('canvas');
  canvas.width = vw; canvas.height = vh;
  const c2d = canvas.getContext('2d');
  c2d.drawImage(videoEl, 0, 0, vw, vh);
  const imgData = c2d.getImageData(0, 0, vw, vh);
  const mat = cv.matFromImageData(imgData);
  return mat; // CV_8UC4
}

// Order 4 points TL, TR, BR, BL using sums/diffs
function orderCornersTLTRBRBL(pts) {
  // pts: [{x,y} * 4]
  const sum = pts.map(p => p.x + p.y);
  const diff = pts.map(p => p.x - p.y);
  const TL = pts[sum.indexOf(Math.min(...sum))];
  const BR = pts[sum.indexOf(Math.max(...sum))];
  const TR = pts[diff.indexOf(Math.max(...diff))];
  const BL = pts[diff.indexOf(Math.min(...diff))];
  return [TL, TR, BR, BL];
}

// Convert VIDEO-space points -> OVERLAY pixels (matches how we render with object-fit: cover)
function videoPtToOverlayPx(pt) {
  const overlayW = overlay.width, overlayH = overlay.height;
  const videoW = video.videoWidth || overlayW, videoH = video.videoHeight || overlayH;
  const { displayW, displayH, offsetX, offsetY } = getCoverMapping(overlayW, overlayH, videoW, videoH);
  const nx = pt.x / videoW, ny = pt.y / videoH;
  return { px: offsetX + nx * displayW, py: offsetY + ny * displayH };
}

// Build 3x3 homography mapping OVERLAY pixels -> SHEET coords
function computeHomographyOverlay(srcOverlayPts /* TL,TR,BR,BL */) {
  // src: overlay pixels; dst: sheet coords (0,0)-(W,0)-(W,H)-(0,H)
  const dst = [
    {x:0,        y:0},
    {x:SHEET_W,  y:0},
    {x:SHEET_W,  y:SHEET_H},
    {x:0,        y:SHEET_H}
  ];

  const srcMat = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array([
    srcOverlayPts[0].px, srcOverlayPts[0].py,
    srcOverlayPts[1].px, srcOverlayPts[1].py,
    srcOverlayPts[2].px, srcOverlayPts[2].py,
    srcOverlayPts[3].px, srcOverlayPts[3].py,
  ]));
  const dstMat = cv.matFromArray(4, 1, cv.CV_32FC2, new Float32Array([
    dst[0].x, dst[0].y,
    dst[1].x, dst[1].y,
    dst[2].x, dst[2].y,
    dst[3].x, dst[3].y,
  ]));

  const Hmat = cv.getPerspectiveTransform(srcMat, dstMat); // 3x3 CV_64F
  // Export to a plain Float64Array for fast apply
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r*3+c] = Hmat.doubleAt(r, c);

  srcMat.delete(); dstMat.delete(); Hmat.delete();
  return out;
}

// Apply H to an OVERLAY pixel -> SHEET coordinate
function applyHomography(px, py) {
  if (!H) return { x: (px / overlay.width) * SHEET_W, y: (py / overlay.height) * SHEET_H };
  const x = px, y = py;
  const w = H[6]*x + H[7]*y + H[8];
  const sx = (H[0]*x + H[1]*y + H[2]) / w;
  const sy = (H[3]*x + H[4]*y + H[5]) / w;
  return { x: sx, y: sy };
}

// Detect 4 largest square-ish contours (the black corner markers) in VIDEO space
function detectCornerSquares(videoMat /* CV_8UC4 */) {
  // Convert to gray
  const gray = new cv.Mat();
  cv.cvtColor(videoMat, gray, cv.COLOR_RGBA2GRAY, 0);

  // Blur a bit to reduce noise
  const blur = new cv.Mat();
  cv.GaussianBlur(gray, blur, new cv.Size(5,5), 0);

  // Adaptive threshold works across lighting
  const bin = new cv.Mat();
  cv.adaptiveThreshold(blur, bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5);

  // Find contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(bin, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

  const vw = videoMat.cols, vh = videoMat.rows;
  const minArea = 0.0005 * vw * vh; // ignore tiny specks

  const candidates = [];
  for (let i=0; i<contours.size(); i++) {
    const cnt = contours.get(i);
    const peri = cv.arcLength(cnt, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, 0.03 * peri, true);

    if (approx.rows === 4 && cv.isContourConvex(approx)) {
      const area = cv.contourArea(approx);
      if (area > minArea) {
        // Get bounding rect to check squareness
        const rect = cv.boundingRect(approx);
        const ar = rect.width / rect.height;
        const squarish = ar > 0.6 && ar < 1.4;

        if (squarish) {
          // Use centroid as point location
          let cx = 0, cy = 0;
          for (let j=0; j<4; j++) {
            const px = approx.intPtr(j,0)[0];
            const py = approx.intPtr(j,0)[1];
            cx += px; cy += py;
          }
          cx /= 4; cy /= 4;
          candidates.push({ area, rect, cx, cy, approx });
        }
      }
    }

    approx.delete();
    cnt.delete();
  }

  // Sort by area (largest first), take top 6, then pick the 4 that are most "spread out"
  candidates.sort((a,b) => b.area - a.area);
  const top = candidates.slice(0, 6);

  if (top.length < 4) {
    gray.delete(); blur.delete(); bin.delete(); contours.delete(); hierarchy.delete();
    return null;
  }

  // Choose 4 by maximizing pairwise distances (greedy)
  let bestSet = null, bestScore = -1;
  const pts = top.map(t => ({ x: t.cx, y: t.cy }));
  for (let i=0;i<pts.length;i++)
    for (let j=i+1;j<pts.length;j++)
      for (let k=j+1;k<pts.length;k++)
        for (let l=k+1;l<pts.length;l++) {
          const set = [pts[i], pts[j], pts[k], pts[l]];
          // score = sum of squared distances (spread)
          let s=0;
          for (let a=0;a<4;a++) for (let b=a+1;b<4;b++) {
            const dx = set[a].x - set[b].x, dy = set[a].y - set[b].y;
            s += dx*dx + dy*dy;
          }
          if (s > bestScore) { bestScore = s; bestSet = set; }
        }

  gray.delete(); blur.delete(); bin.delete(); contours.delete(); hierarchy.delete();
  return bestSet; // 4 points in VIDEO coords, unordered
}


async function onStartCamera() {
  await initCamera();
  if (!audioCtx) await initAudio();
  if (!handLandmarker) await initHands();
  statusEl.textContent = "Camera on — show the printed sheet and tap a pad.";
  requestAnimationFrame(loop);
}

async function onCalibrate() {
  statusEl.textContent = "Calibrating…";
  await waitForOpenCV();

  // 1) capture a frame from the video
  const mat = frameToMat(video);

  // 2) detect 4 corners in VIDEO coordinates
  let cornersVideo = detectCornerSquares(mat); // [{x,y} * 4] or null
  mat.delete();

  if (!cornersVideo || cornersVideo.length !== 4) {
    statusEl.textContent = "Couldn’t find 4 corners. Ensure all corner squares are visible & lighting is even.";
    H = null; calibCorners = null;
    return;
  }

  // 3) order TL,TR,BR,BL in VIDEO coords
  cornersVideo = orderCornersTLTRBRBL(cornersVideo);

  // 4) convert VIDEO coords -> OVERLAY px (to match our fingertip px)
  const cornersOverlay = cornersVideo.map(pt => videoPtToOverlayPx(pt));

  // 5) compute homography OVERLAY px -> SHEET coords
  H = computeHomographyOverlay(cornersOverlay);
  calibCorners = cornersOverlay; // for visual debug
  statusEl.textContent = "Calibrated ✅";
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
  // Debug: show detected corner dots after calibration
if (calibCorners && calibCorners.length === 4) {
  ctx.fillStyle = "rgba(0,255,120,0.9)";
  for (const c of calibCorners) {
    ctx.beginPath();
    ctx.arc(c.px, c.py, 6, 0, Math.PI*2);
    ctx.fill();
  }
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

    // If calibrated, use homography; else fall back to simple mapping
    tipSheet = H ? applyHomography(p.px, p.py) : overlayPxToSheet(p.px, p.py);
  }

  if (tipSheet) {
    const dt = (ts - lastTime) / 1000;

    // Scalar speed (kept for reference/volume)
    let v = 0;
    if (lastTip && dt > 0) {
      const dx = tipSheet.x - lastTip.x;
      const dy = tipSheet.y - lastTip.y;
      v = Math.hypot(dx, dy) / dt;
    }

    // --- Hysteresis-based retriggering inside pads ---
    const V_HIT = 220;              // speed to trigger
    const V_ARM = 120;              // speed below which we "re-arm"
    const MIN_RETRIGGER_MS = 100;   // debounce between hits
    const REQUIRE_DOWNWARD = false; // set true for downward-only strokes

    // per-pad state container
    if (!window.__padState) window.__padState = new Map(); // name -> { armed, lastTrig, inside }

    // velocity components (re-use dt)
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

