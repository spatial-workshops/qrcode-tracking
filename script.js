const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

let sampleCanvas;
let sampleCtx;
let currentStream;
let currentFacingMode = 'environment';
let tickStarted = false;

const UNKNOWN_CODE_COLOR = '#00ff00';
const GROUP_COLOR = '#ffffff';

// QR detection drops out for the odd frame, so a code is kept on screen until
// it has gone unseen for this long. That stops names and groups flickering.
const TRACK_LINGER_MS = 400;
// After the finger leaves a code, its personality stays up this long.
const REVEAL_LINGER_MS = 800;
// Two codes count as "together" when their centers are closer than this many
// code-widths apart.
const GROUP_DISTANCE = 2.2;

// Hand tracking (MediaPipe HandLandmarker) powers "point at a mouse to see its
// personality". It loads its WASM runtime and model from a CDN asynchronously,
// so `handLandmarker` stays null until that finishes — tick() just skips
// pointing detection until then.
let handLandmarker = null;

async function initHandLandmarker() {
  const { HandLandmarker, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14'
  );
  const filesetResolver = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
  });
}

initHandLandmarker().catch((error) => {
  console.error('Unable to load hand tracking:', error);
});

// A finger moves far less between frames than it takes to run the model, so
// the hand is only tracked every HAND_INTERVAL frames and the last fingertip
// is reused in between. That frees time for QR scanning.
const HAND_INTERVAL = 2;
let handFrame = 0;
let lastFingertip = null;

// Landmark 8 is the index fingertip in MediaPipe's 21-point hand model.
// Coordinates come back normalized (0-1) relative to the video frame.
function getIndexFingertip() {
  if (!handLandmarker) {
    return null;
  }

  handFrame = (handFrame + 1) % HAND_INTERVAL;
  if (handFrame !== 0) {
    return lastFingertip;
  }

  const result = handLandmarker.detectForVideo(video, performance.now());
  const landmarks = result.landmarks[0];
  if (!landmarks) {
    lastFingertip = null;
    return null;
  }

  const tip = landmarks[8];
  lastFingertip = { x: tip.x * sampleCanvas.width, y: tip.y * sampleCanvas.height };
  return lastFingertip;
}

// Treats the finger as "pointing at" a code when its tip lands within one
// code-width of the code's center — close enough to be unambiguous without
// requiring pixel-perfect aim.
function isPointingAt(fingertip, location) {
  if (!fingertip) {
    return false;
  }

  const center = centerOf(location);
  return Math.hypot(fingertip.x - center.x, fingertip.y - center.y) < widthOf(location);
}

function drawFingertip(point) {
  overlayCtx.fillStyle = '#ffffff';
  overlayCtx.beginPath();
  overlayCtx.arc(point.x, point.y, screenPx(8), 0, Math.PI * 2);
  overlayCtx.fill();
}

// Approximate size of a QR code in the captured frame, in pixels.
const QR_SIZE = 150;
// jsQR only ever returns one decoded symbol per call, so to find multiple
// codes in a frame we scan overlapping crop windows across the image and
// decode each one separately. The window is bigger than a code (with room
// for its quiet zone) and the step is small enough that the overlap between
// adjacent windows is at least one code-width, so no code can fall entirely
// across a window boundary and get missed.
const TILE_SIZE = QR_SIZE * 2;
const TILE_STEP = QR_SIZE;

function startCamera(facingMode) {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
  }

  const constraints = {
    video: {
      // Lower than the camera's max resolution on purpose: scanning cost
      // scales with frame area (more/bigger tiles to decode), and QR codes
      // don't need full HD detail to read reliably at normal distances.
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: { ideal: facingMode },
    },
    audio: false,
  };

  return navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      currentStream = stream;
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error('Unable to access webcam:', error);
    });
}

video.addEventListener('loadedmetadata', () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = video.videoWidth;
  sampleCanvas.height = video.videoHeight;
  sampleCtx = sampleCanvas.getContext('2d', { willReadFrequently: true });

  if (!tickStarted) {
    tickStarted = true;
    requestAnimationFrame(tick);
  }
});

// There's no on-screen UI, so switching between front and back cameras is a
// double-click/double-tap anywhere, or the "C" key.
function toggleCamera() {
  currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  startCamera(currentFacingMode);
}

document.addEventListener('dblclick', toggleCamera);
document.addEventListener('keydown', (event) => {
  if (event.key === 'c' || event.key === 'C') {
    toggleCamera();
  }
});

startCamera(currentFacingMode);

// QR text -> { data, mouse, location, lastSeen, revealUntil }
const tracked = new Map();

function updateTracked(detections, now) {
  for (const { data, location } of detections) {
    const existing = tracked.get(data);
    if (existing) {
      existing.location = location;
      existing.lastSeen = now;
    } else {
      tracked.set(data, { data, mouse: getMouse(data), location, lastSeen: now, revealUntil: 0 });
    }
  }

  for (const [data, code] of tracked) {
    if (now - code.lastSeen > TRACK_LINGER_MS) {
      tracked.delete(data);
    }
  }
}

async function tick() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    const detections = await detectCodes();
    const now = performance.now();
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    updateTracked(detections, now);
    const codes = [...tracked.values()];
    const fingertip = getIndexFingertip();

    const groups = findGroups(codes.filter((code) => code.mouse));
    const grouped = new Set(groups.flat());

    for (const group of groups) {
      drawGroup(group);
    }

    for (const code of codes) {
      const { mouse, location } = code;
      const color = mouse ? mouse.color : UNKNOWN_CODE_COLOR;
      drawBox(location, color);

      if (!mouse) {
        drawNameTag(location, code.data, color);
        continue;
      }

      // Mice in a group show the group's personality instead of their own.
      if (!grouped.has(code) && isPointingAt(fingertip, location)) {
        code.revealUntil = now + REVEAL_LINGER_MS;
      }

      if (!grouped.has(code) && now < code.revealUntil) {
        drawPersonality(location, mouse);
      } else {
        drawNameTag(location, mouse.name, color);
      }
    }

    if (fingertip) {
      drawFingertip(fingertip);
    }
  }

  requestAnimationFrame(tick);
}

// Clusters mice whose codes sit close together. A chain counts too: if A is
// next to B and B is next to C, all three form one group. Only groups of two
// or more are returned.
function findGroups(codes) {
  const groupOf = new Map(codes.map((code) => [code, [code]]));

  for (let i = 0; i < codes.length; i++) {
    for (let j = i + 1; j < codes.length; j++) {
      const a = codes[i];
      const b = codes[j];
      const groupA = groupOf.get(a);
      const groupB = groupOf.get(b);
      if (groupA === groupB || !areTogether(a.location, b.location)) {
        continue;
      }
      groupA.push(...groupB);
      groupB.forEach((code) => groupOf.set(code, groupA));
    }
  }

  return [...new Set(groupOf.values())].filter((group) => group.length > 1);
}

function areTogether(a, b) {
  const centerA = centerOf(a);
  const centerB = centerOf(b);
  const size = Math.max(widthOf(a), widthOf(b));
  return Math.hypot(centerA.x - centerB.x, centerA.y - centerB.y) < size * GROUP_DISTANCE;
}

// The browser's built-in QR detector (Chrome on macOS and Android, among
// others) finds every code in the frame in one hardware-accelerated call, so
// it's used whenever available. jsQR is the fallback (e.g. iPhone Safari).
let barcodeDetector = null;

async function initBarcodeDetector() {
  if (!('BarcodeDetector' in window)) {
    return;
  }
  const formats = await BarcodeDetector.getSupportedFormats();
  if (formats.includes('qr_code')) {
    barcodeDetector = new BarcodeDetector({ formats: ['qr_code'] });
  }
}

initBarcodeDetector().catch((error) => {
  console.error('Native QR detection unavailable, using jsQR:', error);
});

async function detectCodes() {
  if (barcodeDetector) {
    try {
      const codes = await barcodeDetector.detect(video);
      return dedupeDetections(codes.map(({ rawValue, cornerPoints }) => {
        const [topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner] = cornerPoints;
        return { data: rawValue, location: { topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner } };
      }));
    } catch (error) {
      console.error('Native QR detection failed, switching to jsQR:', error);
      barcodeDetector = null;
    }
  }

  sampleCtx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
  return dedupeDetections([...rescanTrackedCodes(), ...sweepNextTiles()]);
}

// With jsQR, decoding every tile of the frame is the slow part, so the sweep
// that finds new codes is spread across SWEEP_FRAMES frames, a few tiles at a
// time. Codes already found are re-decoded every frame, just in a small
// window around where they last were.
const SWEEP_FRAMES = 6;
// Window size around a known code, in code-widths; leaves room for the code
// to move between frames and for its quiet zone.
const RESCAN_SCALE = 2.2;
let sweepIndex = 0;

function sweepNextTiles() {
  const tiles = [];
  for (const y of getTilePositions(sampleCanvas.height)) {
    for (const x of getTilePositions(sampleCanvas.width)) {
      tiles.push({ x, y });
    }
  }

  const perFrame = Math.ceil(tiles.length / SWEEP_FRAMES);
  if (sweepIndex >= tiles.length) {
    sweepIndex = 0;
  }
  const batch = tiles.slice(sweepIndex, sweepIndex + perFrame);
  sweepIndex += perFrame;

  return batch
    .map(({ x, y }) => decodeRegion(x, y, TILE_SIZE, TILE_SIZE))
    .filter(Boolean);
}

function rescanTrackedCodes() {
  const detections = [];

  for (const code of tracked.values()) {
    const center = centerOf(code.location);
    const size = Math.min(
      Math.ceil(widthOf(code.location) * RESCAN_SCALE),
      sampleCanvas.width,
      sampleCanvas.height,
    );
    const x = Math.round(Math.min(Math.max(center.x - size / 2, 0), sampleCanvas.width - size));
    const y = Math.round(Math.min(Math.max(center.y - size / 2, 0), sampleCanvas.height - size));
    const detection = decodeRegion(x, y, size, size);
    if (detection) {
      detections.push(detection);
    }
  }

  return dedupeDetections(detections);
}

function decodeRegion(x, y, width, height) {
  const region = sampleCtx.getImageData(x, y, width, height);
  // Our codes are printed black-on-white, so skip jsQR's color-inverted
  // decoding pass (its default) — it roughly doubles work per region for a
  // case we never hit.
  const qrCode = jsQR(region.data, width, height, { inversionAttempts: 'dontInvert' });
  return qrCode ? offsetQRCode(qrCode, x, y) : null;
}

// Start offsets for tiles of TILE_SIZE covering `dimension`, stepping by
// TILE_STEP and with a final tile flush against the far edge so the whole
// frame is covered even when it doesn't divide evenly by the step.
function getTilePositions(dimension) {
  if (dimension <= TILE_SIZE) {
    return [0];
  }

  const positions = [];
  for (let pos = 0; pos + TILE_SIZE <= dimension; pos += TILE_STEP) {
    positions.push(pos);
  }

  const lastPosition = dimension - TILE_SIZE;
  if (positions[positions.length - 1] !== lastPosition) {
    positions.push(lastPosition);
  }

  return positions;
}

function offsetQRCode(qrCode, offsetX, offsetY) {
  const shift = (point) => ({ x: point.x + offsetX, y: point.y + offsetY });
  const { topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner } = qrCode.location;

  return {
    data: qrCode.data,
    location: {
      topLeftCorner: shift(topLeftCorner),
      topRightCorner: shift(topRightCorner),
      bottomRightCorner: shift(bottomRightCorner),
      bottomLeftCorner: shift(bottomLeftCorner),
    },
  };
}

// The same QR code is often found in more than one overlapping tile, so
// collapse detections of the same text. Only the same text is collapsed —
// two different mice placed side by side must both survive.
function dedupeDetections(detections) {
  const unique = new Map();
  for (const detection of detections) {
    if (!unique.has(detection.data)) {
      unique.set(detection.data, detection);
    }
  }
  return [...unique.values()];
}

function centerOf(location) {
  const { topLeftCorner, bottomRightCorner } = location;
  return {
    x: (topLeftCorner.x + bottomRightCorner.x) / 2,
    y: (topLeftCorner.y + bottomRightCorner.y) / 2,
  };
}

function widthOf(location) {
  const { topLeftCorner, topRightCorner } = location;
  return Math.hypot(topRightCorner.x - topLeftCorner.x, topRightCorner.y - topLeftCorner.y);
}

function cornersOf(location) {
  const { topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner } = location;
  return [topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner];
}

// The overlay is drawn at the camera's resolution but displayed scaled to fill
// the screen (object-fit: cover), so this converts an on-screen size in CSS
// pixels to canvas pixels. That keeps text the same readable size on a phone
// as on a laptop, regardless of camera resolution.
function screenPx(px) {
  const displayScale = Math.max(overlay.clientWidth / overlay.width, overlay.clientHeight / overlay.height);
  return px / displayScale;
}

function drawBox(location, color) {
  const corners = cornersOf(location);

  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = screenPx(4);
  overlayCtx.beginPath();
  corners.forEach((point, index) => (index === 0
    ? overlayCtx.moveTo(point.x, point.y)
    : overlayCtx.lineTo(point.x, point.y)));
  overlayCtx.closePath();
  overlayCtx.stroke();
}

// Name only, centered just above the code.
function drawNameTag(location, name, color) {
  const top = Math.min(...cornersOf(location).map((point) => point.y));
  const { x } = centerOf(location);
  drawLabel(x, top, name, [], color, 'above');
}

// Name plus personality traits, centered just below the code.
function drawPersonality(location, mouse) {
  const bottom = Math.max(...cornersOf(location).map((point) => point.y));
  const { x } = centerOf(location);
  drawLabel(x, bottom, mouse.name, mouse.tags, mouse.color, 'below', widthOf(location) * 1.5);
}

// Links the group's codes with a line, names each mouse, and shows the
// group's combined personality below the whole group.
function drawGroup(group) {
  const centers = group.map((code) => centerOf(code.location));

  overlayCtx.strokeStyle = GROUP_COLOR;
  overlayCtx.lineWidth = screenPx(3);
  overlayCtx.setLineDash([screenPx(10), screenPx(8)]);
  overlayCtx.beginPath();
  for (let i = 0; i < centers.length; i++) {
    for (let j = i + 1; j < centers.length; j++) {
      if (areTogether(group[i].location, group[j].location)) {
        overlayCtx.moveTo(centers[i].x, centers[i].y);
        overlayCtx.lineTo(centers[j].x, centers[j].y);
      }
    }
  }
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);

  const personality = getGroupPersonality(group.map((code) => code.mouse));
  const corners = group.flatMap((code) => cornersOf(code.location));
  const left = Math.min(...corners.map((point) => point.x));
  const right = Math.max(...corners.map((point) => point.x));
  const bottom = Math.max(...corners.map((point) => point.y));
  drawLabel((left + right) / 2, bottom, personality.title, personality.tags, GROUP_COLOR, 'below', right - left);
}

// Splits tags into lines no wider than maxWidth, using the current font.
function wrapTags(tags, maxWidth) {
  const lines = [];
  let line = '';
  for (const tag of tags) {
    const candidate = line ? `${line} · ${tag}` : tag;
    if (line && overlayCtx.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = tag;
    } else {
      line = candidate;
    }
  }
  if (line) {
    lines.push(line);
  }
  return lines;
}

// Draws a title (in `color`) with tags wrapped underneath on a dark panel,
// horizontally centered on centerX and placed just above or below edgeY.
function drawLabel(centerX, edgeY, title, tags, color, placement, wrapWidth = 0) {
  const titleSize = screenPx(26);
  const tagSize = screenPx(18);
  const lineHeight = tagSize * 1.4;
  const padding = titleSize * 0.4;

  overlayCtx.textBaseline = 'top';
  const tagFont = `600 ${tagSize}px Nunito, sans-serif`;
  const titleFont = `700 ${titleSize}px Fredoka, Nunito, sans-serif`;

  overlayCtx.font = tagFont;
  const lines = wrapTags(tags, Math.max(wrapWidth, titleSize * 9));
  let width = Math.max(0, ...lines.map((line) => overlayCtx.measureText(line).width));
  overlayCtx.font = titleFont;
  width = Math.max(width, overlayCtx.measureText(title).width);
  const height = titleSize + lines.length * lineHeight;

  const x = centerX - width / 2;
  const y = placement === 'above'
    ? edgeY - padding * 2 - height
    : edgeY + padding * 2;

  overlayCtx.fillStyle = 'rgba(30, 22, 17, 0.8)';
  overlayCtx.beginPath();
  overlayCtx.roundRect(x - padding, y - padding, width + padding * 2, height + padding * 2, padding);
  overlayCtx.fill();

  overlayCtx.fillStyle = color;
  overlayCtx.fillText(title, x, y);

  overlayCtx.font = tagFont;
  overlayCtx.fillStyle = '#ffffff';
  lines.forEach((line, index) => {
    overlayCtx.fillText(line, x, y + titleSize + (lineHeight - tagSize) + index * lineHeight);
  });
}
