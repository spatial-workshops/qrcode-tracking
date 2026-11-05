const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

const VIDEO_CONSTRAINTS = {
  video: {
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  },
  audio: false,
};

// BarcodeDetector finds every QR code in the full frame in a single call, at
// any size, so no tiling or size hints are needed. Browsers with a native
// implementation (Chrome/Edge/Android/Safari) use hardware-backed code; others
// fall back to the zxing-wasm based polyfill.
const POLYFILL_URL = 'https://cdn.jsdelivr.net/npm/barcode-detector@3.2.2/+esm';

const MEDIAPIPE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MAX_HANDS = 2;
const INDEX_FINGER_TIP = 8;
// A QR code is selected when an index fingertip is within this many pixels
// (in video frame coordinates) of it.
const SELECT_DISTANCE = 50;

// QR codes whose contents start with this prefix mark the corners of a zone;
// the rest of the contents is the zone's identifier.
const ZONE_PREFIX = 'zone:';
const MIN_ZONE_CORNERS = 3;
// QR codes whose contents start with this prefix cast a ray from their top
// edge in the direction the code is facing "up", out to the edge of the frame.
const RAY_PREFIX = 'ray:';

let detector;
let isDetecting = false;
let handLandmarker;
let handConnections = [];
let lastVideoTime = -1;

// Latest results from each tracker, redrawn together on every frame.
let barcodes = [];
let hands = [];

async function createDetector() {
  const formats = ['qr_code'];
  if ('BarcodeDetector' in window) {
    const supported = await BarcodeDetector.getSupportedFormats();
    if (supported.includes('qr_code')) {
      return new BarcodeDetector({ formats });
    }
  }
  const { BarcodeDetector: Polyfill } = await import(POLYFILL_URL);
  return new Polyfill({ formats });
}

// Hand tracking is loaded in the background so a failure to fetch MediaPipe
// never blocks QR tracking.
async function createHandLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(`${MEDIAPIPE_URL}/+esm`);
  const fileset = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_URL}/wasm`);
  handConnections = HandLandmarker.HAND_CONNECTIONS;
  const create = (delegate) => HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: MAX_HANDS,
  });
  return create('GPU').catch(() => create('CPU'));
}

navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS)
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((error) => {
    console.error('Unable to access webcam:', error);
  });

video.addEventListener('loadedmetadata', async () => {
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  detector = await createDetector();
  requestAnimationFrame(tick);

  createHandLandmarker()
    .then((landmarker) => { handLandmarker = landmarker; })
    .catch((error) => console.error('Unable to load hand tracking:', error));
});

function tick() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    // Detection is async; skip frames while one is in flight rather than queue.
    if (!isDetecting) {
      isDetecting = true;
      detector.detect(video)
        .then((results) => { barcodes = results; })
        .catch((error) => console.error('Detection failed:', error))
        .finally(() => { isDetecting = false; });
    }

    // Hand detection is synchronous and only needs to run once per new frame.
    if (handLandmarker && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = handLandmarker.detectForVideo(video, performance.now()).landmarks;
    }
  }

  render();
  requestAnimationFrame(tick);
}

function render() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const fingertips = hands.map((landmarks) => ({
    x: landmarks[INDEX_FINGER_TIP].x * overlay.width,
    y: landmarks[INDEX_FINGER_TIP].y * overlay.height,
  }));

  for (const [id, corners] of findZones(barcodes)) {
    drawZone(id, corners);
  }

  for (const barcode of barcodes) {
    if (barcode.rawValue.startsWith(RAY_PREFIX)) {
      drawRay(barcode.cornerPoints);
    }
  }

  for (const barcode of barcodes) {
    const location = toLocation(barcode.cornerPoints);
    if (fingertips.some((tip) => isNear(tip, barcode.cornerPoints, SELECT_DISTANCE))) {
      fillQuad(barcode.cornerPoints);
    }
    drawBox(location);
    drawLabel(location, barcode.rawValue);
  }
  for (const landmarks of hands) {
    drawHand(landmarks);
  }
}

// Groups zone QR codes by identifier and returns a Map of id -> polygon, using
// each code's center as a corner. Detection order is arbitrary, so corners are
// sorted clockwise around their centroid. Zones with too few visible corners
// are left out.
function findZones(barcodes) {
  const groups = new Map();
  for (const barcode of barcodes) {
    if (!barcode.rawValue.startsWith(ZONE_PREFIX)) {
      continue;
    }
    const id = barcode.rawValue.slice(ZONE_PREFIX.length);
    if (!groups.has(id)) {
      groups.set(id, []);
    }
    groups.get(id).push(centroid(barcode.cornerPoints));
  }

  const zones = new Map();
  for (const [id, corners] of groups) {
    if (corners.length >= MIN_ZONE_CORNERS) {
      zones.set(id, sortClockwise(corners));
    }
  }
  return zones;
}

function centroid(points) {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

// With y pointing down on screen, increasing atan2 angle runs clockwise.
function sortClockwise(points) {
  const center = centroid(points);
  const angle = (p) => Math.atan2(p.y - center.y, p.x - center.x);
  return [...points].sort((a, b) => angle(a) - angle(b));
}

// cornerPoints are ordered top-left, top-right, bottom-right, bottom-left.
function toLocation([topLeft, topRight, bottomRight, bottomLeft]) {
  return {
    topLeftCorner: topLeft,
    topRightCorner: topRight,
    bottomRightCorner: bottomRight,
    bottomLeftCorner: bottomLeft,
  };
}

function drawBox(location) {
  const { topLeftCorner, topRightCorner, bottomRightCorner, bottomLeftCorner } = location;

  overlayCtx.strokeStyle = '#00ff00';
  overlayCtx.lineWidth = Math.max(4, overlay.width * 0.006);
  overlayCtx.beginPath();
  overlayCtx.moveTo(topLeftCorner.x, topLeftCorner.y);
  overlayCtx.lineTo(topRightCorner.x, topRightCorner.y);
  overlayCtx.lineTo(bottomRightCorner.x, bottomRightCorner.y);
  overlayCtx.lineTo(bottomLeftCorner.x, bottomLeftCorner.y);
  overlayCtx.closePath();
  overlayCtx.stroke();
}

function drawLabel(location, text) {
  const { bottomLeftCorner, bottomRightCorner } = location;

  const fontSize = Math.max(16, overlay.width * 0.02);
  const padding = fontSize * 0.25;
  const x = Math.min(bottomLeftCorner.x, bottomRightCorner.x);
  const y = Math.max(bottomLeftCorner.y, bottomRightCorner.y) + padding;

  overlayCtx.font = `${fontSize}px monospace`;
  overlayCtx.textBaseline = 'top';
  const textWidth = overlayCtx.measureText(text).width;

  overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  overlayCtx.fillRect(x - padding, y - padding, textWidth + padding * 2, fontSize + padding * 2);

  overlayCtx.fillStyle = '#00ff00';
  overlayCtx.fillText(text, x, y);
}

// Landmarks are normalized to [0, 1], so scale them to the overlay.
function drawHand(landmarks) {
  overlayCtx.strokeStyle = '#ff0000';
  overlayCtx.lineWidth = Math.max(4, overlay.width * 0.004);
  overlayCtx.lineCap = 'round';
  overlayCtx.beginPath();
  for (const { start, end } of handConnections) {
    overlayCtx.moveTo(landmarks[start].x * overlay.width, landmarks[start].y * overlay.height);
    overlayCtx.lineTo(landmarks[end].x * overlay.width, landmarks[end].y * overlay.height);
  }
  overlayCtx.stroke();
}

// True if the point is inside the polygon or within `distance` of its edge.
function isNear(point, polygon, distance) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    if (distanceToSegment(point, a, b) <= distance) {
      return true;
    }
    if ((a.y > point.y) !== (b.y > point.y) &&
        point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

function distanceToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared === 0
    ? 0
    : Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSquared));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

function fillQuad(points) {
  overlayCtx.fillStyle = 'rgba(255, 255, 153, 0.6)';
  overlayCtx.beginPath();
  overlayCtx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) {
    overlayCtx.lineTo(point.x, point.y);
  }
  overlayCtx.closePath();
  overlayCtx.fill();
}

// The ray starts at the middle of the code's top edge and points away from the
// middle of its bottom edge, so it follows the code's rotation in the frame.
function drawRay([topLeft, topRight, bottomRight, bottomLeft]) {
  const start = midpoint(topLeft, topRight);
  const bottom = midpoint(bottomLeft, bottomRight);
  const end = rayToFrameEdge(start, { x: start.x - bottom.x, y: start.y - bottom.y });
  if (!end) {
    return;
  }

  overlayCtx.strokeStyle = '#00ffff';
  overlayCtx.lineWidth = Math.max(4, overlay.width * 0.004);
  overlayCtx.lineCap = 'round';
  overlayCtx.beginPath();
  overlayCtx.moveTo(start.x, start.y);
  overlayCtx.lineTo(end.x, end.y);
  overlayCtx.stroke();
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// Where a ray from `origin` along `direction` leaves the overlay, or null if
// the direction is degenerate.
function rayToFrameEdge(origin, direction) {
  const distances = [];
  if (direction.x > 0) distances.push((overlay.width - origin.x) / direction.x);
  if (direction.x < 0) distances.push(-origin.x / direction.x);
  if (direction.y > 0) distances.push((overlay.height - origin.y) / direction.y);
  if (direction.y < 0) distances.push(-origin.y / direction.y);
  if (distances.length === 0) {
    return null;
  }

  const t = Math.max(0, Math.min(...distances));
  return { x: origin.x + direction.x * t, y: origin.y + direction.y * t };
}
