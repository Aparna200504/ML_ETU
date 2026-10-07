/**
 * usePoseTracker.js — v2.0
 *
 * Changes from v1:
 *   - startTracking(videoType) now accepts a video type ('up-down' | 'right-left')
 *     and passes it to /start_session so the backend uses the correct reference.
 *
 * Coordinate system (same as extract_reference.py):
 *   yaw   = (nose.x - eye_mid_x) / eye_span
 *   pitch = (nose.y - eye_mid_y) / face_height
 *   roll  = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
 */

import { useRef, useCallback, useState, useEffect } from 'react';
import { 
    FaceLandmarker,
    FilesetResolver,
    HandLandmarker
  } from '@mediapipe/tasks-vision';

// const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API = 'https://mletu-production.up.railway.app';


const SAMPLE_INTERVAL_MS = 167; // ~6 frames/sec — matches extract_reference.py's sample_rate=6,
                                 // so fast head turns aren't clipped below the true peak angle

function isValidLandmark(point) {
  return (
    point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    Number.isFinite(point.z)
  );
}


const FINGER_LANDMARKS = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  pinky:  [17, 18, 19, 20],
};


/**
 * Determine which individual fingers have usable landmarks.
 *
 * IMPORTANT:
 * MediaPipe does not provide a true visibility score for every
 * hand landmark. Therefore this function does NOT claim that a
 * finger is visually visible with certainty.
 *
 * It only determines whether the landmarks needed for calculating
 * that finger's features are usable.
 */
function calculateFingerValidity(landmarks) {
  const result = {};

  for (const [finger, indices] of Object.entries(FINGER_LANDMARKS)) {
    const validCount = indices.filter(
      index => isValidLandmark(landmarks[index])
    ).length;

    result[finger] = {
      valid: validCount === indices.length,
      validLandmarks: validCount,
      totalLandmarks: indices.length,
    };
  }

  return result;
}


function calculateHandReliability(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    return {
      reliable: false,
      visibleLandmarks: 0,
      confidence: 0,
      fingers: {},
    };
  }

  const validLandmarks = landmarks.filter(isValidLandmark);

  const visibilityRatio =
    validLandmarks.length / 21;

  const fingers = calculateFingerValidity(landmarks);

  return {
    reliable: visibilityRatio >= 0.8,
    visibleLandmarks: validLandmarks.length,
    confidence: Number(visibilityRatio.toFixed(3)),
    fingers,
  };
}
                                 
const WASM_BASE = `${window.location.origin}/mediapipe/wasm`;

const MP_MODELS = [
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/face_landmarker.task',
];

const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';


// ---------------------------------------------------------------------------
// HandLandmarker singleton
// ---------------------------------------------------------------------------

let _handLandmarkerPromise = null;

async function getHandLandmarker() {
  if (_handLandmarkerPromise) return _handLandmarkerPromise;

  _handLandmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

    try {
      const hl = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      console.log('[MP] HandLandmarker ready.');
      return hl;
    } catch (gpuError) {
      console.warn(
        '[MP] Hand GPU failed. Trying CPU...',
        gpuError.message
      );

      const hl = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      console.log('[MP] HandLandmarker ready (CPU fallback).');
      return hl;
    }
  })();

  return _handLandmarkerPromise;
}

// ---------------------------------------------------------------------------
// FaceLandmarker singleton
// ---------------------------------------------------------------------------
let _landmarkerPromise = null;

async function getLandmarker() {
  if (_landmarkerPromise) return _landmarkerPromise;

  _landmarkerPromise = (async () => {
    let lastErr;
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

    for (const modelUrl of MP_MODELS) {
      try {
        const fl = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelUrl, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFacialTransformationMatrixes: false,
        });
        console.log('[MP] FaceLandmarker ready. Model:', modelUrl);
        return fl;
      } catch (e) {
        console.warn('[MP] Model load failed:', modelUrl, e.message);
        lastErr = e;
      }
    }

    // Retry with CPU delegate if GPU/WebGL is unavailable
    for (const modelUrl of MP_MODELS) {
      try {
        const fl = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFacialTransformationMatrixes: false,
        });
        console.log('[MP] FaceLandmarker ready (CPU fallback). Model:', modelUrl);
        return fl;
      } catch (e) {
        console.warn('[MP] CPU fallback failed:', modelUrl, e.message);
        lastErr = e;
      }
    }

    _landmarkerPromise = null;
    throw lastErr || new Error('FaceLandmarker init failed on all model URLs');
  })();

  return _landmarkerPromise;
}

// ---------------------------------------------------------------------------
// CORE: compute yaw/pitch/roll from landmarks — identical math to Python
// ---------------------------------------------------------------------------
function computePoseFromLandmarks(landmarks) {
  const nose     = landmarks[1];    // nose_tip
  const leftEye  = landmarks[33];   // left_eye_outer
  const rightEye = landmarks[263];  // right_eye_outer
  const chin     = landmarks[152];
  const forehead = landmarks[10];

  // YAW: nose horizontal offset from eye midpoint, normalized by eye span
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const eyeSpan = Math.abs(rightEye.x - leftEye.x) || 0.001;
  const yaw     = (nose.x - eyeMidX) / eyeSpan;

  // PITCH: nose vertical offset from eye midpoint, normalized by face height
  const eyeMidY = (leftEye.y + rightEye.y) / 2;
  const faceH   = Math.abs(chin.y - forehead.y) || 0.001;
  const pitch   = (nose.y - eyeMidY) / faceH;

  // ROLL: eye line angle
  const dx   = rightEye.x - leftEye.x;
  const dy   = rightEye.y - leftEye.y;
  const roll = Math.atan2(dy, dx);

  return {
    yaw:   parseFloat(yaw.toFixed(6)),
    pitch: parseFloat(pitch.toFixed(6)),
    roll:  parseFloat(roll.toFixed(6)),
  };
}

// ---------------------------------------------------------------------------
// Detect pose from webcam frame
// ---------------------------------------------------------------------------
let _lastTime = -1;
let _lastPose = null;

async function detectPose(videoEl) {
  let landmarker;
  try { landmarker = await getLandmarker(); } catch { return null; }
  if (!videoEl || videoEl.readyState < 2) return null;

  const t = videoEl.currentTime;
  if (t === _lastTime) return _lastPose;
  _lastTime = t;

  let result;
  try { result = landmarker.detectForVideo(videoEl, performance.now()); }
  catch { return null; }

  if (!result?.faceLandmarks?.length) { _lastPose = null; return null; }

  const pose = computePoseFromLandmarks(result.faceLandmarks[0]);
  _lastPose = pose;
  return pose;
}

async function detectHands(videoEl) {
  let handLandmarker;

  try {
    handLandmarker = await getHandLandmarker();
  } catch {
    return null;
  }

  if (!videoEl || videoEl.readyState < 2) return null;

  let result;

  try {
    result = handLandmarker.detectForVideo(
      videoEl,
      performance.now()
    );
  } catch {
    return null;
  }

  if (!result?.landmarks?.length) {
    return {
      left: null,
      right: null,
    };
  }

  const hands = {
    left: null,
    right: null,
  };

  result.landmarks.forEach((landmarks, index) => {
    const handedness =
      result.handednesses?.[index]?.[0]?.categoryName;

    const handData = extractHandFeatures(landmarks);

    if (handedness === 'Left') {
      hands.left = handData;
    } else if (handedness === 'Right') {
      hands.right = handData;
    }
  });

  return hands;
}

// ---------------------------------------------------------------------------
// Extract hand features for mudra recognition
//
// Input:
//   21 MediaPipe hand landmarks
//
// Output:
//   Raw landmarks + normalized geometry + joint angles + distances
//
// The raw landmarks are intentionally preserved because later ML models
// may need information that our manually engineered features don't capture.
// ---------------------------------------------------------------------------

function distance3D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z || 0) - (b.z || 0);

  return Math.sqrt(
    dx * dx +
    dy * dy +
    dz * dz
  );
}

function angle3D(a, b, c) {
  // Angle ABC
  const v1 = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z || 0) - (b.z || 0),
  };

  const v2 = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z || 0) - (b.z || 0),
  };

  const dot =
    v1.x * v2.x +
    v1.y * v2.y +
    v1.z * v2.z;

  const mag1 = Math.sqrt(
    v1.x ** 2 +
    v1.y ** 2 +
    v1.z ** 2
  );

  const mag2 = Math.sqrt(
    v2.x ** 2 +
    v2.y ** 2 +
    v2.z ** 2
  );

  if (mag1 === 0 || mag2 === 0) return 0;

  const cosine = Math.max(
    -1,
    Math.min(1, dot / (mag1 * mag2))
  );

  return Math.acos(cosine) * (180 / Math.PI);
}

function vectorMagnitude(v) {
  return Math.sqrt(
    v.x * v.x +
    v.y * v.y +
    v.z * v.z
  );
}

function normalizeVector(v) {
  const mag = vectorMagnitude(v) || 0.001;

  return {
    x: v.x / mag,
    y: v.y / mag,
    z: v.z / mag,
  };
}

function calculateWristOrientation(landmarks) {
  const wrist = landmarks[0];
  const indexMCP = landmarks[5];
  const pinkyMCP = landmarks[17];
  const middleMCP = landmarks[9];

  // Wrist → middle finger direction
  const forward = normalizeVector({
    x: middleMCP.x - wrist.x,
    y: middleMCP.y - wrist.y,
    z: (middleMCP.z || 0) - (wrist.z || 0),
  });

  // Index MCP → Pinky MCP
  const side = normalizeVector({
    x: pinkyMCP.x - indexMCP.x,
    y: pinkyMCP.y - indexMCP.y,
    z: (pinkyMCP.z || 0) - (indexMCP.z || 0),
  });

  // Palm normal
  const palmNormal = normalizeVector({
    x:
      forward.y * side.z -
      forward.z * side.y,

    y:
      forward.z * side.x -
      forward.x * side.z,

    z:
      forward.x * side.y -
      forward.y * side.x,
  });

  // Convert orientation vectors to angles
  const yaw =
    Math.atan2(
      forward.x,
      forward.z
    ) * (180 / Math.PI);

  const pitch =
    Math.atan2(
      -forward.y,
      Math.sqrt(
        forward.x ** 2 +
        forward.z ** 2
      )
    ) * (180 / Math.PI);

  const roll =
    Math.atan2(
      side.y,
      side.x
    ) * (180 / Math.PI);

  return {
    yaw: Number(yaw.toFixed(3)),
    pitch: Number(pitch.toFixed(3)),
    roll: Number(roll.toFixed(3)),

    direction: {
      x: Number(forward.x.toFixed(6)),
      y: Number(forward.y.toFixed(6)),
      z: Number(forward.z.toFixed(6)),
    },

    palmNormal: {
      x: Number(palmNormal.x.toFixed(6)),
      y: Number(palmNormal.y.toFixed(6)),
      z: Number(palmNormal.z.toFixed(6)),
    },
  };
}

// ---------------------------------------------------------------------------  
// Normalize hand landmarks to a wrist-centered coordinate system
// and scale them by the palm size (wrist → middle finger MCP).
// ---------------------------------------------------------------------------
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // Palm scale:
  // wrist → middle-finger MCP
  const palmScale = distance3D(
    landmarks[0],
    landmarks[9]
  ) || 0.001;

  return landmarks.map((point) => ({
    x: Number(((point.x - wrist.x) / palmScale).toFixed(6)),
    y: Number(((point.y - wrist.y) / palmScale).toFixed(6)),
    z: Number(((point.z - wrist.z) / palmScale).toFixed(6)),
  }));
}

/**
 * function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // Palm scale:
  // wrist → middle-finger MCP
  const palmScale = distance3D(
    landmarks[0],
    landmarks[9]
  ) || 0.001;

  return landmarks.map((point) => ({
    x: Number(((point.x - wrist.x) / palmScale).toFixed(6)),
    y: Number(((point.y - wrist.y) / palmScale).toFixed(6)),
    z: Number(((point.z - wrist.z) / palmScale).toFixed(6)),
  }));
}
 */

/**
 * function calculateHandReliability(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    return {
      reliable: false,
      visibleLandmarks: 0,
      confidence: 0,
    };
  }

  // MediaPipe landmark coordinates are still returned even
  // when some parts of the hand are occluded.
  // Check whether landmarks contain valid coordinates.
  const validLandmarks = landmarks.filter(
    (p) =>
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      Number.isFinite(p.z)
  );

  const visibilityRatio = validLandmarks.length / 21;

  return {
    reliable: visibilityRatio >= 0.8,
    visibleLandmarks: validLandmarks.length,
    confidence: Number(visibilityRatio.toFixed(3)),
  };
}
 */

function calculateHandReliability(landmarks) {
  if (!landmarks || landmarks.length !== 21) {
    return {
      reliable: false,
      visibleLandmarks: 0,
      confidence: 0,
    };
  }

  // MediaPipe landmark coordinates are still returned even
  // when some parts of the hand are occluded.
  // Check whether landmarks contain valid coordinates.
  const validLandmarks = landmarks.filter(
    (p) =>
      Number.isFinite(p.x) &&
      Number.isFinite(p.y) &&
      Number.isFinite(p.z)
  );

  const visibilityRatio = validLandmarks.length / 21;

  return {
    reliable: visibilityRatio >= 0.8,
    visibleLandmarks: validLandmarks.length,
    confidence: Number(visibilityRatio.toFixed(3)),
  };
}


function extractHandFeatures(landmarks) {
    const reliability = calculateHandReliability(landmarks);
  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  // ---------------------------------------------------------
  // 1. Normalized landmarks
  // ---------------------------------------------------------

  const normalizedLandmarks =
    normalizeLandmarks(landmarks);

    const wrist = landmarks[0];

    const wristPosition = {
      x: Number(wrist.x.toFixed(6)),
      y: Number(wrist.y.toFixed(6)),
      z: Number((wrist.z || 0).toFixed(6)),
    };

    const wristOrientation =
      calculateWristOrientation(landmarks);

      
  // ---------------------------------------------------------
  // 2. Finger joint angles
  //
  // Thumb:  1-2-3-4
  // Index:  5-6-7-8
  // Middle: 9-10-11-12
  // Ring:   13-14-15-16
  // Pinky:  17-18-19-20
  // ---------------------------------------------------------
  
    const fingerValidity = reliability.fingers;

    const jointAngles = {
      thumb: fingerValidity.thumb.valid
        ? [
            angle3D(landmarks[1], landmarks[2], landmarks[3]),
            angle3D(landmarks[2], landmarks[3], landmarks[4]),
          ]
        : null,

      index: fingerValidity.index.valid
        ? [
            angle3D(landmarks[5], landmarks[6], landmarks[7]),
            angle3D(landmarks[6], landmarks[7], landmarks[8]),
          ]
        : null,

      middle: fingerValidity.middle.valid
        ? [
            angle3D(landmarks[9], landmarks[10], landmarks[11]),
            angle3D(landmarks[10], landmarks[11], landmarks[12]),
          ]
        : null,

      ring: fingerValidity.ring.valid
        ? [
            angle3D(landmarks[13], landmarks[14], landmarks[15]),
            angle3D(landmarks[14], landmarks[15], landmarks[16]),
          ]
        : null,

      pinky: fingerValidity.pinky.valid
        ? [
            angle3D(landmarks[17], landmarks[18], landmarks[19]),
            angle3D(landmarks[18], landmarks[19], landmarks[20]),
          ]
        : null,
    };

  /** 
   * const jointAngles = {
      thumb: [
        angle3D(landmarks[1], landmarks[2], landmarks[3]),
        angle3D(landmarks[2], landmarks[3], landmarks[4]),
      ],

      index: [
        angle3D(landmarks[5], landmarks[6], landmarks[7]),
        angle3D(landmarks[6], landmarks[7], landmarks[8]),
      ],

      middle: [
        angle3D(landmarks[9], landmarks[10], landmarks[11]),
        angle3D(landmarks[10], landmarks[11], landmarks[12]),
      ],

      ring: [
        angle3D(landmarks[13], landmarks[14], landmarks[15]),
        angle3D(landmarks[14], landmarks[15], landmarks[16]),
      ],

      pinky: [
        angle3D(landmarks[17], landmarks[18], landmarks[19]),
        angle3D(landmarks[18], landmarks[19], landmarks[20]),
      ],
    };

  */
  
  // ---------------------------------------------------------
  // 3. Important fingertip distances
  // ---------------------------------------------------------
    
    const fingertipDistances = {
      thumb_index:
        fingerValidity.thumb.valid && fingerValidity.index.valid
          ? distance3D(landmarks[4], landmarks[8])
          : null,

      thumb_middle:
        fingerValidity.thumb.valid && fingerValidity.middle.valid
          ? distance3D(landmarks[4], landmarks[12])
          : null,

      thumb_ring:
        fingerValidity.thumb.valid && fingerValidity.ring.valid
          ? distance3D(landmarks[4], landmarks[16])
          : null,

      thumb_pinky:
        fingerValidity.thumb.valid && fingerValidity.pinky.valid
          ? distance3D(landmarks[4], landmarks[20])
          : null,

      index_middle:
        fingerValidity.index.valid && fingerValidity.middle.valid
          ? distance3D(landmarks[8], landmarks[12])
          : null,

      middle_ring:
        fingerValidity.middle.valid && fingerValidity.ring.valid
          ? distance3D(landmarks[12], landmarks[16])
          : null,

      ring_pinky:
        fingerValidity.ring.valid && fingerValidity.pinky.valid
          ? distance3D(landmarks[16], landmarks[20])
          : null,
    };

  /**
   * const fingertipDistances = {
    thumb_index: distance3D(
      landmarks[4],
      landmarks[8]
    ),

    thumb_middle: distance3D(
      landmarks[4],
      landmarks[12]
    ),

    thumb_ring: distance3D(
      landmarks[4],
      landmarks[16]
    ),

    thumb_pinky: distance3D(
      landmarks[4],
      landmarks[20]
    ),

    index_middle: distance3D(
      landmarks[8],
      landmarks[12]
    ),

    middle_ring: distance3D(
      landmarks[12],
      landmarks[16]
    ),

    ring_pinky: distance3D(
      landmarks[16],
      landmarks[20]
    ),
  };
*/
  
  // ---------------------------------------------------------
  // 4. Finger lengths
  //
  // Useful for understanding whether fingers are extended,
  // folded, or positioned close to the palm.
  // ---------------------------------------------------------

  const fingerLengths = {
    thumb:
      fingerValidity.thumb.valid
        ? distance3D(landmarks[1], landmarks[4])
        : null,

    index:
      fingerValidity.index.valid
        ? distance3D(landmarks[5], landmarks[8])
        : null,

    middle:
      fingerValidity.middle.valid
        ? distance3D(landmarks[9], landmarks[12])
        : null,

    ring:
      fingerValidity.ring.valid
        ? distance3D(landmarks[13], landmarks[16])
        : null,

    pinky:
      fingerValidity.pinky.valid
        ? distance3D(landmarks[17], landmarks[20])
        : null,
  };

/**
 *const fingerLengths = {
    thumb:
      distance3D(landmarks[1], landmarks[4]),

    index:
      distance3D(landmarks[5], landmarks[8]),

    middle:
      distance3D(landmarks[9], landmarks[12]),

    ring:
      distance3D(landmarks[13], landmarks[16]),

    pinky:
      distance3D(landmarks[17], landmarks[20]),
  };

 * 
 */

  // ---------------------------------------------------------
  // 5. Palm orientation
  //
  // Construct two vectors from the wrist:
  // wrist → index MCP
  // wrist → pinky MCP
  //
  // Their cross product gives an approximate palm normal.
  // ---------------------------------------------------------

  //const wrist = landmarks[0];
  const indexMCP = landmarks[5];
  const pinkyMCP = landmarks[17];

  const v1 = {
    x: indexMCP.x - wrist.x,
    y: indexMCP.y - wrist.y,
    z: (indexMCP.z || 0) - (wrist.z || 0),
  };

  const v2 = {
    x: pinkyMCP.x - wrist.x,
    y: pinkyMCP.y - wrist.y,
    z: (pinkyMCP.z || 0) - (wrist.z || 0),
  };

  const palmNormal = {
    x: v1.y * v2.z - v1.z * v2.y,
    y: v1.z * v2.x - v1.x * v2.z,
    z: v1.x * v2.y - v1.y * v2.x,
  };

  const palmNormalMagnitude = Math.sqrt(
    palmNormal.x ** 2 +
    palmNormal.y ** 2 +
    palmNormal.z ** 2
  ) || 0.001;

  const normalizedPalmNormal = {
    x: Number(
      (palmNormal.x / palmNormalMagnitude).toFixed(6)
    ),
    y: Number(
      (palmNormal.y / palmNormalMagnitude).toFixed(6)
    ),
    z: Number(
      (palmNormal.z / palmNormalMagnitude).toFixed(6)
    ),
  };

  // ---------------------------------------------------------
  // Return complete representation
  // ---------------------------------------------------------

  return {
    // Raw MediaPipe 3D landmarks
    landmarks: landmarks.map((point, index) => {
      if (!isValidLandmark(point)) {
        return {
          id: index,
          x: null,
          y: null,
          z: null,
        };
      }

      return {
        id: index,
        x: Number(point.x.toFixed(6)),
        y: Number(point.y.toFixed(6)),
        z: Number(point.z.toFixed(6)),
      };
    }),

    // Translation + scale normalized 3D landmarks
    normalizedLandmarks,

    // 3D geometric features
    jointAngles,
    fingertipDistances,
    fingerLengths,

    // Hand position/orientation
    wristPosition,
    wristOrientation,
    palmNormal: normalizedPalmNormal,

    reliability,

    fingers: fingerValidity,
  };
}


//---------------------------------------------------------------------------
// Debugging utility: log hand summary to console
//---------------------------------------------------------------------------

function logHandSummary(hand, side) {
  if (!hand) return;

  console.log(`%c[${side} HAND]`, 'font-weight:bold; color:green;');

  console.log('Landmarks:', hand.landmarks.length);

  console.log('Joint Angles:', {
    thumb: hand.jointAngles.thumb,
    index: hand.jointAngles.index,
    middle: hand.jointAngles.middle,
    ring: hand.jointAngles.ring,
    pinky: hand.jointAngles.pinky,
  });

  console.log('Wrist Position:', hand.wristPosition);

  console.log('Wrist Orientation:', {
    yaw: hand.wristOrientation.yaw,
    pitch: hand.wristOrientation.pitch,
    roll: hand.wristOrientation.roll,
  });

  console.log('Palm Normal:', hand.palmNormal);

  console.log('Fingertip Distances:', hand.fingertipDistances);

  console.log('Reliability:', hand.reliability);
}


// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------
export function usePoseTracker() {
  const videoRef     = useRef(null);
  const streamRef    = useRef(null);
  const intervalRef  = useRef(null);
  const sessionIdRef = useRef(null);
  const countRef     = useRef(0);
  const trackingModeRef = useRef(TRACKING_MODE.FACE);
  const maxFramesRef = useRef(Infinity); // caps sampling to the teacher clip's duration

  const [isTracking,  setIsTracking]  = useState(false);
  const [frameCount,  setFrameCount]  = useState(0);
  const [cameraError, setCameraError] = useState(null);
  const [streamReady, setStreamReady] = useState(false);
  const [mpStatus,    setMpStatus]    = useState('idle');

  // Pre-warm on mount
  useEffect(() => {
    setMpStatus('loading');
    getLandmarker()
      .then(() => setMpStatus('ready'))
      .catch(e => { console.warn('[MP] Unavailable:', e.message); setMpStatus('error'); });
  }, []);

  // Attach stream to <video>
  useEffect(() => {
    if (!streamReady) return;
    const vid    = videoRef.current;
    const stream = streamRef.current;
    if (!vid || !stream) return;
    vid.srcObject = stream;
    const onMeta = () => { vid.play().catch(() => {}); _startInterval(); };
    vid.addEventListener('loadedmetadata', onMeta, { once: true });
    if (vid.readyState >= 1) { vid.play().catch(() => {}); _startInterval(); }
    return () => vid.removeEventListener('loadedmetadata', onMeta);
  }, [streamReady]); // eslint-disable-line

  function _startInterval() {
    if (intervalRef.current) return;
    countRef.current = 0;

    intervalRef.current = setInterval(async () => {
      const vid = videoRef.current;
      const sid = sessionIdRef.current;
      if (!vid || !sid || vid.readyState < 2) return;

      if (countRef.current >= maxFramesRef.current) {
        // Reached the teacher clip's duration — stop sampling further
        // frames even if the caller hasn't called stopTracking() yet.
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        return;
      }

      let rawPose = null;
      let hands = null;

      if (trackingModeRef.current === TRACKING_MODE.FACE) {
        rawPose = await detectPose(vid);
      } else if (trackingModeRef.current === TRACKING_MODE.HAND) {
        hands = await detectHands(vid);
      }
      /** 
       if (hands) {
        console.log('[HANDS]', {
          left: !!hands.left,
          right: !!hands.right,
        });

        if (hands.left) {
          console.log('[LEFT HAND FEATURES]', hands.left);
        }

        if (hands.right) {
          console.log('[RIGHT HAND FEATURES]', hands.right);
        }
      }
      */

      if (hands) {
        if (hands.left) {
          logHandSummary(hands.left, 'LEFT');
        }

        if (hands.right) {
          logHandSummary(hands.right, 'RIGHT');
        }
      }     

      //const visible = !!rawPose;
      //const pose    = rawPose || { yaw: 0, pitch: 0, roll: 0 };
      const isFaceMode =
        trackingModeRef.current === TRACKING_MODE.FACE;

      const visible = isFaceMode
        ? !!rawPose
        : !!(hands?.left || hands?.right);

      const pose = rawPose || {
        yaw: 0,
        pitch: 0,
        roll: 0,
      };


      try {
        await fetch(`${API}/submit_pose`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sid,
            time: parseFloat(
              (
                countRef.current *
                SAMPLE_INTERVAL_MS /
                1000
              ).toFixed(3)
            ),

            tracking_mode: trackingModeRef.current,

            yaw: isFaceMode ? pose.yaw : null,
            pitch: isFaceMode ? pose.pitch : null,
            roll: isFaceMode ? pose.roll : null,

            visible,

            hands:
              trackingModeRef.current === TRACKING_MODE.HAND
                ? hands
                : null,
          }),

//          body: JSON.stringify({
//            session_id: sid,
//            time: parseFloat((countRef.current * SAMPLE_INTERVAL_MS / 1000).toFixed(3)),
//            yaw:   pose.yaw,
//            pitch: pose.pitch,
//            roll:  pose.roll,
//            visible,
//            hands,
//          }),
        });
        countRef.current += 1;
        setFrameCount(countRef.current);
      } catch { /* network blip */ }
    }, SAMPLE_INTERVAL_MS);
  }

  /**
   * startTracking(videoType, durationSec)
   * @param {string} videoType   'up-down' | 'right-left'
   * @param {number} [durationSec]  teacher clip duration, used to cap the
   *   number of frames sampled so capture can never run past the reference
   *   clip (optional — if omitted, no cap is applied and the caller is
   *   expected to stop tracking when the teacher video ends, as before).
   */
  const startTracking = useCallback(async (videoType = 'up-down', durationSec) => {
    const trackingMode =
      VIDEO_TRACKING_MODE[videoType] || TRACKING_MODE.FACE;

    trackingModeRef.current = trackingMode;

    setCameraError(null);
    setFrameCount(0);
    setStreamReady(false);
    countRef.current = 0;
    maxFramesRef.current = (typeof durationSec === 'number' && durationSec > 0)
      // +2 frames of buffer for interval/scheduling jitter near the end of the clip
      ? Math.ceil((durationSec * 1000) / SAMPLE_INTERVAL_MS) + 2
      : Infinity;

    let sessionId;
    try {
      const res  = await fetch(`${API}/start_session`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_type: videoType }),
      });
      let data = {};
      try { data = await res.json(); } catch { /* non-JSON error body */ }
      if (!res.ok) {
        throw new Error(data.detail || `Backend error (${res.status}) starting session.`);
      }
      sessionId  = data.session_id;
      sessionIdRef.current = sessionId;
    } catch (err) {
      setCameraError(err.message || 'Cannot reach backend. Is the FastAPI server running?');
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? 'Camera permission denied. Allow access and retry.'
        : `Camera error: ${err.message}`;
      setCameraError(msg);
      await fetch(`${API}/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
      return false;
    }

    setIsTracking(true);
    setStreamReady(true);
    return true;
  }, []);

  const stopTracking = useCallback(async () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (streamRef.current)   { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current)    { videoRef.current.srcObject = null; }
    maxFramesRef.current = Infinity;
    setIsTracking(false);
    setStreamReady(false);

    const sid = sessionIdRef.current;
    if (!sid) return null;
    sessionIdRef.current = null;
    try {
      const res = await fetch(`${API}/finish_session/${sid}`, { method: 'POST' });
      return await res.json();
    } catch { return null; }
  }, []);

  return { videoRef, isTracking, frameCount, cameraError, mpStatus, startTracking, stopTracking };
}