"""
extract_reference.py — REAL MEDIAPIPE FACEMESH (legacy solutions API, bundled model)

Uses mediapipe.solutions.face_mesh, which ships its own model binaries inside
the pip wheel (no internet download required at runtime).

requirements.txt pins: mediapipe==0.10.13  (last version with bundled solutions API)

The pose formula here is IDENTICAL to the one used in the frontend's
usePoseTracker.js (computePoseFromLandmarks), guaranteeing both sides of the
comparison operate in the same coordinate system:

    yaw   = (nose.x - eye_mid_x) / eye_span
    pitch = (nose.y - eye_mid_y) / face_height
    roll  = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)

Landmark indices (standard 468-point MediaPipe FaceMesh):
    nose_tip = 1, left_eye_outer = 33, right_eye_outer = 263,
    chin = 152, forehead = 10

Usage:
    python backend\services\extract_reference.py --video backend\reference\pataka.mp4 --output backend\reference\pataka.json --mode hands --rate 6
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
import mediapipe as mp
from hand_features import extract_hand_features


def compute_pose(lm):
    """Same math as frontend computePoseFromLandmarks()."""
    nose, leftE, rightE, chin, forehead = lm[1], lm[33], lm[263], lm[152], lm[10]

    eye_mid_x = (leftE.x + rightE.x) / 2
    eye_span  = abs(rightE.x - leftE.x) or 0.001
    yaw = (nose.x - eye_mid_x) / eye_span

    eye_mid_y = (leftE.y + rightE.y) / 2
    face_h    = abs(chin.y - forehead.y) or 0.001
    pitch = (nose.y - eye_mid_y) / face_h

    dx, dy = rightE.x - leftE.x, rightE.y - leftE.y
    roll = float(np.arctan2(dy, dx))

    return {
        'yaw':   round(float(yaw), 6),
        'pitch': round(float(pitch), 6),
        'roll':  round(float(roll), 6),
    }


def extract_reference(
    video_path,
    output_path,
    sample_rate=6,
    mode="face"
):
    """
    Extract reference features ONCE from a reference video.

    mode="face":
        Uses MediaPipe FaceMesh only.

    mode="hands":
        Uses MediaPipe Hands only.

        The hand landmarks are passed to hand_features.py.
        No hand mathematics is calculated here.

    The calculated features are saved to JSON and reused later.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Reference video not found: {video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    step = max(
        1,
        int(round(fps / sample_rate))
    )

    sequence = []

    # ---------------------------------------------------------
    # FACE MODE
    # ---------------------------------------------------------

    if mode == "face":

        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4
        ) as face_mesh:

            frame_index = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                if frame_index % step != 0:
                    frame_index += 1
                    continue

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    frame_index += 1
                    continue

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                pose = compute_pose(
                    face_landmarks.landmark
                )

                sequence.append({
                    "frame": frame_index,
                    "time": frame_index / fps,
                    **pose
                })

                frame_index += 1

        extraction_method = (
            "mediapipe_facemesh_solutions_api"
        )

        feature_type = "face"

    # ---------------------------------------------------------
    # HAND MODE
    # ---------------------------------------------------------

    elif mode == "hands":

        mp_hands = mp.solutions.hands

        detected_hand_frames = 0

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.4
        ) as hands:

            frame_index = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                if frame_index % step != 0:
                    frame_index += 1
                    continue

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb)

                hand_data = {
                    "left": None,
                    "right": None
                }

                if results.multi_hand_landmarks:

                    detected_hand_frames += 1

                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness
                    ):

                        # -----------------------------------------
                        # Convert MediaPipe landmarks to x/y/z
                        # -----------------------------------------

                        landmarks = [
                            {
                                "x": float(point.x),
                                "y": float(point.y),
                                "z": float(point.z)
                            }
                            for point
                            in hand_landmarks.landmark
                        ]

                        # -----------------------------------------
                        # MediaPipe handedness
                        # -----------------------------------------

                        side = (
                            handedness.classification[0].label
                            .lower()
                        )

                        if side not in ("left", "right"):
                            continue

                        # -----------------------------------------
                        # ALL hand mathematics happens here
                        # through hand_features.py
                        # -----------------------------------------

                        features = extract_hand_features(
                            landmarks
                        )

                        hand_data[side] = features

                # ---------------------------------------------
                # IMPORTANT:
                # Keep the frame even when no hand is detected.
                #
                # This preserves the temporal sequence.
                # Missing hand = None.
                # ---------------------------------------------

                sequence.append({
                    "frame": frame_index,
                    "time": frame_index / fps,
                    "hands": hand_data
                })

                frame_index += 1

        extraction_method = (
            "mediapipe_hands_solutions_api"
        )

        feature_type = "hands"

    else:

        cap.release()

        raise ValueError(
            "mode must be either 'face' or 'hands'"
        )

    cap.release()

    if not sequence:
        raise RuntimeError(
            f"No usable {mode} features were extracted "
            f"from {video_path}"
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "sample_rate": sample_rate,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
        "feature_type": feature_type,
        "extraction_method": extraction_method,
        "coordinate_system": "mediapipe_normalized_3d",
        "sequence_length": len(sequence),
        "sequence": sequence
    }

    if mode == "hands":

        metadata["detected_hand_frames"] = (
            detected_hand_frames
        )

        metadata["hand_detection_rate"] = (
            detected_hand_frames / len(sequence)
            if sequence
            else 0
        )

    else:

        yaw_values = [
            frame["yaw"]
            for frame in sequence
            if frame.get("yaw") is not None
        ]

        pitch_values = [
            frame["pitch"]
            for frame in sequence
            if frame.get("pitch") is not None
        ]

        roll_values = [
            frame["roll"]
            for frame in sequence
            if frame.get("roll") is not None
        ]

        metadata["signal_stats"] = {
            "yaw_min": min(yaw_values) if yaw_values else None,
            "yaw_max": max(yaw_values) if yaw_values else None,
            "pitch_min": min(pitch_values) if pitch_values else None,
            "pitch_max": max(pitch_values) if pitch_values else None,
            "roll_min": min(roll_values) if roll_values else None,
            "roll_max": max(roll_values) if roll_values else None,
        }

    # ---------------------------------------------------------
    # SAVE ONCE
    # ---------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        f"Saved {feature_type} reference features:"
        f" {output_path}"
    )

    print(
        f"Frames extracted: {len(sequence)}"
    )

    return metadata 


def _finalize(sequence, fps, total, duration, sample_rate, output_path):
    yaws    = [s['yaw']   for s in sequence]
    pitches = [s['pitch'] for s in sequence]
    rolls   = [s['roll']  for s in sequence]

    stats = {
        'yaw_mean':   round(float(np.mean(yaws)), 4),    'yaw_std':   round(float(np.std(yaws)), 4),
        'yaw_min':    round(float(np.min(yaws)), 4),      'yaw_max':   round(float(np.max(yaws)), 4),
        'pitch_mean': round(float(np.mean(pitches)), 4),  'pitch_std': round(float(np.std(pitches)), 4),
        'pitch_min':  round(float(np.min(pitches)), 4),   'pitch_max': round(float(np.max(pitches)), 4),
        'roll_mean':  round(float(np.mean(rolls)), 4),    'roll_std':  round(float(np.std(rolls)), 4),
        'roll_min':   round(float(np.min(rolls)), 4),     'roll_max':  round(float(np.max(rolls)), 4),
    }

    print(f"[extract_reference] Extracted {len(sequence)} frames")
    print(f"  Yaw  : {stats['yaw_min']:.4f} → {stats['yaw_max']:.4f}  std={stats['yaw_std']:.4f}")
    print(f"  Pitch: {stats['pitch_min']:.4f} → {stats['pitch_max']:.4f}  std={stats['pitch_std']:.4f}")
    print(f"  Roll : {stats['roll_min']:.4f} → {stats['roll_max']:.4f}  std={stats['roll_std']:.4f}")

    reference = {
        'fps': fps,
        'sample_rate': sample_rate,
        'total_frames': total,
        'duration': round(duration, 2),
        'movements': 3,
        'extraction_method': 'mediapipe_facemesh_solutions_api',
        'coordinate_system': 'mediapipe_normalized',
        'signal_stats': stats,
        'sequence': sequence,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)
    print(f"[extract_reference] Saved → {output_path}")
    return reference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="reference/up-down.mp4"
    )

    parser.add_argument(
        "--output",
        default="reference/reference.json"
    )

    parser.add_argument(
        "--rate",
        type=int,
        default=6
    )
    parser.add_argument(
        "--mode",
        choices=["face", "hands"],
        default="face",
        help="Feature type to extract from the reference video"
    )
    args = parser.parse_args()
    extract_reference(
        video_path=args.video,
        output_path=args.output,
        sample_rate=args.rate,
        mode=args.mode
    )
