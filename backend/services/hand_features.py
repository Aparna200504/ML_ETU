#hand_features.py is a calculator not a video processor. 
# It takes a list of 21 hand landmarks and computes canonical features from them, 
# including joint angles, fingertip distances, finger lengths, wrist orientation, and palm normal. 
# It also normalizes the landmarks relative to the wrist and middle finger MCP joint.

import math
from typing import Optional, Dict, List, Any

FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def is_valid_landmark(point):
    return (
        isinstance(point, dict)
        and all(
            key in point
            and isinstance(point[key], (int, float))
            and math.isfinite(point[key])
            for key in ("x", "y", "z")
        )
    )


def distance_3d(a, b):
    if not is_valid_landmark(a) or not is_valid_landmark(b):
        return None

    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]

    return math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


def angle_3d(a, b, c):
    """
    Angle ABC, with B as the joint.

    BA = A - B
    BC = C - B
    """

    if not all(
        is_valid_landmark(p)
        for p in (a, b, c)
    ):
        return None

    ba = {
        "x": a["x"] - b["x"],
        "y": a["y"] - b["y"],
        "z": a["z"] - b["z"],
    }

    bc = {
        "x": c["x"] - b["x"],
        "y": c["y"] - b["y"],
        "z": c["z"] - b["z"],
    }

    dot = (
        ba["x"] * bc["x"]
        + ba["y"] * bc["y"]
        + ba["z"] * bc["z"]
    )

    mag_ba = math.sqrt(
        ba["x"] ** 2 +
        ba["y"] ** 2 +
        ba["z"] ** 2
    )

    mag_bc = math.sqrt(
        bc["x"] ** 2 +
        bc["y"] ** 2 +
        bc["z"] ** 2
    )

    if mag_ba == 0 or mag_bc == 0:
        return None

    cosine = max(
        -1.0,
        min(1.0, dot / (mag_ba * mag_bc))
    )

    return math.degrees(
        math.acos(cosine)
    )


def calculate_finger_validity(landmarks):
    result = {}

    for finger, indices in FINGER_LANDMARKS.items():
        valid_count = sum(
            1
            for i in indices
            if is_valid_landmark(landmarks[i])
        )

        result[finger] = {
            "valid": valid_count == len(indices),
            "valid_landmarks": valid_count,
            "total_landmarks": len(indices),
        }

    return result


def normalize_landmarks(landmarks):
    if len(landmarks) != 21:
        return [None] * len(landmarks)

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    palm_scale = distance_3d(wrist, middle_mcp)

    if palm_scale is None or palm_scale == 0:
        return [None] * len(landmarks)

    normalized = []

    for point in landmarks:
        if not is_valid_landmark(point):
            normalized.append(None)
            continue

        normalized.append({
            "x": round((point["x"] - wrist["x"]) / palm_scale, 6),
            "y": round((point["y"] - wrist["y"]) / palm_scale, 6),
            "z": round((point["z"] - wrist["z"]) / palm_scale, 6),
        })

    return normalized


def vector_magnitude(v):
    return math.sqrt(
        v["x"] ** 2 +
        v["y"] ** 2 +
        v["z"] ** 2
    )


def normalize_vector(v):
    magnitude = vector_magnitude(v)

    if magnitude == 0:
        return None

    return {
        "x": v["x"] / magnitude,
        "y": v["y"] / magnitude,
        "z": v["z"] / magnitude,
    }


def calculate_wrist_orientation(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    if not all(
        is_valid_landmark(p)
        for p in (
            wrist,
            index_mcp,
            pinky_mcp,
            middle_mcp,
        )
    ):
        return None

    forward = normalize_vector({
        "x": middle_mcp["x"] - wrist["x"],
        "y": middle_mcp["y"] - wrist["y"],
        "z": middle_mcp["z"] - wrist["z"],
    })

    side = normalize_vector({
        "x": pinky_mcp["x"] - index_mcp["x"],
        "y": pinky_mcp["y"] - index_mcp["y"],
        "z": pinky_mcp["z"] - index_mcp["z"],
    })

    if forward is None or side is None:
        return None

    palm_normal = normalize_vector({
        "x": (
            forward["y"] * side["z"]
            - forward["z"] * side["y"]
        ),
        "y": (
            forward["z"] * side["x"]
            - forward["x"] * side["z"]
        ),
        "z": (
            forward["x"] * side["y"]
            - forward["y"] * side["x"]
        ),
    })

    yaw = math.degrees(
        math.atan2(
            forward["x"],
            forward["z"]
        )
    )

    pitch = math.degrees(
        math.atan2(
            -forward["y"],
            math.sqrt(
                forward["x"] ** 2 +
                forward["z"] ** 2
            )
        )
    )

    roll = math.degrees(
        math.atan2(
            side["y"],
            side["x"]
        )
    )

    return {
        "yaw": round(yaw, 3),
        "pitch": round(pitch, 3),
        "roll": round(roll, 3),

        "direction": {
            "x": round(forward["x"], 6),
            "y": round(forward["y"], 6),
            "z": round(forward["z"], 6),
        },

        "palm_normal": {
            "x": round(palm_normal["x"], 6),
            "y": round(palm_normal["y"], 6),
            "z": round(palm_normal["z"], 6),
        } if palm_normal else None,
    }


def calculate_palm_normal(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    if not all(
        is_valid_landmark(p)
        for p in (
            wrist,
            index_mcp,
            pinky_mcp,
        )
    ):
        return None

    v1 = {
        "x": index_mcp["x"] - wrist["x"],
        "y": index_mcp["y"] - wrist["y"],
        "z": index_mcp["z"] - wrist["z"],
    }

    v2 = {
        "x": pinky_mcp["x"] - wrist["x"],
        "y": pinky_mcp["y"] - wrist["y"],
        "z": pinky_mcp["z"] - wrist["z"],
    }

    normal = normalize_vector({
        "x": (
            v1["y"] * v2["z"]
            - v1["z"] * v2["y"]
        ),
        "y": (
            v1["z"] * v2["x"]
            - v1["x"] * v2["z"]
        ),
        "z": (
            v1["x"] * v2["y"]
            - v1["y"] * v2["x"]
        ),
    })

    if normal is None:
        return None

    return {
        "x": round(normal["x"], 6),
        "y": round(normal["y"], 6),
        "z": round(normal["z"], 6),
    }


def extract_hand_features(landmarks):
    """
    Canonical hand-feature calculation.

    Input:
        21 MediaPipe hand landmarks containing x, y, z.

    Output:
        Dictionary containing all calculated 3D hand features.

    This function is called by extract_reference.py when creating
    reference JSON files and can also be called for live student frames.
    """

    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return None

    finger_validity = calculate_finger_validity(landmarks)

    normalized_landmarks = normalize_landmarks(landmarks)

    # ---------------------------------------------------------
    # Joint angles
    # ---------------------------------------------------------

    finger_joint_map = {
        "thumb": [(1, 2, 3), (2, 3, 4)],
        "index": [(5, 6, 7), (6, 7, 8)],
        "middle": [(9, 10, 11), (10, 11, 12)],
        "ring": [(13, 14, 15), (14, 15, 16)],
        "pinky": [(17, 18, 19), (18, 19, 20)],
    }

    joint_angles = {}

    for finger, joints in finger_joint_map.items():

        if not finger_validity[finger]["valid"]:
            joint_angles[finger] = None
            continue

        joint_angles[finger] = [
            angle_3d(
                landmarks[a],
                landmarks[b],
                landmarks[c]
            )
            for a, b, c in joints
        ]

    # ---------------------------------------------------------
    # Finger lengths
    # ---------------------------------------------------------

    finger_lengths = {}

    for finger, indices in FINGER_LANDMARKS.items():

        if not finger_validity[finger]["valid"]:
            finger_lengths[finger] = None
            continue

        length = 0.0

        for i in range(len(indices) - 1):
            d = distance_3d(
                landmarks[indices[i]],
                landmarks[indices[i + 1]]
            )

            if d is None:
                length = None
                break

            length += d

        finger_lengths[finger] = length

    # ---------------------------------------------------------
    # Fingertip distances
    # ---------------------------------------------------------

    fingertips = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    fingertip_distances = {}

    for finger_a, index_a in fingertips.items():

        if not finger_validity[finger_a]["valid"]:
            fingertip_distances[finger_a] = None
            continue

        fingertip_distances[finger_a] = {}

        for finger_b, index_b in fingertips.items():

            if finger_a == finger_b:
                continue

            if not finger_validity[finger_b]["valid"]:
                fingertip_distances[finger_a][finger_b] = None
                continue

            fingertip_distances[finger_a][finger_b] = distance_3d(
                landmarks[index_a],
                landmarks[index_b]
            )

    # ---------------------------------------------------------
    # Wrist position
    # ---------------------------------------------------------

    wrist_position = landmarks[0] if is_valid_landmark(landmarks[0]) else None

    # ---------------------------------------------------------
    # Wrist orientation
    # ---------------------------------------------------------

    wrist_orientation = calculate_wrist_orientation(landmarks)

    # ---------------------------------------------------------
    # Palm orientation
    # ---------------------------------------------------------

    palm_normal = calculate_palm_normal(landmarks)

    # ---------------------------------------------------------
    # Final feature dictionary
    # ---------------------------------------------------------

    return {
        "raw_landmarks": landmarks,
        "normalized_landmarks": normalized_landmarks,
        "joint_angles": joint_angles,
        "finger_lengths": finger_lengths,
        "fingertip_distances": fingertip_distances,
        "wrist_position": wrist_position,
        "wrist_orientation": wrist_orientation,
        "palm_normal": palm_normal,
        "finger_validity": finger_validity,
    }


# import math
# from typing import List, Dict


# def distance_3d(a, b):
#     dx = a["x"] - b["x"]
#     dy = a["y"] - b["y"]
#     dz = a["z"] - b["z"]

#     return math.sqrt(dx * dx + dy * dy + dz * dz)


# def angle_3d(a, b, c):
#     """
#     Angle ABC, with B as the joint.
#     """

#     ba = {
#         "x": a["x"] - b["x"],
#         "y": a["y"] - b["y"],
#         "z": a["z"] - b["z"],
#     }

#     bc = {
#         "x": c["x"] - b["x"],
#         "y": c["y"] - b["y"],
#         "z": c["z"] - b["z"],
#     }

#     dot = (
#         ba["x"] * bc["x"]
#         + ba["y"] * bc["y"]
#         + ba["z"] * bc["z"]
#     )

#     mag_ba = math.sqrt(
#         ba["x"] ** 2 +
#         ba["y"] ** 2 +
#         ba["z"] ** 2
#     )

#     mag_bc = math.sqrt(
#         bc["x"] ** 2 +
#         bc["y"] ** 2 +
#         bc["z"] ** 2
#     )

#     if mag_ba == 0 or mag_bc == 0:
#         return 0.0

#     cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))

#     return math.degrees(math.acos(cosine))


# def normalize_landmarks(landmarks):
#     wrist = landmarks[0]

#     palm_scale = distance_3d(
#         landmarks[0],
#         landmarks[9]
#     ) or 0.001

#     return [
#         {
#             "x": round((p["x"] - wrist["x"]) / palm_scale, 6),
#             "y": round((p["y"] - wrist["y"]) / palm_scale, 6),
#             "z": round((p["z"] - wrist["z"]) / palm_scale, 6),
#         }
#         for p in landmarks
#     ]


# def extract_hand_features(landmarks):
#     """
#     Extract the same 3D features used by the frontend.
#     """

#     if not landmarks or len(landmarks) != 21:
#         return None

#     normalized = normalize_landmarks(landmarks)

#     joint_angles = {
#         "thumb": [
#             angle_3d(landmarks[1], landmarks[2], landmarks[3]),
#             angle_3d(landmarks[2], landmarks[3], landmarks[4]),
#         ],
#         "index": [
#             angle_3d(landmarks[5], landmarks[6], landmarks[7]),
#             angle_3d(landmarks[6], landmarks[7], landmarks[8]),
#         ],
#         "middle": [
#             angle_3d(landmarks[9], landmarks[10], landmarks[11]),
#             angle_3d(landmarks[10], landmarks[11], landmarks[12]),
#         ],
#         "ring": [
#             angle_3d(landmarks[13], landmarks[14], landmarks[15]),
#             angle_3d(landmarks[14], landmarks[15], landmarks[16]),
#         ],
#         "pinky": [
#             angle_3d(landmarks[17], landmarks[18], landmarks[19]),
#             angle_3d(landmarks[18], landmarks[19], landmarks[20]),
#         ],
#     }

#     fingertip_distances = {
#         "thumb_index": distance_3d(landmarks[4], landmarks[8]),
#         "thumb_middle": distance_3d(landmarks[4], landmarks[12]),
#         "thumb_ring": distance_3d(landmarks[4], landmarks[16]),
#         "thumb_pinky": distance_3d(landmarks[4], landmarks[20]),
#         "index_middle": distance_3d(landmarks[8], landmarks[12]),
#         "middle_ring": distance_3d(landmarks[12], landmarks[16]),
#         "ring_pinky": distance_3d(landmarks[16], landmarks[20]),
#     }

#     finger_lengths = {
#         "thumb": distance_3d(landmarks[1], landmarks[4]),
#         "index": distance_3d(landmarks[5], landmarks[8]),
#         "middle": distance_3d(landmarks[9], landmarks[12]),
#         "ring": distance_3d(landmarks[13], landmarks[16]),
#         "pinky": distance_3d(landmarks[17], landmarks[20]),
#     }

#     return {
#         "landmarks": landmarks,
#         "normalized_landmarks": normalized,
#         "joint_angles": joint_angles,
#         "fingertip_distances": fingertip_distances,
#         "finger_lengths": finger_lengths,
#     }
