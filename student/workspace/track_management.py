"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    pos_sens = np.asarray(meas.z, dtype=float).ravel()[:3]
    transform = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    pos_veh = transform[:3, :3] @ pos_sens + transform[:3, 3]

    x = np.asmatrix(np.r_[pos_veh, [0.0, 0.0, 0.0]]).T

    rotation = transform[:3, :3]
    R_meas = np.asarray(meas.R, dtype=float)[:3, :3]
    R_pos = rotation @ R_meas @ rotation.T

    P = np.zeros((6, 6), dtype=float)
    P[:3, :3] = R_pos
    P[3, 3] = params.sigma_p44
    P[4, 4] = params.sigma_p55
    P[5, 5] = params.sigma_p66
    P = np.asmatrix(P)

    return {
        "x": x,
        "P": P,
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    delta = 1.0 / params.window

    if associated:
        new_score = min(1.0, track["score"] + delta)
    else:
        new_score = track["score"] - delta

    track["score"] = new_score

    if track["state"] == "confirmed":
        track["state"] = "confirmed"
    else:
        if new_score > params.confirmed_threshold:
            track["state"] = "confirmed"
        elif associated:
            track["state"] = "tentative"

    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P = np.asarray(track["P"])
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True

    if track["state"] == "confirmed":
        if track["score"] < params.delete_threshold:
            return True
    else:
        if track["score"] <= 0.0:
            return True

    return False

