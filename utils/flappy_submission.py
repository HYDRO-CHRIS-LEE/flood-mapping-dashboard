"""
flappy_submission.py -- v2.1 plan contract

Validates, saves, and loads DQN model submissions for the Flappy Bird
competition.  Only state_dict uploads are accepted (never full serialized
models).  torch.load always uses weights_only=True for safety.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_ARCHITECTURES = {"model1", "model2", "model3"}
EXPECTED_OBS_DIM = 4
EXPECTED_ACTION_DIM = 2

REQUIRED_METADATA_FIELDS = {
    "team_name",
    "architecture_id",
    "obs_dim",
    "action_dim",
    "framework",
}

# ---------------------------------------------------------------------------
# Architecture builder
# ---------------------------------------------------------------------------


def _build_model(architecture_id: str, dropout: float = 0.2) -> nn.Sequential:
    """Build a whitelisted architecture by ID.

    The architectures exactly mirror those defined in
    ``flappy_bird/flappybird_app_lab.py``.
    """
    obs_dim = EXPECTED_OBS_DIM
    action_dim = EXPECTED_ACTION_DIM

    if architecture_id == "model1":
        # 4 -> 64 -> 32 -> 16 -> 8 -> 2
        return nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(8, action_dim),
        )

    if architecture_id == "model2":
        # 4 -> 64 -> 32 -> 32 -> 8 -> 2
        return nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(8, action_dim),
        )

    if architecture_id == "model3":
        # 4 -> 32 -> 32 -> 16 -> 16 -> 2
        return nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, action_dim),
        )

    raise ValueError(f"Unknown architecture_id: {architecture_id!r}")


# ---------------------------------------------------------------------------
# Validation + loading
# ---------------------------------------------------------------------------


def validate_and_load(
    state_dict_bytes: bytes,
    metadata: dict,
) -> tuple[torch.nn.Module | None, dict | None, str | None]:
    """Validate a submission and return the loaded model.

    Returns
    -------
    (model, metadata, error_message)
        *error_message* is ``None`` on success.  On failure *model* and
        *metadata* are ``None`` and *error_message* describes the problem.
    """

    # 1. file readable -------------------------------------------------------
    if not isinstance(state_dict_bytes, bytes) or len(state_dict_bytes) == 0:
        return None, None, "Validation 1/9 failed: state_dict_bytes is empty or not bytes."

    # 2. metadata readable and has required fields ---------------------------
    if not isinstance(metadata, dict):
        return None, None, "Validation 2/9 failed: metadata is not a dict."

    missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
    if missing:
        return None, None, (
            f"Validation 2/9 failed: metadata missing required fields: {sorted(missing)}"
        )

    architecture_id = metadata["architecture_id"]
    team_name = metadata["team_name"]
    obs_dim = metadata["obs_dim"]
    action_dim = metadata["action_dim"]

    # 3. architecture_id in whitelist ----------------------------------------
    if architecture_id not in ALLOWED_ARCHITECTURES:
        return None, None, (
            f"Validation 3/9 failed: architecture_id {architecture_id!r} "
            f"not in {sorted(ALLOWED_ARCHITECTURES)}."
        )

    # 4. team_name exists ----------------------------------------------------
    if not team_name or not str(team_name).strip():
        return None, None, "Validation 4/9 failed: team_name is empty."

    # 5. obs_dim == 4 --------------------------------------------------------
    if int(obs_dim) != EXPECTED_OBS_DIM:
        return None, None, (
            f"Validation 5/9 failed: obs_dim={obs_dim}, expected {EXPECTED_OBS_DIM}."
        )

    # 6. action_dim == 2 -----------------------------------------------------
    if int(action_dim) != EXPECTED_ACTION_DIM:
        return None, None, (
            f"Validation 6/9 failed: action_dim={action_dim}, expected {EXPECTED_ACTION_DIM}."
        )

    # 7. model builds from whitelist -----------------------------------------
    dropout = metadata.get("dropout", 0.2)
    try:
        model = _build_model(architecture_id, dropout=float(dropout))
    except Exception as exc:
        return None, None, f"Validation 7/9 failed: could not build model -- {exc}"

    # 8. state_dict loads on CPU with weights_only=True ----------------------
    try:
        buf = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as exc:
        return None, None, f"Validation 8/9 failed: torch.load error -- {exc}"

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        return None, None, (
            f"Validation 8/9 failed: state_dict does not match "
            f"{architecture_id} architecture -- {exc}"
        )

    # 9. dummy forward pass succeeds on shape (1, 4) -------------------------
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, EXPECTED_OBS_DIM)
            output = model(dummy)
        if output.shape != (1, EXPECTED_ACTION_DIM):
            return None, None, (
                f"Validation 9/9 failed: output shape {tuple(output.shape)}, "
                f"expected (1, {EXPECTED_ACTION_DIM})."
            )
    except Exception as exc:
        return None, None, f"Validation 9/9 failed: forward pass error -- {exc}"

    return model, metadata, None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_submission(
    data_root: str,
    stage_id: int,
    team_name: str,
    state_dict_bytes: bytes,
    metadata: dict,
) -> str:
    """Save a validated submission to disk.

    Storage layout::

        {data_root}/flappy_submissions/stage_{N}/{team_name}/
            state_dict.pt
            metadata.json
            submission_meta.json

    Returns the submission directory path.
    """
    team_dir = os.path.join(
        data_root, "flappy_submissions", f"stage_{stage_id}", team_name
    )
    os.makedirs(team_dir, exist_ok=True)

    # state_dict.pt -- raw bytes coming from the uploader
    sd_path = os.path.join(team_dir, "state_dict.pt")
    with open(sd_path, "wb") as f:
        f.write(state_dict_bytes)

    # metadata.json -- the caller-provided metadata dict
    meta_path = os.path.join(team_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # submission_meta.json -- bookkeeping added by the platform
    sub_meta = {
        "team_name": team_name,
        "stage_id": stage_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state_dict_size_bytes": len(state_dict_bytes),
    }
    sub_meta_path = os.path.join(team_dir, "submission_meta.json")
    with open(sub_meta_path, "w", encoding="utf-8") as f:
        json.dump(sub_meta, f, ensure_ascii=False, indent=2)

    return team_dir


def load_submission(
    team_dir: str,
) -> tuple[torch.nn.Module | None, dict | None, str | None]:
    """Load a previously saved submission from *team_dir*.

    Returns
    -------
    (model, metadata, error_message)
        *error_message* is ``None`` on success.
    """
    meta_path = os.path.join(team_dir, "metadata.json")
    sd_path = os.path.join(team_dir, "state_dict.pt")

    if not os.path.isfile(meta_path):
        return None, None, f"metadata.json not found in {team_dir}"
    if not os.path.isfile(sd_path):
        return None, None, f"state_dict.pt not found in {team_dir}"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        return None, None, f"Failed to read metadata.json -- {exc}"

    try:
        with open(sd_path, "rb") as f:
            state_dict_bytes = f.read()
    except Exception as exc:
        return None, None, f"Failed to read state_dict.pt -- {exc}"

    return validate_and_load(state_dict_bytes, metadata)


def list_stage_submissions(data_root: str, stage_id: int) -> list[dict]:
    """List all submissions for a given stage.

    Returns a list of dicts, each with at least ``team_name`` and
    ``team_dir`` keys, plus any fields from ``submission_meta.json``.
    """
    stage_dir = os.path.join(
        data_root, "flappy_submissions", f"stage_{stage_id}"
    )

    if not os.path.isdir(stage_dir):
        return []

    submissions: list[dict] = []
    for entry in sorted(os.listdir(stage_dir)):
        team_dir = os.path.join(stage_dir, entry)
        if not os.path.isdir(team_dir):
            continue

        record: dict = {"team_name": entry, "team_dir": team_dir}

        sub_meta_path = os.path.join(team_dir, "submission_meta.json")
        if os.path.isfile(sub_meta_path):
            try:
                with open(sub_meta_path, "r", encoding="utf-8") as f:
                    record.update(json.load(f))
            except Exception:
                pass  # best-effort

        meta_path = os.path.join(team_dir, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    record["metadata"] = json.load(f)
            except Exception:
                pass

        submissions.append(record)

    return submissions
