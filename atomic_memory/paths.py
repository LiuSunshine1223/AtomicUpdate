"""Project locations and output/backend configuration for all experiment stages."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess


def resolve_ollama_url():
    """Use the configured service, or the Windows host when running in WSL NAT."""
    configured_url = os.environ.get("ATOMICUPDATE_OLLAMA_URL")
    if configured_url:
        return configured_url
    local_url = "http://127.0.0.1:11434"
    if platform.system() != "Linux":
        return local_url
    try:
        kernel = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return local_url
    if "microsoft" not in kernel.lower():
        return local_url

    # Mirrored networking permits localhost; NAT requires the Windows gateway.
    try:
        mode = subprocess.run(
            ["wslinfo", "--networking-mode"], capture_output=True, text=True,
            timeout=2, check=False,
        )
        if mode.returncode == 0 and mode.stdout.strip().lower() == "mirrored":
            return local_url
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        routes = Path("/proc/net/route").read_text(encoding="utf-8")
    except OSError:
        return local_url
    gateways = []
    for row in routes.splitlines()[1:]:
        fields = row.split()
        if len(fields) < 8 or fields[1] != "00000000":
            continue
        try:
            gateway = int(fields[2], 16)
            flags = int(fields[3], 16)
            metric = int(fields[6])
            if gateway and flags & 3 == 3:
                address = ".".join(str(octet) for octet in gateway.to_bytes(4, "little"))
                gateways.append((metric, address))
        except (ValueError, OverflowError):
            continue
    if gateways:
        return "http://" + min(gateways)[1] + ":11434"
    return local_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
configured = Path(os.environ.get("ATOMICUPDATE_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()
OUTPUT_ROOT = (configured if configured.is_absolute() else PROJECT_ROOT / configured).resolve()
OLLAMA_URL = resolve_ollama_url()

DEV = OUTPUT_ROOT / "preparation" / "dev"
SELECTION = OUTPUT_ROOT / "preparation" / "selection"
CALIBRATION = OUTPUT_ROOT / "preparation" / "calibration"
BASELINE = OUTPUT_ROOT / "exp1_baseline"
WORDING = OUTPUT_ROOT / "exp2_wording"
UNIT_LOAD = OUTPUT_ROOT / "exp3_unit_load"
PROFILE = OUTPUT_ROOT / "exp4_profile"
FOLLOWUP = PROFILE / "followup"


def require_output_stage(stage_dir, owner_code):
    """Allow outputs/ while protecting records frozen by another source version.

    Empty stages can be constructed in place. An existing stage is writable
    only when its manifest records the exact current runner or builder bytes;
    the stage's own checks still enforce inputs, prompts, models and settings.
    """
    stage_dir = Path(stage_dir).resolve()
    owner_code = Path(owner_code).resolve()
    if not stage_dir.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Experiment stage must be within the configured outputs directory")
    manifests = sorted(stage_dir.glob("*freeze_manifest.json"))
    if (stage_dir / "integration_smoke_manifest.json").is_file():
        manifests.append(stage_dir / "integration_smoke_manifest.json")

    def owner_hashes(value):
        if not isinstance(value, dict):
            return set()
        hashes = set()
        for key, item in value.items():
            if key == "runner_sha256":
                hashes.add(str(item))
            elif key == "code_sha256" and isinstance(item, dict):
                hashes.update(str(digest) for digest in item.values())
            elif isinstance(item, dict):
                hashes.update(owner_hashes(item))
        return hashes

    current_hash = hashlib.sha256(owner_code.read_bytes()).hexdigest()
    for manifest in manifests:
        record = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if current_hash not in owner_hashes(record):
            raise RuntimeError(
                "Existing frozen records belong to a different source version: "
                + str(stage_dir)
                + ". Use analyze --check to reproduce saved results. Full model "
                "execution requires backing up the complete experiment directory "
                "and starting from an empty directory at the same outputs path."
            )
    if not manifests and (list(stage_dir.glob("*results.jsonl"))
                          or (stage_dir / "responses.jsonl").exists()):
        raise RuntimeError("Response records exist without their frozen manifest: " + str(stage_dir))
