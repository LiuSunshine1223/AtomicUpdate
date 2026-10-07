"""Recompute saved analyses in memory and compare scientific outputs."""
import argparse
from contextlib import ExitStack
import csv
import io
import json
from pathlib import Path
import socket
from unittest.mock import patch

PRODUCER_FIELDS = {"analysis_script_sha256", "created_at_utc"}


def scientific_json(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRODUCER_FIELDS:
                continue
            if key == "outputs":
                result[key] = {name: Path(path).name for name, path in item.items()}
            else:
                result[key] = scientific_json(item)
        return result
    if isinstance(value, list):
        return [scientific_json(item) for item in value]
    return value


def compare_json(generated, saved, label):
    if scientific_json(generated) != scientific_json(saved):
        raise ValueError("Recomputed scientific output differs: " + str(label))


def check_analysis(analyze, namespace):
    """Run an existing analyzer with every writer redirected to memory."""
    captured = {}
    original_open = Path.open

    def capture_json(path, value):
        captured[Path(path)] = ("json", value)

    def capture_jsonl(path, rows):
        captured[Path(path)] = ("jsonl", list(rows))

    def capture_csv(path, rows, fieldnames):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        captured[Path(path)] = ("csv", stream.getvalue())

    def capture_text(path, text):
        captured[Path(path)] = ("text", text)

    class ReportBuffer(io.StringIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            if not self.closed:
                captured[self.path] = ("text", self.getvalue())
            super().close()

    def memory_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            if "b" in mode or Path(path).name != "report.md":
                raise RuntimeError("Unexpected writer during analysis check: " + str(path))
            return ReportBuffer(Path(path))
        return original_open(path, mode, *args, **kwargs)

    with ExitStack() as stack:
        for name in ("write_json", "write_json_atomic"):
            if name in namespace:
                stack.enter_context(patch.dict(namespace, {name: capture_json}))
        for name in ("write_jsonl", "write_jsonl_atomic"):
            if name in namespace:
                stack.enter_context(patch.dict(namespace, {name: capture_jsonl}))
        for name in ("write_csv", "write_csv_atomic"):
            if name in namespace:
                stack.enter_context(patch.dict(namespace, {name: capture_csv}))
        if "write_text_atomic" in namespace:
            stack.enter_context(patch.dict(namespace, {"write_text_atomic": capture_text}))
        stack.enter_context(patch.object(Path, "open", memory_open))
        stack.enter_context(patch.object(Path, "mkdir", lambda *args, **kwargs: None))
        stack.enter_context(patch.object(socket.socket, "connect",
                                        side_effect=RuntimeError("Network use is forbidden during offline analysis")))
        analyze()

    numerical_files = 0
    reports = 0
    for path, (kind, value) in captured.items():
        if not path.is_file():
            raise ValueError("Saved analysis file is missing: " + str(path))
        if kind == "json":
            compare_json(value, json.loads(path.read_text(encoding="utf-8-sig")), path)
            numerical_files += 1
        elif kind == "jsonl":
            saved = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if value != saved:
                raise ValueError("Recomputed JSONL differs: " + str(path))
            numerical_files += 1
        elif kind == "csv":
            actual = list(csv.reader(io.StringIO(value)))
            saved = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
            if actual != saved:
                raise ValueError("Recomputed CSV differs: " + str(path))
            numerical_files += 1
        else:
            # Paths printed in a narrative report may identify the old release.
            # Every numerical JSON/CSV feeding that report is compared above.
            if path.name != "report.md" or not value.strip():
                raise ValueError("Unexpected report output: " + str(path))
            reports += 1
    if not numerical_files:
        raise ValueError("No numerical analysis outputs were checked")
    print(json.dumps({"mode": "check", "model_calls": 0, "files_written": 0,
                      "verified_json_csv": numerical_files,
                      "reports_recomputed_in_memory": reports}, indent=2))


def run_analysis(analyze, namespace, stage_dir, owner_code):
    parser = argparse.ArgumentParser(description="Offline analysis; --check verifies without writing.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_analysis(analyze, namespace)
    else:
        from atomic_memory.paths import require_output_stage
        require_output_stage(stage_dir, owner_code)
        analyze()
