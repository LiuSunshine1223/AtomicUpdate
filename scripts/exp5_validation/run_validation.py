"""Run the frozen experiment-5 reader inputs; resume only unfinished calls."""
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atomic_memory import paths
from atomic_memory import validation as v
from atomic_memory.backend import GenerationConfig, OllamaBackend


def run(out=v.OUT, backend=None):
    out = Path(out)
    paths.require_output_stage(out, Path(__file__))
    freeze, plan = v.verify(out)
    response_path = out / "responses.jsonl"
    raw = v.records(response_path) if response_path.is_file() else []
    cache = v.validate_raw(raw, plan, freeze["model_snapshot"])
    pending = {}
    for row in plan:
        pending.setdefault(row["input_sha256"], row)
    unfinished = [r for key, r in pending.items() if key not in cache]
    print("Frozen rows: {} | saved unique responses: {} | pending calls: {}".format(
        len(plan), len(cache), len(unfinished)))
    if unfinished:
        backend = backend or OllamaBackend(paths.OLLAMA_URL, timeout_seconds=600)
        snapshot = v.normalize_snapshot(backend.describe(v.CONFIG["model"]))
        v.require(snapshot == freeze["model_snapshot"],
                  "Model/runtime changed since freeze; restore the frozen snapshot to resume")
        with response_path.open("a", encoding="utf-8", newline="\n") as stream:
            for i, row in enumerate(unfinished, 1):
                payload = row["payload"]
                started = time.monotonic()
                response = backend.generate(request_id=row["row_id"],
                    system_prompt=payload["system_prompt"], user_prompt=payload["user_prompt"],
                    config=GenerationConfig(**payload["generation_config"]),
                    response_schema=payload["response_schema"])
                raw_row = dict(input_sha256=row["input_sha256"], payload=payload,
                    first_row_id=row["row_id"], model=response.model,
                    model_snapshot=snapshot, metadata=dict(response.metadata),
                    response_text=response.text, response_sha256=v.sha256_json(response.text),
                    response_id=response.response_id, usage=dict(response.usage),
                    latency_seconds=time.monotonic() - started, saved_at_utc=v.now())
                # Persist even invalid JSON. Validation/scoring never trigger a retry.
                stream.write(v.canonical_json(raw_row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                v.validate_raw([raw_row], plan, snapshot)
                cache[row["input_sha256"]] = raw_row
                print("[{}/{}] {}".format(i, len(unfinished), row["row_id"]), flush=True)
    cache = v.validate_raw(v.records(response_path), plan, freeze["model_snapshot"], complete=True)
    results = v.score(plan, cache)
    v.write_jsonl_atomic(out / "reader_results.jsonl", results)
    summary = dict(status="COMPLETE", completed_at_utc=v.now(), logical_rows=len(results),
                   unique_responses=len(cache), within_batch_cache_reuses=len(plan) - len(cache),
                   new_calls_this_invocation=len(unfinished), older_responses_reused=False,
                   freeze_manifest_sha256=v.digest(out / "freeze_manifest.json"),
                   responses_sha256=v.digest(response_path),
                   reader_results_sha256=v.digest(out / "reader_results.jsonl"),
                   model_snapshot=freeze["model_snapshot"])
    v.write_json_atomic(out / "run_summary.json", summary)
    print("COMPLETE | all {} logical rows retained; run analyze_validation.py".format(len(results)))


if __name__ == "__main__":
    run()
