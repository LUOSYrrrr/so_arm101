"""Export and optionally publish the reviewed battery-insertion dataset.

The dashboard used a fixed Pick-and-Lift task string. Correct only the export;
the original recording and its human outcome labels remain untouched.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi


REPO = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("SO101_STORAGE_ROOT", REPO.parent)).resolve()
SOURCE = STORAGE / "datasets/so101_battery_insertion_v2_20260930_231513_30cc"
NAME = "so101_battery_insertion_v2"
TASK = "Pick up the black battery, insert it into the charger slot until seated, and release the gripper."


def export() -> tuple[Path, dict, dict]:
    report = json.loads((SOURCE / "meta/validation.json").read_text())
    info = json.loads((SOURCE / "meta/info.json").read_text())
    labels = [json.loads(line) for line in (SOURCE / "meta/benchmark_episodes.jsonl").read_text().splitlines() if line]
    selected = report["valid_success_episodes"]
    if not report["passed"] or report["total_frames"] != info["total_frames"]:
        raise RuntimeError("Source validation is missing, failed, or stale")
    if info["codebase_version"] != "v3.0" or len(selected) != 100 or len(labels) != info["total_episodes"]:
        raise RuntimeError("Unexpected source dataset version or outcome counts")
    if any(not labels[i]["human_success"] for i in selected):
        raise RuntimeError("Success selection contains a non-success episode")

    dest = STORAGE / "outputs/hf_exports" / NAME
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".building")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for folder in ("data", "meta", "videos"):
            shutil.copytree(SOURCE / folder, tmp / folder)
        benchmark = json.loads((tmp / "meta/benchmark.json").read_text())
        old_task = benchmark["task"]
        tasks = pd.read_parquet(tmp / "meta/tasks.parquet")
        if tasks.index.name != "task" or set(tasks.index) != {old_task}:
            raise RuntimeError("Unexpected source task table")
        tasks.index = pd.Index([TASK] * len(tasks), name="task")
        tasks.to_parquet(tmp / "meta/tasks.parquet")
        for path in (tmp / "meta/episodes").rglob("*.parquet"):
            episodes = pd.read_parquet(path)
            episodes["tasks"] = episodes["tasks"].map(
                lambda values: [TASK if value == old_task else value for value in values]
            )
            episodes.to_parquet(path, index=False)
        benchmark.update(
            task=TASK,
            task_type="battery_insertion",
            repo_id="LUOSYrrrrr/" + NAME,
            annotation_correction={
                "original_task": old_task,
                "reason": "The collector used fixed Pick-and-Lift text; the owner confirmed battery insertion into the charger and gripper release.",
            },
        )
        for key in ("session_dir", "dataset_root", "monitor_url", "cube_dimensions_mm", "lift_height_mm", "hold_seconds"):
            benchmark.pop(key, None)
        benchmark["calibration_sha256"] = {Path(k).name: v for k, v in benchmark["calibration_sha256"].items()}
        (tmp / "meta/benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n")
        export_report = json.loads((tmp / "meta/validation.json").read_text())
        export_report["warnings"] = [w for w in export_report["warnings"] if "5 cm" not in w]
        export_report["warnings"].append(
            "Success is human annotated and was not verified automatically as a fully seated battery with gripper release."
        )
        (tmp / "meta/validation.json").write_text(json.dumps(export_report, ensure_ascii=False, indent=2) + "\n")
        (tmp / "README.md").write_text(
            f"""---
tags:
- lerobot
- robotics
- so101
- imitation-learning
---
# SO-101 battery insertion — dual RGB

Task: {TASK}

LeRobot v3.0, 20 FPS, {info['total_episodes']} recorded episodes and {len(selected)} human-labeled successful demonstrations. Cameras: `observation.images.wrist` and `observation.images.third_person`, 640×480 RGB. State and action are six joint values; body joints are in degrees and gripper is 0–100. Actions are sent joint targets.

The original collector accidentally stored a Pick-and-Lift task description. This export corrects the task table, episode task metadata and benchmark description without changing the recorded frames, videos, actions, or outcome labels. See `meta/benchmark.json` for provenance.

Failures and excluded attempts are retained. Train imitation-learning policies with episode indices from `meta/act_success_episodes.json`, or use `meta/validation.json` → `valid_success_episodes`. Technical validation passed, but does not automatically verify physical insertion. Initial-pose warnings require review.

The two camera streams are software paired by host receipt time, not hardware synchronized. Preview rotation in the dashboard does not rotate the stored videos.
"""
        )
        files = {
            str(path.relative_to(tmp)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp.rglob("*") if path.is_file() and path.name != "export_sha256.json"
        }
        (tmp / "meta/export_sha256.json").write_text(json.dumps(files, indent=2) + "\n")
        tmp.rename(dest)
    else:
        benchmark = json.loads((dest / "meta/benchmark.json").read_text())
        if benchmark["task"] != TASK:
            raise RuntimeError("Existing export has a different task description")
    # The training code reads dataset.meta.stats even when --dataset.episodes
    # selects successes. Compute state/action normalization from those same
    # successes so excluded and failed episodes cannot affect the processors.
    stats_path = dest / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    selected_frames = sum(labels[i]["frames"] for i in selected)
    if stats["action"]["count"] != [selected_frames]:
        values = {"action": [], "observation.state": []}
        selected_set = set(selected)
        for path in sorted((dest / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "action", "observation.state"])
            episode_index = np.asarray(table["episode_index"].to_numpy())
            mask = np.isin(episode_index, list(selected_set))
            if not mask.any():
                continue
            for key in values:
                values[key].append(np.asarray(table[key].to_pylist(), dtype=np.float64)[mask])
        for key, chunks in values.items():
            array = np.concatenate(chunks)
            if len(array) != selected_frames or array.shape[1] != 6 or not np.isfinite(array).all():
                raise RuntimeError(f"Selected {key} frames are incomplete or invalid")
            stats[key] = {
                "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
                "mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist(),
                "count": [len(array)],
                **{f"q{round(q * 100):02d}": np.quantile(array, q, axis=0).tolist()
                   for q in (0.01, 0.10, 0.50, 0.90, 0.99)},
            }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        benchmark_path = dest / "meta/benchmark.json"
        benchmark = json.loads(benchmark_path.read_text())
        benchmark["normalization_stats"] = "observation.state and action use only the 100 selected success episodes"
        benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n")
        readme_path = dest / "README.md"
        readme_path.write_text(readme_path.read_text().replace(
            "Failures and excluded attempts are retained.",
            "Failures and excluded attempts are retained. State/action normalization statistics use only the 100 selected successes."
        ))
        files = {
            str(path.relative_to(dest)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dest.rglob("*") if path.is_file() and path.name != "export_sha256.json"
        }
        (dest / "meta/export_sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    return dest, info, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    dest, info, report = export()
    repo_id = "LUOSYrrrrr/" + NAME
    revision = None
    if args.upload:
        api = HfApi()
        if api.whoami()["name"] != "LUOSYrrrrr":
            raise RuntimeError("Authenticated Hugging Face account does not own the target namespace")
        api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        remote = api.dataset_info(repo_id)
        if remote.private == args.public:
            raise RuntimeError("Existing repository visibility differs from requested visibility")
        commit = api.upload_folder(
            folder_path=dest,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Publish battery insertion recordings with corrected task metadata",
        )
        revision = commit.oid
        remote = api.dataset_info(repo_id, revision=revision, files_metadata=True)
        local_files = {str(path.relative_to(dest)): path.stat().st_size for path in dest.rglob("*") if path.is_file()}
        remote_files = {file.rfilename: file.size for file in remote.siblings}
        if any(remote_files.get(path) != size for path, size in local_files.items()):
            raise RuntimeError("Remote dataset file list or sizes do not match the export")
    index_path = REPO / "data/datasets.json"
    index = json.loads(index_path.read_text())
    index["battery_insertion"] = {
        "repo_id": repo_id,
        "revision": revision,
        "root": str(dest),
        "success_episodes": report["valid_success_episodes"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "private": not args.public,
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"repo_id": repo_id, "revision": revision, "export": str(dest), "successes": len(report["valid_success_episodes"])}))


if __name__ == "__main__":
    main()
