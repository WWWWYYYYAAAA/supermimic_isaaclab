#!/usr/bin/env python3
"""Fetch SuperMimic's original assets without Docker (Python standard library only).

This performs downloads and integrity checks, not MJCF/USD conversion. LAFAN1
motions are CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
ROOT = Path(__file__).resolve().parents[1]
API = f"https://huggingface.co/api/datasets/{REPO}"


def read_json(url: str):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def digest(path: Path, entry: dict) -> str:
    if entry.get("lfs"):
        result = hashlib.sha256()
    else:
        result = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def expected_digest(entry: dict) -> str:
    return entry["lfs"]["oid"] if entry.get("lfs") else entry["oid"]


def valid(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["size"]
            and digest(path, entry) == expected_digest(entry))


def fetch(revision: str, entry: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid(target, entry):
        print(f"verified {target.relative_to(ROOT)}", flush=True)
        return
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}"
    temporary = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if not valid(temporary, entry):
                raise ValueError(f"Integrity check failed for {entry['path']}")
            os.replace(temporary, target)
            print(f"downloaded {target.relative_to(ROOT)}", flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    revision = read_json(API)["sha"]
    motions = read_json(f"{API}/tree/{revision}/g1")
    meshes = read_json(f"{API}/tree/{revision}/robot_description/g1/meshes")
    robot = read_json(f"{API}/tree/{revision}/robot_description/g1")
    motions = [entry for entry in motions if entry["type"] == "file" and entry["path"].endswith(".csv")]
    meshes = [entry for entry in meshes if entry["type"] == "file"]
    urdf = next(entry for entry in robot if entry["path"].endswith("/g1_29dof_rev_1_0.urdf"))
    if len(motions) != 40 or len(meshes) != 38:
        raise ValueError(f"Unexpected source contents: {len(motions)} motions, {len(meshes)} meshes")
    jobs = [(entry, ROOT / "data/lafan1_g1" / Path(entry["path"]).name) for entry in motions]
    jobs += [(entry, ROOT / "data/g1_description/meshes" / Path(entry["path"]).name) for entry in meshes]
    jobs += [(urdf, ROOT / "data/g1_description" / Path(urdf["path"]).name)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, revision, entry, target) for entry, target in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    frames = 0
    for entry in motions:
        with (ROOT / "data/lafan1_g1" / Path(entry["path"]).name).open(newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 36:
                    raise ValueError(f"Unexpected CSV columns in {entry['path']}: {len(row)}")
                [float(value) for value in row]
                frames += 1
    if frames != 264705:
        raise ValueError(f"Unexpected total frame count: {frames}")
    manifest = {"dataset": REPO, "revision": revision, "clips": len(motions),
                "frames": frames, "fps": 30, "meshes": len(meshes),
                "files": [{**entry, "local_path": str(target.relative_to(ROOT))} for entry, target in jobs]}
    (ROOT / "data/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {len(motions)} clips, {frames} frames, {len(meshes)} meshes, 1 URDF; revision {revision}")


if __name__ == "__main__":
    main()
