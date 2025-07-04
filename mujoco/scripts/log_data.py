#!/usr/bin/env python3
# run_batch.py  ── launch several `model.py` jobs in parallel
#
# Only ONE helper function and one parallel loop.
# Edit the `runs` list and run:  python3 run_batch.py
# ---------------------------------------------------------------------

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------
# 1)  Describe every run (only the four paths you care about)
# ---------------------------------------------------------------------
runs = [
    {
        "traj_path"  : "../../data/run_data.json",
        "models_path": "../models/dynobench/2payload.yaml",
        "mj"         : "../models/xml/2cfs_payload_tendons_30cm.xml",
        "out_path"   : f"../../data/output/run_data{i}.json",
    }

    # # Add more dicts ↓ (copy-paste and change the paths)
    # {
    #     "traj_path"  : "../../data/run_data.json",
    #     "models_path": "../models/dynobench/2payload.yaml",
    #     "mj"         : "../models/xml/2cfs_payload_tendons_30cm.xml",
    #     "out_path"   : "../../data/output/run_data2.json",
    # },
    # {
    #     "traj_path"  : "../../data/run_data.json",
    #     "models_path": "../models/dynobench/2payload.yaml",
    #     "mj"         : "../models/xml/2cfs_payload_tendons_30cm.xml",
    #     "out_path"   : "../../data/output/run_data3.json",
    # },
    #     {
    #     "traj_path"  : "../../data/run_data.json",
    #     "models_path": "../models/dynobench/2payload.yaml",
    #     "mj"         : "../models/xml/2cfs_payload_tendons_30cm.xml",
    #     "out_path"   : "../../data/output/run_data4.json",
    # },
for i in range(10)]

# ---------------------------------------------------------------------
# 2)  One helper that builds *exactly* the CLI you gave
# ---------------------------------------------------------------------
def launch(run):
    cmd = [
        "python3", "model.py",
        "--traj_path",   run["traj_path"],
        "--models_path", run["models_path"],
        "-p", "-t",
        "--mj",          run["mj"],
        "--out_path",    run["out_path"],
    ]
    print("↳", " ".join(cmd))
    subprocess.run(cmd, check=True)          # raises if model.py fails

# ---------------------------------------------------------------------
# 3)  Parallel pool – as many workers as CPU cores (or fewer if < runs)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    max_workers = min(os.cpu_count() or 1, len(runs))
    print(f"Starting {len(runs)} jobs on {max_workers} CPU cores.\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(launch, r) for r in runs]
        for f in as_completed(futures):      # propagates any exceptions
            f.result()

    print("\n✓ All jobs finished without errors.")
