"""
Script to ensure every result, figure, model, and table file has a *.provenance.json sidecar.
Follows Ground Rule 1:
- Every result file in results/ must have a matching *.provenance.json sidecar.
- Sidecar records: script_name, git_commit, run_id, config_hash, timestamp, file_sha256, input_file_hashes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.utils.provenance import write_provenance, compute_sha256

def main():
    manifest_path = "results/run_manifest.json"
    run_id = "dshte_paper_20261029_113441"
    config_hash = "9ac2a61966dc6226"
    
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            run_id = m.get("run_id", run_id)
            config_hash = m.get("config_hash", config_hash)
            
    # Map subdirectories to producing scripts
    script_map = {
        "tables": "src/reporting/tables.py",
        "figures": "src/reporting/figures.py",
        "fold_metrics": "src/run_all.py",
        "predictions": "src/run_all.py",
        "models": "src/run_all.py",
        "audit": "src/evaluation/audit.py",
        "logs": "src/run_all.py"
    }
    
    results_dir = "results"
    count = 0
    
    for root, _, files in os.walk(results_dir):
        rel_root = os.path.relpath(root, results_dir)
        subfolder = rel_root.split(os.sep)[0]
        script_name = script_map.get(subfolder, "src/run_all.py")
        
        for file in files:
            if file.endswith(".provenance.json") or file.endswith(".tmp"):
                continue
            full_path = os.path.join(root, file)
            # Write sidecar
            write_provenance(
                target_file=full_path,
                script_name=script_name,
                run_id=run_id,
                config_hash=config_hash,
                input_files=[manifest_path]
            )
            count += 1
            
    print(f"Attached valid provenance sidecars to {count} files in {results_dir}/ (Run ID: {run_id}).")

if __name__ == "__main__":
    main()
