import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HiFuse import DATASETS, run_dataset, select_datasets


METRIC_COLUMNS = (
    "AMI",
    "ARI",
    "Completeness",
    "F1",
    "FMI",
    "Jaccard",
    "MI",
    "NMI",
    "VMeasure",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed HiFuse baseline on the bundled datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional exact dataset names. The default is all bundled datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "Data",
        help="Directory containing one folder per dataset.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline_14",
        help="Output directory for per-dataset files and summaries.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a dataset only when its metrics.json already exists.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip UMAP and spatial comparison plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without training.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed dataset.",
    )
    return parser.parse_args()


def _load_metrics(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return _metric_subset(metrics)


def _metric_subset(metrics):
    return {key: metrics[key] for key in METRIC_COLUMNS if key in metrics}


def _write_metrics(path, metrics):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_metric_subset(metrics), handle, indent=2, ensure_ascii=False)


def _validate_inputs(data_root, datasets):
    missing = []
    for spec in datasets:
        dataset_dir = data_root / spec.name
        for filename in (
            spec.rna_filename,
            spec.secondary_filename,
            spec.labels.filename,
        ):
            path = dataset_dir / filename
            if not path.is_file():
                missing.append(path)
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{details}")


def _write_summaries(records, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    clean_records = []
    for record in records:
        clean_record = dict(record)
        if "metrics" in clean_record:
            clean_record["metrics"] = _metric_subset(clean_record["metrics"] or {})
        clean_records.append(clean_record)

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clean_records, handle, indent=2, ensure_ascii=False)

    rows = []
    for record in clean_records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"metrics", "traceback"}
        }
        metrics = record.get("metrics") or {}
        row.update({key: metrics.get(key) for key in METRIC_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / "summary.csv", index=False)

    successful = [row for row in rows if row.get("status") in {"ok", "skipped_existing"}]
    final_columns = ["dataset"] + list(METRIC_COLUMNS)
    pd.DataFrame(successful).reindex(columns=final_columns).to_csv(
        output_root / "final_metrics.csv",
        index=False,
    )


def main():
    args = parse_args()
    datasets = select_datasets(args.datasets)
    data_root = args.data_dir.resolve()
    output_root = args.outdir.resolve()
    _validate_inputs(data_root, datasets)

    if args.dry_run:
        return 0

    records = []
    for index, spec in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {spec.name}", flush=True)
        metrics_path = output_root / spec.name / "metrics.json"
        if args.skip_existing and metrics_path.is_file():
            metrics = _load_metrics(metrics_path)
            _write_metrics(metrics_path, metrics)
            record = {
                "dataset": spec.name,
                "status": "skipped_existing",
                "seed": 2022,
                "epochs": spec.epochs,
                "output": str(metrics_path.parent),
                "metrics": metrics,
            }
        else:
            started = time.perf_counter()
            try:
                record = run_dataset(
                    spec,
                    data_root=data_root,
                    output_root=output_root,
                    device_name=args.device,
                    seed=2022,
                    make_plots=not args.no_plots,
                )
                record["duration_seconds"] = round(time.perf_counter() - started, 2)
            except KeyboardInterrupt:
                record = {
                    "dataset": spec.name,
                    "status": "interrupted",
                    "duration_seconds": round(time.perf_counter() - started, 2),
                }
                records.append(record)
                _write_summaries(records, output_root)
                print("Interrupted; completed results were retained.", flush=True)
                return 130
            except Exception as error:
                record = {
                    "dataset": spec.name,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "duration_seconds": round(time.perf_counter() - started, 2),
                }
                error_dir = output_root / "logs"
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / f"{spec.name}.log").write_text(
                    record["traceback"],
                    encoding="utf-8",
                )
                print(f"  failed: {error}", flush=True)

        records.append(record)
        _write_summaries(records, output_root)
        metrics = record.get("metrics") or {}
        if "ARI" in metrics:
            print(
                f"  status={record['status']} ARI={metrics['ARI']:.6f} "
                f"NMI={metrics['NMI']:.6f}",
                flush=True,
            )
        if record["status"] == "failed" and args.fail_fast:
            break

    failures = [record for record in records if record["status"] == "failed"]
    print(f"Finished: {len(records) - len(failures)} succeeded, {len(failures)} failed.", flush=True)
    print(f"Summary: {output_root / 'final_metrics.csv'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
