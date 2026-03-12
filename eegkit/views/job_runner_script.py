"""Standalone job runner entry point executed by JobRunner."""
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import resource

from eegkit.controller.eeg_controller import EEGController
import inspect
import importlib
import pkgutil
from eegkit.models import dtos as dtos_pkg
from eegkit.models.dtos.base import BaseTaskDTO
from eegkit.models.subject_model import EEGSubjectModel
from eegkit.utils.system import configure_logging

matplotlib.use('Agg')


def save_output(result, out_dir: Path):
    """Save controller result to disk under out_dir.

    Supports DataFrame, single Figure, list of Figures, JSON-serializable,
    text, and generic repr fallback. Returns a brief summary dict.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    summary = {"dir": str(out_dir)}
    import pandas as _pd
    import matplotlib.pyplot as _plt
    import numpy as _np
    from joblib import dump as _joblib_dump

    def _value_counts(arr):
        """Return JSON-serializable value counts for 1D-like arrays."""
        try:
            vals, counts = _np.unique(_np.asarray(arr), return_counts=True)
            return {str(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}
        except Exception:
            return {}

    if isinstance(result, _pd.DataFrame):
        fp = out_dir / 'dataframe.csv'
        result.to_csv(fp, index=False)
        summary.update({"type": "dataframe", "path": str(fp)})
        return summary

    # Single matplotlib Figure
    if hasattr(result, 'savefig'):
        fp = out_dir / 'figure.png'
        result.savefig(fp, dpi=150, bbox_inches='tight')
        _plt.close(result)
        summary.update({"type": "figure", "path": str(fp)})
        return summary

    # List of Figures
    if isinstance(result, list) and result and all(hasattr(f, 'savefig') for f in result):
        mdir = out_dir / 'multi'
        mdir.mkdir(exist_ok=True)
        for i, fig in enumerate(result, start=1):
            fp = mdir / f'fig_{i:02d}.png'
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            _plt.close(fig)
        summary.update({"type": "figures", "count": len(result), "path": str(mdir)})
        return summary

    # ML dataset dict: persist x/y/group as arrays + csv.
    if isinstance(result, dict) and {"x", "y", "group"}.issubset(result.keys()):
        dataset_name = str(result.get('name') or 'dataset')
        safe_name = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in dataset_name).strip('_')
        ds_dir = out_dir / (safe_name or 'dataset')
        ds_dir.mkdir(exist_ok=True)

        x = _np.asarray(result.get('x'))
        y = _np.asarray(result.get('y'))
        group = _np.asarray(result.get('group'))

        x_fp = ds_dir / 'x.npy'
        y_fp = ds_dir / 'y.npy'
        g_fp = ds_dir / 'group.npy'
        _np.save(x_fp, x)
        _np.save(y_fp, y)
        _np.save(g_fp, group)

        # Optional tabular export for quick inspection.
        rows = int(min(len(y), len(group), x.shape[0] if x.ndim > 0 else 0))
        preview = _pd.DataFrame({
            'y': y[:rows].tolist(),
            'group': group[:rows].tolist(),
        })
        preview_fp = ds_dir / 'labels_groups.csv'
        preview.to_csv(preview_fp, index=False)

        dataset_summary = {
            "name": dataset_name,
            "rows": rows,
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "group_shape": list(group.shape),
            "y_class_counts": _value_counts(y[:rows]),
            "group_counts": _value_counts(group[:rows]),
        }
        summary_fp = ds_dir / 'dataset_summary.json'
        summary_fp.write_text(json.dumps(dataset_summary, indent=2, default=str))

        summary.update({
            "type": "dataset",
            "name": dataset_name,
            "path": str(ds_dir),
            "x_path": str(x_fp),
            "y_path": str(y_fp),
            "group_path": str(g_fp),
            "rows": rows,
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "group_shape": list(group.shape),
            "y_class_counts": dataset_summary["y_class_counts"],
            "group_counts": dataset_summary["group_counts"],
            "dataset_summary_path": str(summary_fp),
        })
        return summary

    # Trained model payload: persist model object and metadata.
    if isinstance(result, dict) and "model" in result:
        model_obj = result.get("model")
        model_name = str(result.get("model_name") or "model")
        safe_name = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in model_name).strip('_')

        # Detect PyTorch nn.Module and save state dict with torch.save.
        if hasattr(model_obj, 'state_dict') and callable(model_obj.state_dict):
            import torch as _torch
            model_fp = out_dir / f"{safe_name or 'model'}_model.pt"
            _torch.save(model_obj.state_dict(), model_fp)
        else:
            model_fp = out_dir / f"{safe_name or 'model'}_model.gz"
            _joblib_dump(model_obj, model_fp)

        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        metadata = {**metadata, "model_path": str(model_fp)}
        meta_fp = model_fp.with_suffix(".json")
        meta_fp.write_text(json.dumps(metadata, indent=2, default=str))

        summary_payload = {k: v for k, v in result.items() if k != "model"}
        summary_payload["model_path"] = str(model_fp)
        output_fp = out_dir / 'output.json'
        output_fp.write_text(json.dumps(summary_payload, indent=2, default=str))

        summary.update({
            "type": "model",
            "model_path": str(model_fp),
            "metadata_path": str(meta_fp),
            "path": str(output_fp),
        })
        return summary

    # JSON-like (dict/list)
    if isinstance(result, (dict, list)):
        fp = out_dir / 'output.json'
        with open(fp, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        summary.update({"type": "json", "path": str(fp)})
        return summary

    # Plain text
    if isinstance(result, str):
        fp = out_dir / 'output.txt'
        fp.write_text(result)
        summary.update({"type": "text", "path": str(fp)})
        return summary

    # Fallback repr
    fp = out_dir / 'repr.txt'
    fp.write_text(repr(result))
    summary.update({"type": "repr", "path": str(fp)})
    return summary


def main(spec_path: str):
    """Load a job spec, run the controller action, and persist outputs."""
    configure_logging()
    log = logging.getLogger(__name__)
    with open(spec_path, 'r') as f:
        SPEC = json.load(f)
    JOB_DIR = Path(SPEC['job_dir'])
    JOB_DIR.mkdir(exist_ok=True, parents=True)

    jobs_root = None
    for parent in [JOB_DIR, *JOB_DIR.parents]:
        if parent.name == "jobs":
            jobs_root = parent
            break
    if jobs_root is None:
        jobs_root = JOB_DIR.parent

    # Build schema/params maps dynamically from exported DTOs to avoid manual updates.
    # - schema_map: subclasses of BaseTaskDTO (e.g., TaskDTO, SubjectFilterDTO)
    # - params_map: any class whose name ends with "ParamsDTO" (filtering, epoching, AI, etc.)
    # Collect classes from all submodules under eegkit.models.dtos
    discovered: dict[str, type] = {}
    # Ensure the package is imported; then scan submodules
    for modinfo in pkgutil.walk_packages(dtos_pkg.__path__, dtos_pkg.__name__ + "."):
        try:
            module = importlib.import_module(modinfo.name)
        except Exception:  # pragma: no cover - defensive; skip broken imports
            continue
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith("eegkit.models.dtos"):
                discovered[name] = obj

    schema_map = {
        name: cls for name, cls in discovered.items()
        if issubclass(cls, BaseTaskDTO) and cls is not BaseTaskDTO
    }
    params_map = {
        name: cls for name, cls in discovered.items()
        if name.endswith("ParamsDTO")
    }

    SchemaCls = schema_map.get(SPEC['schema_class'])
    if SchemaCls is None:
        raise RuntimeError(f"Unknown schema class {SPEC['schema_class']}")

    params_dto = None
    if SPEC['params_class']:
        ParamsCls = params_map.get(SPEC['params_class'])
        if ParamsCls is None:
            raise RuntimeError(f"Unknown params class {SPEC['params_class']}")
        params_dto = ParamsCls(**SPEC['params_kwargs'])

    task_dto = SchemaCls(**SPEC['schema_kwargs'])
    subject_model = EEGSubjectModel(SPEC['data_dir'])
    controller = EEGController(subject_model, jobs_root=jobs_root)

    log.info("[JOB] Starting %s -> %s/%s", SPEC['job_id'], SPEC['group'], SPEC['key'])
    log.info("[JOB] task_dto = %s", task_dto)
    log.info("[JOB] params    = %s", params_dto)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    log.info("[JOB][RSS] start: %s KB", usage.ru_maxrss)

    try:
        result = controller.show(task_dto, SPEC['group'], SPEC['key'], params_dto)
    except Exception as e:
        log.exception('[JOB][ERROR] %s', e)
        (JOB_DIR / 'ERROR').write_text(str(e))
        error_json = {
            'error': str(e),
            'type': type(e).__name__
        }
        (JOB_DIR / 'error.json').write_text(json.dumps(error_json, indent=2))
        sys.exit(1)

    # Per-subject batch dict handling (subject -> result).
    # Keep dataset payloads {x, y, group} on the normal save_output path.
    is_dataset_payload = isinstance(result, dict) and {"x", "y", "group"}.issubset(result.keys())
    is_model_payload = isinstance(result, dict) and "model" in result
    if isinstance(result, dict) and result and not is_dataset_payload and not is_model_payload and all(isinstance(k, str) for k in result.keys()):
        manifest = {"subjects": [], "errors": {}}
        summary_rows = []
        for subj, value in result.items():
            subj_dir = JOB_DIR / subj
            row = {"subject": subj, "status": "ok", "path": str(subj_dir), "error": None}
            if isinstance(value, dict) and 'error' in value:
                manifest['errors'][subj] = value['error']
                (subj_dir / 'error.json').write_text(json.dumps(value, indent=2))
                row["status"] = "error"
                row["error"] = value['error']
            else:
                save_output(value, subj_dir)
            manifest['subjects'].append(subj)
            summary_rows.append(row)
        (JOB_DIR / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            pd.DataFrame(summary_rows).to_csv(JOB_DIR / 'subjects.csv', index=False)
        except Exception:
            (JOB_DIR / 'subjects.json').write_text(json.dumps(summary_rows, indent=2))
        try:
            meta_df = subject_model.get_subjects_metadata(manifest['subjects'])
            meta_fp = JOB_DIR / 'participants_selected.csv'
            meta_df.to_csv(meta_fp, index=False)
            log.info("[JOB] Wrote metadata table -> %s", meta_fp)
        except Exception as e:
            log.warning("[JOB][WARN] Failed to write participants_selected.csv: %s", e)
        log.info("[JOB] Saved per-subject batch outputs for %d subjects", len(manifest['subjects']))
    else:
        info = save_output(result, JOB_DIR)
        log.info("[JOB] Saved output -> %s", info.get('path', info.get('dir')))

    plt.close('all')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    log.info("[JOB][RSS] end: %s KB", usage.ru_maxrss)
    log.info('[JOB] Done.')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python job_runner_script.py spec.json")
        sys.exit(1)
    main(sys.argv[1])
