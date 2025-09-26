import json, matplotlib, sys
import os, resource
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from eegkit import *

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
    
    if isinstance(result, _pd.DataFrame):
        fp = out_dir / 'dataframe.csv'
        result.to_csv(fp, index=False)
        summary.update({"type": "dataframe", "path": str(fp)})
        return summary

    # Single matplotlib Figure
    if hasattr(result, 'savefig'):
        fp = out_dir / 'figure.png'
        result.savefig(fp, dpi=150, bbox_inches='tight')
        try:
            _plt.close(result)
        except Exception:
            pass
        summary.update({"type": "figure", "path": str(fp)})
        return summary

    # List of Figures
    if isinstance(result, list) and result and all(hasattr(f, 'savefig') for f in result):
        mdir = out_dir / 'multi'
        mdir.mkdir(exist_ok=True)
        for i, fig in enumerate(result, start=1):
            fp = mdir / f'fig_{i:02d}.png'
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            try:
                _plt.close(fig)
            except Exception:
                pass
        summary.update({"type": "figures", "count": len(result), "path": str(mdir)})
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
    with open(spec_path, 'r') as f:
        SPEC = json.load(f)
    JOB_DIR = Path(SPEC['job_dir'])
    JOB_DIR.mkdir(exist_ok=True, parents=True)

    schema_map = {c.__name__: c for c in [TaskDTO, SubjectFilterDTO]}
    params_map = {c.__name__: c for c in [
        FilterParamsDTO, EpochParamsDTO, PSDParamsDTO,
        EpochPSDParamsDTO, TimeDomainParamsDTO, TableInfoDTO, EvokedParamsDTO, EvokedTopoParamsDTO, EvokedJointParamsDTO
    ]}

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
    controller = EEGController(subject_model)

    print(f"[JOB] Starting {SPEC['job_id']} -> {SPEC['group']}/{SPEC['key']}", flush=True)
    print(f"[JOB] task_dto = {task_dto}", flush=True)
    print(f"[JOB] params    = {params_dto}", flush=True)
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(f"[JOB][RSS] start: {usage.ru_maxrss} KB", flush=True)
    except Exception:
        pass

    try:
        result = controller.show(task_dto, SPEC['group'], SPEC['key'], params_dto)
    except Exception as e:
        import traceback
        print('[JOB][ERROR]', e, file=sys.stderr)
        traceback.print_exc()
        (JOB_DIR / 'ERROR').write_text(str(e))
        error_json = {
            'error': str(e),
            'type': type(e).__name__
        }
        (JOB_DIR / 'error.json').write_text(json.dumps(error_json, indent=2))
        sys.exit(1)

    # Per-subject batch dict handling (subject -> result)
    if isinstance(result, dict) and result and all(isinstance(k, str) for k in result.keys()):
        manifest = {"subjects": [], "errors": {}}
        for subj, value in result.items():
            subj_dir = JOB_DIR / subj
            if isinstance(value, dict) and 'error' in value:
                manifest['errors'][subj] = value['error']
                (subj_dir / 'error.json').write_text(json.dumps(value, indent=2))
            else:
                save_output(value, subj_dir)
            manifest['subjects'].append(subj)
        (JOB_DIR / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f"[JOB] Saved per-subject batch outputs for {len(manifest['subjects'])} subjects")
    else:
        info = save_output(result, JOB_DIR)
        print(f"[JOB] Saved output -> {info.get('path', info.get('dir'))}")
        
    try:
        plt.close('all')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(f"[JOB][RSS] end: {usage.ru_maxrss} KB", flush=True)
    except Exception:
        pass
    print('[JOB] Done.', flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python job_runner_script.py spec.json")
        sys.exit(1)
    main(sys.argv[1])
