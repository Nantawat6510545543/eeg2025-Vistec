#!/usr/bin/env python3
import json, matplotlib, sys
matplotlib.use('Agg')
from pathlib import Path
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

from eegkit.models.dtos import (
    TaskDTO, SubjectFilterDTO, FilterParamsDTO, EpochParamsDTO, PSDParamsDTO,
    EpochPSDParamsDTO, TimeDomainParamsDTO, TableInfoDTO
)
from eegkit.models.subject_model import EEGSubjectModel
from eegkit.controller.eeg_controller import EEGController

def main(spec_path: str):
    SPEC = json.load(open(spec_path, 'r'))
    JOB_DIR = Path(SPEC['job_dir'])
    FIG_DIR = JOB_DIR / 'figures'
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # Map class names → DTOs
    schema_map = {c.__name__: c for c in [TaskDTO, SubjectFilterDTO]}
    params_map = {c.__name__: c for c in [
        FilterParamsDTO, EpochParamsDTO, PSDParamsDTO,
        EpochPSDParamsDTO, TimeDomainParamsDTO, TableInfoDTO
    ]}

    # Reconstruct schema and params
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
        result = controller.show(task_dto, SPEC['group'], SPEC['key'], params_dto)
    except Exception as e:
        import traceback
        print('[JOB][ERROR]', e, file=sys.stderr)
        traceback.print_exc()
        (JOB_DIR / 'ERROR').write_text(str(e))
        sys.exit(1)

    # Persist outputs
    print(type(result))
    if isinstance(result, pd.DataFrame):
        out_csv = JOB_DIR / 'dataframe.csv'
        result.to_csv(out_csv, index=False)
        print(f"[JOB] Saved DataFrame -> {out_csv}")
    elif isinstance(result, list) and result and all(hasattr(f, 'savefig') for f in result):
        if len(result) > 1:
            multi_dir = FIG_DIR / 'multi'
            multi_dir.mkdir(exist_ok=True)
            for i, fig in enumerate(result, start=1):
                fpath = multi_dir / f'fig_{i:02d}.png'
                fig.savefig(fpath, dpi=150, bbox_inches='tight')
                plt.close(fig)
            print(f"[JOB] Saved {len(result)} figures in {multi_dir}")
        else:
            fpath = FIG_DIR / 'figure.png'
            result[0].savefig(fpath, dpi=150, bbox_inches='tight')
            plt.close(result[0])
            print(f"[JOB] Saved single figure -> {fpath}")
    elif isinstance(result, (dict, list)):
        out_json = JOB_DIR / 'output.json'
        json.dump(result, open(out_json, 'w'), indent=2)
        print(f"[JOB] Saved JSON -> {out_json}")
    elif isinstance(result, str):
        out_txt = JOB_DIR / 'output.txt'
        out_txt.write_text(result)
        print(f"[JOB] Saved text -> {out_txt}")
    elif result is not None:
        out_txt = JOB_DIR / 'repr.txt'
        out_txt.write_text(repr(result))
        print(f"[JOB] Saved repr -> {out_txt}")

    print('[JOB] Done.', flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python job_runner_script.py spec.json")
        sys.exit(1)
    main(sys.argv[1])
