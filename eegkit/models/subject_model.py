from pathlib import Path
import re
from collections import defaultdict
import pandas as pd

from .task_model import EEGTaskModel
from .dtos import BaseTaskDTO
from .cohort_model import EEGCohortModel


class EEGSubjectModel:
    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)
        self._subject_ids = self._discover_subjects()
        self._task_index = self._discover_tasks()
        self._cache = {}
        self._participants_df = None

    def _discover_subjects(self):
        return sorted([p.name for p in self._data_dir.glob("sub-*") if p.is_dir()])

    def _discover_tasks(self):
        task_map = defaultdict(list)
        pattern = re.compile(
            r"(sub-(?P<subject>[^_]+))_task-(?P<task>[^_]+)(?:_run-(?P<run>\d+))?_eeg\.set"
        )

        for subj_dir in self._data_dir.glob("sub-*"):
            eeg_dir = subj_dir / "eeg"
            if not eeg_dir.exists():
                continue

            for eeg_file in eeg_dir.glob("sub-*_task-*_eeg.set"):
                match = pattern.match(eeg_file.name)
                if match:
                    full_subj = match.group(1)
                    task = match.group("task")
                    run = match.group("run")
                    task_map[full_subj].append((task, run))

        for subj, tasks in list(task_map.items()):
            task_runs = defaultdict(set)
            for task, run in tasks:
                task_runs[task].add(run)

            for task, runs in task_runs.items():
                runs_trial = len([r for r in runs if r is not None])
                if runs_trial > 1:
                    task_map[subj].append((task, f"All-{runs_trial}"))

        return dict(task_map)

    def list_subjects(self):
        return self._subject_ids

    def list_tasks(self, subject):
        return sorted(self._task_index.get(subject, []))
    
    def get_task(self, task_dto: BaseTaskDTO):
        # single-subject
        if hasattr(task_dto, "subject") and getattr(task_dto, "subject") is not None:
            key = ("single", hash(task_dto))
            if key not in self._cache:
                self._cache[key] = EEGTaskModel(task_dto, self._data_dir)
            return self._cache[key]

        # cohort
        subjects = self.filter_subjects_by_dto(task_dto)
        print(f"{len(subjects)} subjects found")
        key = ("cohort", hash(task_dto))
        if key not in self._cache:
            self._cache[key] = EEGCohortModel(task_dto, subjects, self._data_dir)
        return self._cache[key]

    @property
    def _participants_path(self):
        return self._data_dir / "participants.tsv"

    def _load_participants(self, filt: BaseTaskDTO):
        if self._participants_df is not None:
            return self._participants_df

        cols = {"participant_id"}
        for name in getattr(filt, "__dataclass_fields__", {}).keys():
            if name in ("task", "subject", "run"):
                continue
            if name.endswith("_range"):
                base = name[:-6]
                cols.add(base)
            else:
                cols.add(name)

        df = pd.read_csv(self._participants_path, sep="\t")
        task = getattr(filt, "task", None)
        df = self.filter_available(df, task)
        df = df[[c for c in cols if c in df.columns]]
        df["participant_id"] = df["participant_id"].astype(str)
        self._participants_df = df
        return df

    def filter_available(self, df: pd.DataFrame, task: str) -> pd.DataFrame:
        cols = df.columns

        if task in cols:
            mask = df[task].astype('string').str.lower().eq('available')
            return df[mask].copy()

        prefix = f"{task}_"
        group_cols = [c for c in cols if c.startswith(prefix)]
        if group_cols:
            mask = (
                df[group_cols]
                .astype('string')
                .apply(lambda s: s.str.lower().eq('available'))
                .any(axis=1)
            )
            return df[mask].copy()

        raise KeyError(f"'{task}' not found as a column: {cols}")

    def filter_subjects_by_dto(self, filt: BaseTaskDTO):
        df = self._load_participants(filt).copy()

        for name in getattr(filt, "__dataclass_fields__", {}).keys():
            if name in ("task", "subject", "run", "ui_name", "ui_value"):
                continue
            if name.endswith("_range"):
                col = name[:-6]
                if col not in df.columns:
                    print("skip " + col)
                    continue
                rng = getattr(filt, name, None)
                if isinstance(rng, (tuple, list)) and len(rng) == 2:
                    lo, hi = rng
                    vals = pd.to_numeric(df[col], errors="coerce")
                    df = df[(vals >= lo) & (vals <= hi)]
            else:
                if name not in df.columns:
                    print("name " + col)
                    continue
                val = getattr(filt, name, None)
                if isinstance(val, (list, tuple)):
                    choices = [v for v in val if v is not None]
                    if choices:
                        df = df[df[name].isin(choices)]
                elif val is not None:
                    df = df[df[name] == val]

        subs = df["participant_id"].tolist()
        subs = [s for s in subs if s in self._subject_ids]

        return sorted(subs)
