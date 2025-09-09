from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm

from .task_model import EEGTaskModel
from .dtos import BaseTaskDTO, TaskDTO, SubjectFilterDTO
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
        key = ("cohort", hash(task_dto))
        if key in self._cache:
            cohort_model = self._cache[key]
            print(f"{cohort_model.subject_length} subjects available")
            return cohort_model

        subjects = self.filter_subjects_by_dto(task_dto)
        subject_length = len(subjects)
        print(f"{subject_length} subjects found")

        task_models = []
        wanted_task = getattr(task_dto, "task", None)

        for subj in tqdm(subjects,
                         total=len(subjects),
                         desc="Loading task models",
                         leave=False):
            subj_tasks = self._task_index.get(subj, [])
            has_task = any(t == wanted_task for t, _ in subj_tasks)
            if not has_task:
                tqdm.write(f"Task '{wanted_task}' not found for subject {subj}")
                subject_length -= 1
                continue

            runs = [run for (t, run) in subj_tasks if t == wanted_task]
            if not runs:
                runs = [None]

            for run in runs:
                per_subj_dto = TaskDTO(subject=subj, task=wanted_task, run=run)
                tqdm.write(str(per_subj_dto))
                single_key = ("single", hash(per_subj_dto))
                if single_key not in self._cache:
                    self._cache[single_key] = EEGTaskModel(per_subj_dto, self._data_dir)
                task_models.append(self._cache[single_key])

        self._cache[key] = EEGCohortModel(task_dto, task_models, subject_length)
        return self._cache[key]

    @property
    def _participants_path(self):
        return self._data_dir / "participants.tsv"

    def _load_participants(self, filt: SubjectFilterDTO):
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

    def filter_subjects_by_dto(self, dto: SubjectFilterDTO):
        df = self._load_participants(dto).copy()

        for field_name in getattr(dto, "__dataclass_fields__", {}).keys():
            if field_name in ("task", "subject", "run", "ui_name", "ui_value"):
                continue

            if field_name.endswith("_range"):
                column_name = field_name[:-6]
                if column_name not in df.columns:
                    continue
                range_value = getattr(dto, field_name, None)
                if isinstance(range_value, (tuple, list)) and len(range_value) == 2:
                    lower, upper = range_value
                    numeric_values = pd.to_numeric(df[column_name], errors="coerce")
                    df = df[(numeric_values >= lower) & (numeric_values <= upper)]
            else:
                if field_name not in df.columns:
                    continue
                field_value = getattr(dto, field_name, None)
                if isinstance(field_value, (list, tuple)):
                    allowed_values = [v for v in field_value if v is not None]
                    if allowed_values:
                        df = df[df[field_name].isin(allowed_values)]
                elif field_value is not None:
                    df = df[df[field_name] == field_value]

        subject_ids = df["participant_id"].tolist()
        subject_ids = [s for s in subject_ids if s in self._subject_ids]

        return sorted(subject_ids)
