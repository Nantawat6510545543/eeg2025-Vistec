from pathlib import Path
import re
from collections import defaultdict
from .task_model import EEGTaskModel
from .dtos import TaskDTO


class EEGSubjectModel:
    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)
        self._subject_ids = self._discover_subjects()
        self._task_index = self._discover_tasks()
        self._cache = {}

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
                    task_map[subj].append((task, f"All {runs_trial}"))

        return dict(task_map)

    def list_subjects(self):
        return self._subject_ids

    def list_tasks(self, subject):
        return sorted(self._task_index.get(subject, []))

    def get_task(self, task_dto: TaskDTO):
        key = (task_dto.subject, task_dto.task, task_dto.run)
        if key not in self._cache:
            task_model = EEGTaskModel(task_dto, self._data_dir)
            self._cache[key] = task_model

        return self._cache[key]
