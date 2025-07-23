from pathlib import Path
import re
from collections import defaultdict
from .task import EEGTaskData


class EEGSubjectData:
    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)
        self._subject_ids = self._discover_subjects()
        self._task_index = self._discover_tasks()
        self._cache = {}  # (subj, task, run) → EEGTaskData

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

    def get_task(self, subject, task, run=None):
        key = (subject, task, run)
        if key not in self._cache:
            task_data = EEGTaskData(
                subject=subject,
                task=task,
                run=run,
                data_dir=self._data_dir,
            )
            self._cache[key] = task_data
        return self._cache[key]