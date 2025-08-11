import os
import re
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

class BIDS_EEG_Viewer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.subjects = self._detect_subjects()
        self.subject = None
        self.task_info = []

    def _suppress_warnings(self):
        warnings.filterwarnings("ignore", message=".*boundary.*data discontinuities.*")

    def _detect_subjects(self):
        pattern = re.compile(r"sub-(NDAR\w+)_task-.*_eeg\.set")
        subjects = set()
        for file in self.data_dir.glob("sub-*_task-*_eeg.set"):
            match = pattern.match(file.name)
            if match:
                subjects.add(match.group(1))
        return sorted(subjects)

    def set_subject(self, subject_id: str):
        self.subject = subject_id
        self.task_info = []
        self._suppress_warnings()
        self._gather_tasks()

    def _gather_tasks(self):
        pattern = re.compile(rf"sub-{self.subject}_task-(?P<task>.+?)(?:_run-(?P<run>\\d+))?_eeg.set")
        eeg_files = sorted(self.data_dir.glob(f"sub-{self.subject}_task-*_eeg.set"))
        for file in eeg_files:
            match = pattern.match(file.name)
            if match:
                self.task_info.append({
                    "task": match.group("task"),
                    "run": match.group("run")
                })

    def get_available_tasks(self):
        return sorted({info['task'] for info in self.task_info})

    def _build_path(self, task, ext, run=None):
        name = f"sub-{self.subject}_task-{task}"
        if run:
            name += f"_run-{run}"
        return self.data_dir / f"{name}_{ext}"

    def _load_metadata(self, task, run):
        path = self._build_path(task, 'eeg.json', run)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _load_tsv(self, task, kind, run):
        path = self._build_path(task, f'{kind}.tsv', run)
        return pd.read_csv(path, sep='\t') if path.exists() else None

    def _prepare_raw(self, task, run):
        eeg_path = self._build_path(task, 'eeg.set', run)
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True, montage_units='cm')
        raw.drop_channels(['Cz'])
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        raw.set_montage(montage, match_case=False)
        return raw

    def get_raw(self, task, run=None):
        return self._prepare_raw(task, run)

    def get_metadata(self, task, run=None):
        return self._load_metadata(task, run)

    def get_event_data(self, task, run=None):
        return self._load_tsv(task, 'events', run)

    def get_channel_data(self, task, run=None):
        return self._load_tsv(task, 'channels', run)

    def get_electrode_data(self, task, run=None):
        return self._load_tsv(task, 'electrodes', run)


if __name__ == '__main__':
    viewer = BIDS_EEG_Viewer(data_dir='/mount/sub/cmi_bids_R1/eeg')
    viewer.set_subject('NDARAC904DMU')
    print("Subjects:", viewer.subjects)
    print("Available tasks:", viewer.get_available_tasks())
