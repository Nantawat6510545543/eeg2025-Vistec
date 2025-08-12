from pathlib import Path
import json
import pandas as pd
import mne
from .dtos import TaskDTO


class EEGTaskLoader:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto
        self.data_dir = Path(data_dir)

    def load_raw(self):
        path = self._get_file("eeg.set")
        raw = mne.io.read_raw_eeglab(path, preload=True, montage_units='cm')
        raw.drop_channels(['Cz'])
        raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), match_case=False)
        return raw

    def load_metadata(self):
        return self._load_json("eeg.json")

    def load_events(self):
        return self._load_tsv("events.tsv")

    def load_channels(self):
        return self._load_tsv("channels.tsv")

    def load_electrodes(self):
        return self._load_tsv("electrodes.tsv")

    def _get_file(self, ext):
        base = f"{self.task_dto.subject}_task-{self.task_dto.task}"
        if self.task_dto.run:
            base += f"_run-{self.task_dto.run}"
        return self.data_dir / self.task_dto.subject / "eeg" / f"{base}_{ext}"

    def _load_json(self, name):
        path = self._get_file(name)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_tsv(self, name):
        path = self._get_file(name)
        if path.exists():
            return pd.read_csv(path, sep='\t')
        return None
