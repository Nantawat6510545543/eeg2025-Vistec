"""Low-level file loading helpers for raw EEG, metadata, events and channel tables."""

import json
import warnings
from pathlib import Path

import mne
import pandas as pd

from ..dtos import TaskDTO


class EEGTaskLoader:
    """Load raw data & associated tables for a given TaskDTO from disk."""

    def __init__(self, task_dto: TaskDTO, data_dir):
        """Bind DTO and root data directory for subsequent file lookups."""
        self.task_dto = task_dto
        self.data_dir = Path(data_dir)

    def load_raw(self):
        """Load EEGLAB .set into MNE Raw with montage and optional preload."""
        path = self.get_file("eeg.set")
        fdt_path = path.with_suffix('.fdt')
        preload = not fdt_path.exists()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*boundary.*data discontinuities.*")
            raw = mne.io.read_raw_eeglab(path, preload=preload, montage_units='cm')
        if 'Cz' in raw.ch_names:
            raw.drop_channels(['Cz'])
        raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), match_case=False)
        return raw

    def load_metadata(self):
        """Load task-level metadata JSON (returns {} if missing)."""
        return self._load_json("eeg.json")

    def load_events(self):
        """Load events TSV as DataFrame (or None)."""
        return self._load_tsv("events.tsv")

    def load_channels(self):
        """Load channels TSV as DataFrame (or None)."""
        return self._load_tsv("channels.tsv")

    def load_electrodes(self):
        """Load electrodes TSV as DataFrame (or None)."""
        return self._load_tsv("electrodes.tsv")

    def get_file(self, ext):
        """Return path for file with given extension under subject/eeg folder."""
        base = f"{self.task_dto.subject}_task-{self.task_dto.task}"
        if self.task_dto.run:
            base += f"_run-{self.task_dto.run}"
        p = self.data_dir / self.task_dto.subject / "eeg" / f"{base}_{ext}"
        return p

    def _load_json(self, name):
        path = self.get_file(name)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_tsv(self, name):
        path = self.get_file(name)
        if path.exists():
            return pd.read_csv(path, sep='\t')
        return None
