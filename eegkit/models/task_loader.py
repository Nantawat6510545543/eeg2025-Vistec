from pathlib import Path
import json
import pandas as pd
import mne
import warnings
import time
import logging
from .dtos import TaskDTO


class EEGTaskLoader:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto
        self.data_dir = Path(data_dir)
        self._log = logging.getLogger(__name__)

    def load_raw(self):
        path = self.get_file("eeg.set")
        fdt_path = path.with_suffix('.fdt')
        preload = not fdt_path.exists()
        self._log.info("Loading raw EEGLAB: %s (preload=%s)", path, preload)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*boundary.*data discontinuities.*")
            raw = mne.io.read_raw_eeglab(path, preload=preload, montage_units='cm')
        t_read = time.perf_counter() - t0
        self._log.info("EEGLAB read completed in %.2fs (n_ch=%d, sfreq=%.2f)", t_read, len(raw.ch_names), raw.info.get('sfreq', 0))
        if 'Cz' in raw.ch_names:
            raw.drop_channels(['Cz'])
        t_mont0 = time.perf_counter()
        raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-128"), match_case=False)
        self._log.info("Montage applied in %.2fs", time.perf_counter() - t_mont0)
        return raw

    def load_metadata(self):
        t0 = time.perf_counter()
        meta = self._load_json("eeg.json")
        self._log.debug("Loaded metadata in %.2fs (keys=%d)", time.perf_counter() - t0, len(meta) if isinstance(meta, dict) else 0)
        return meta

    def load_events(self):
        t0 = time.perf_counter()
        df = self._load_tsv("events.tsv")
        n = 0 if df is None else len(df)
        self._log.debug("Loaded events in %.2fs (rows=%d)", time.perf_counter() - t0, n)
        return df

    def load_channels(self):
        t0 = time.perf_counter()
        df = self._load_tsv("channels.tsv")
        n = 0 if df is None else len(df)
        self._log.debug("Loaded channels in %.2fs (rows=%d)", time.perf_counter() - t0, n)
        return df

    def load_electrodes(self):
        t0 = time.perf_counter()
        df = self._load_tsv("electrodes.tsv")
        n = 0 if df is None else len(df)
        self._log.debug("Loaded electrodes in %.2fs (rows=%d)", time.perf_counter() - t0, n)
        return df

    def get_file(self, ext):
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
