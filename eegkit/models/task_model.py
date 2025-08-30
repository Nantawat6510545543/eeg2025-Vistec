from .task_loader import EEGTaskLoader
from .task_processor import EEGTaskProcessor
from .dtos import TaskDTO, FilterParamsDTO, EpochParamsDTO, TableInfoDTO
import pandas as pd
from mne import concatenate_raws
from ..cache import LocalCache


class EEGTaskModel:
    def __init__(self, task_dto: TaskDTO, data_dir):
        self.task_dto = task_dto

        if task_dto.run and "All" in str(task_dto.run):
            try:
                n_runs = int(str(task_dto.run).split("-")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid run format: {task_dto.run}")

            raws, events_list = [], []
            for i in range(1, n_runs + 1):
                run_dto = TaskDTO(task_dto.subject, task_dto.task, str(i))
                loader = EEGTaskLoader(run_dto, data_dir)
                try:
                    raw = loader.load_raw()
                    raws.append(raw)
                    events_list.append(loader.load_events())
                except FileNotFoundError:
                    print(f"[Warning] Run {i} not found for {task_dto.subject} {task_dto.task}, skipping.")

            if not raws:
                raise FileNotFoundError(f"No runs found for {task_dto.subject} {task_dto.task}")

            self.raw = concatenate_raws(raws)
            self.events = pd.concat(events_list, ignore_index=True) if events_list else None
            self.channels = raws[0].info['ch_names']

        else:
            loader = EEGTaskLoader(task_dto, data_dir)
            self.raw = loader.load_raw()
            self.events = loader.load_events()
            self.channels = loader.load_channels()
            
        self.electrodes = loader.load_electrodes()
        self.metadata = loader.load_metadata()
        data_files = [loader.get_file("eeg.set")]

        self.cache = LocalCache(data_files=data_files, pipeline_ver="v1")

        self.processor = EEGTaskProcessor(self.raw, self.events, task_dto, self.cache)

    def get_filtered_raw(self, filter_params: FilterParamsDTO):
        return self.processor.get_filtered(filter_params)

    def get_epochs(self, epoch_params: EpochParamsDTO):
        return self.processor.get_epochs(epoch_params)
