import pandas as pd
from ..models import (
    BaseTaskDTO, FilterParamsDTO, EpochParamsDTO, TableInfoDTO
)
from .base_service import BaseService
from ..utils.cleaning_utils import EEGCleaner

data_registry = {}


def register_data(name, dto_cls):
    def decorator(func):
        data_registry[name] = {
            "params": dto_cls,
            "function": func
        }
        return func

    return decorator


class EEGDataService(BaseService):
    description = "Provides structured tables from the current selection for quick inspection and lightweight export (annotations, channels/electrodes, metadata, epoch summaries)."
    def __init__(self, get_raw_func, get_epochs_func, get_task_func):
        super().__init__(
            registry=data_registry,
            get_raw_func=get_raw_func,
            get_epochs_func=get_epochs_func,
            get_task_func=get_task_func,
        )

    @register_data("EEG Table", TableInfoDTO)
    def show_table(self, task_dto: BaseTaskDTO, table_info: TableInfoDTO):
        task_model = self.get_task(task_dto)
        df_map = {
            'events': task_model.get_event(),
            'channels': task_model.channels,
            'electrodes': task_model.electrodes
        }
        return df_map.get(table_info.table_type, pd.DataFrame()).head(table_info.rows)

    @register_data("Epochs Table", EpochParamsDTO)
    def show_epochs_table(self, task_dto: BaseTaskDTO, params: EpochParamsDTO):
        epochs, labels = self.get_epochs(task_dto, params)
        if epochs is None:
            return None

        rows = []
        for label, _code in epochs.event_id.items():
            try:
                cond_epochs = epochs[label]
            except Exception:
                continue
            if len(cond_epochs) == 0:
                continue
            n_times = len(cond_epochs.times)
            sfreq = float(cond_epochs.info.get('sfreq', 0.0))
            row = {
                'label': label,
                'n_epochs': len(cond_epochs),
                'n_channels': len(cond_epochs.ch_names),
                'timespan_sec': float(cond_epochs.times[-1] - cond_epochs.times[0]) if n_times > 1 else 0.0,
                'sampling_rate': sfreq,
                'duration_per_epoch_sec': float(n_times / sfreq) if sfreq > 0 and n_times > 0 else 0.0,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    @register_data("Annotations", FilterParamsDTO)
    def get_annotation_df(self, task_dto: BaseTaskDTO, filter_params: FilterParamsDTO):
        raw = self.get_raw(task_dto, filter_params)
        annots = raw.annotations
        df = pd.DataFrame({
            "onset": annots.onset,
            "duration": annots.duration,
            "description": annots.description
        })
        return df

    @register_data("Metadata", None)
    def show_annotations(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
        task_model = self.get_task(task_dto)
        return task_model.metadata

    @register_data("Artifact Sources", FilterParamsDTO)
    def artifact_sources(self, task_dto: BaseTaskDTO, params: FilterParamsDTO):
        """Identify artifact sources by attributing bad channels/windows to cleaning steps.

        Strategy:
        - Get prefiltered Raw by forcing bad_channel_policy='skip'.
        - Run each marking step on a copy and record new bad channels per step.
          Steps: flatline, hf_noise, lowcorr.
        - Run window-based steps to report time windows:
          - power-based windows ('bad_power').
          - ASR windows by running ASR in remove_only mode ('bad_asr') if enabled.

        Returns a DataFrame with rows for channels and windows.
        """
        # 1) Obtain prefiltered raw (no cleaning) by using policy='skip'
        try:
            from dataclasses import replace
            params_skip = replace(params, bad_channel_policy=["skip"]) if hasattr(params, '__dict__') else params
        except Exception:
            params_skip = params

        raw_pref = self.get_raw(task_dto, params_skip)

        # Helper to ensure list of names
        def _listify(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        # 2) Channel-level attributions
        rows = []
        try:
            base_bads = set(raw_pref.info.get('bads', []) or [])

            # Flatline
            r1 = EEGCleaner._mark_bad_flatline_channels(raw_pref.copy(), params)
            bads1 = set(r1.info.get('bads', []) or [])
            added_flat = sorted(bads1 - base_bads)
            for ch in added_flat:
                rows.append({
                    'type': 'channel', 'source': 'flatline', 'channel': ch,
                    'onset': None, 'duration': None
                })

            # High-frequency noise
            r2 = EEGCleaner._mark_bad_highfreq_noise_channels(r1.copy(), params)
            bads2 = set(r2.info.get('bads', []) or [])
            added_hf = sorted(bads2 - bads1)
            for ch in added_hf:
                rows.append({
                    'type': 'channel', 'source': 'hf_noise', 'channel': ch,
                    'onset': None, 'duration': None
                })

            # Low-correlation
            r3 = EEGCleaner._mark_bad_lowcorr_channels(r2.copy(), params)
            bads3 = set(r3.info.get('bads', []) or [])
            added_corr = sorted(bads3 - bads2)
            for ch in added_corr:
                rows.append({
                    'type': 'channel', 'source': 'lowcorr', 'channel': ch,
                    'onset': None, 'duration': None
                })

            # 3) Window-level attributions: power
            rp = EEGCleaner._mark_bad_windows_by_power(raw_pref.copy(), params)
            try:
                ann = getattr(rp, 'annotations', None)
                if ann is not None and len(ann) > 0:
                    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
                        if str(desc) == 'bad_power':
                            rows.append({
                                'type': 'window', 'source': 'bad_power', 'channel': None,
                                'onset': float(onset), 'duration': float(dur)
                            })
            except Exception:
                pass

            # 4) Window-level attributions: ASR (remove_only probe)
            try:
                from dataclasses import replace as dc_replace
                # Only attempt if ASR threshold enabled
                asr_enabled = getattr(params, 'clean_asr_max_std', None)
                if asr_enabled is not None and float(asr_enabled) > 0:
                    p_asr = dc_replace(params, clean_asr_remove_only=True)
                    ra = EEGCleaner._apply_asr(raw_pref.copy(), p_asr)
                    ann = getattr(ra, 'annotations', None)
                    if ann is not None and len(ann) > 0:
                        for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
                            if str(desc) == 'bad_asr':
                                rows.append({
                                    'type': 'window', 'source': 'bad_asr', 'channel': None,
                                    'onset': float(onset), 'duration': float(dur)
                                })
            except Exception:
                pass
        except Exception:
            # If anything goes wrong, return an empty DataFrame with expected columns
            cols = ['type', 'source', 'channel', 'onset', 'duration']
            return pd.DataFrame(columns=cols)

        # Build DataFrame
        df = pd.DataFrame(rows, columns=['type', 'source', 'channel', 'onset', 'duration'])
        # Order and sort for readability
        if not df.empty:
            source_order = {
                'flatline': 0, 'hf_noise': 1, 'lowcorr': 2, 'bad_power': 3, 'bad_asr': 4
            }
            df['sort_key'] = df['source'].map(source_order).fillna(99)
            df = df.sort_values(['type', 'sort_key', 'channel', 'onset'], na_position='last').drop(columns=['sort_key'])
        return df
