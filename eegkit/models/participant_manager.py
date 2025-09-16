from pathlib import Path
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
import pandas as pd
import numpy as np

from .dtos import SubjectFilterDTO


class ParticipantManager:
    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._release_dirs = self._discover_release_dirs()
        self._release_by_number = self._map_release_numbers()
        self._task_index = self._discover_tasks() 
        self._participants_df: Optional[pd.DataFrame] = None 

    # ---------- paths ----------
    @property
    def _combined_path(self) -> Path:
        return self._data_dir / "participants_combind.tsv"

    @property
    def subject_dirs(self):
        for rdir in self._release_dirs:
            for p in rdir.glob("sub-*"):
                if p.is_dir():
                    yield p

    # ---------- discovery ----------
    def _discover_release_dirs(self) -> List[Path]:
        return sorted([p for p in self._data_dir.glob("cmi_bids_R*") if p.is_dir()])

    def _map_release_numbers(self) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        rx = re.compile(r"cmi_bids_(R\d+)$", re.IGNORECASE)
        for rdir in self._release_dirs:
            m = rx.search(rdir.name)
            if m:
                out[m.group(1)] = rdir
        return out

    def _discover_tasks(self):
        task_map = defaultdict(list)
        pat = re.compile(r"(sub-(?P<subject>[^_]+))_task-(?P<task>[^_]+)(?:_run-(?P<run>\d+))?_eeg\.set$", re.IGNORECASE)
        for subj_dir in self.subject_dirs:
            eeg_dir = subj_dir / "eeg"
            if not eeg_dir.exists():
                continue
            for f in eeg_dir.glob("sub-*_task-*_eeg.set"):
                m = pat.match(f.name)
                if not m:
                    continue
                subj = subj_dir.name
                task, run = m.group("task"), m.group("run")
                pair = (task, run)
                if pair not in task_map[subj]:
                    task_map[subj].append(pair)
        return dict(task_map)

    # ---------- normalization ----------
    @staticmethod
    def _norm(s: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    @classmethod
    def _norm_bases(cls, col: str) -> Set[str]:
        n = cls._norm(col)
        out = {n}
        m = re.match(r"^(.*?)(?:_\d+)$", (col or "").lower())
        if m:
            out.add(cls._norm(m.group(1)))
        m2 = re.match(r"^(.*?)(\d+)$", n)
        if m2:
            out.add(cls._norm(m2.group(1)))
        return {x for x in out if x}

    def _subject_task_name_set(self, subject: str, release_dir: Path) -> Set[str]:
        eeg_dir = release_dir / subject / "eeg"
        if not eeg_dir.exists():
            return set()
        names: Set[str] = set()
        for p in eeg_dir.glob(f"{subject}_task-*_eeg.set"):
            m = re.search(r"_task-([^_]+)_eeg\.set$", p.name, re.IGNORECASE)
            if m:
                names.add(self._norm(m.group(1)))
        return names

    def _detect_task_columns(self, df: pd.DataFrame, release_dir: Path) -> List[str]:
        seen: Set[str] = set()
        for subj_dir in release_dir.glob("sub-*"):
            if subj_dir.is_dir():
                seen |= self._subject_task_name_set(subj_dir.name, release_dir)
        cols: List[str] = []
        for col in df.columns:
            ser = df[col]
            statusy = ser.dtype == object and ser.dropna().str.lower().any()
            name_match = any(b in seen for b in self._norm_bases(col))
            if statusy or name_match:
                cols.append(col)
        meta = {"participant_id","release_number","age","sex","ehq_total","commercial_use","full_pheno","p_factor","attention","internalizing","externalizing"}
        return [c for c in cols if c not in meta]

    # ---------- CCD metrics helpers ----------
    def _ccd_event_files(self, subject: str, release_dir: Path) -> List[Path]:
        eeg_dir = release_dir / subject / "eeg"
        if not eeg_dir.exists():
            return []
        pattern = f"{subject}_task-contrastChangeDetection*_events.tsv"
        return sorted(eeg_dir.glob(pattern))

    def _compute_ccd_metrics(self, subject: str, release_dir: Path) -> Tuple[Optional[float], Optional[float]]:
        files = self._ccd_event_files(subject, release_dir)
        if not files:
            return (None, None)
        total_targets = 0
        hits = 0
        rt_hits: List[float] = []

        for fpath in files:
            ev = pd.read_csv(fpath, sep="\t")
            df = ev.copy()

            df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
            df = df.dropna(subset=["onset"])  
            df = df.sort_values("onset").reset_index(drop=True)

            is_target = df["value"].isin(["left_target", "right_target"]).values
            is_press = df["value"].isin(["left_buttonPress", "right_buttonPress"]).values
            is_trial_start = df["value"].eq("contrastTrial_start").values

            idx_targets = np.where(is_target)[0]
            idx_trial_starts = np.where(is_trial_start)[0]

            for i, ti in enumerate(idx_targets):
                t_on = float(df.loc[ti, "onset"])
                # Find end of window: next trial start after this target (strictly after t_on)
                next_ts = idx_trial_starts[idx_trial_starts > ti]
                window_end = float(df.loc[next_ts[0], "onset"]) if next_ts.size > 0 else np.inf
                # First press after target before end of window
                cand_idx = np.where((np.arange(len(df)) > ti) & is_press & (df["onset"].values < window_end))[0]
                total_targets += 1
                if cand_idx.size == 0:
                    continue
                pi = cand_idx[0]
                p_on = float(df.loc[pi, "onset"])
                fb = str(df.loc[pi, "feedback"]) if pd.notna(df.loc[pi, "feedback"]) else None
                val_t = str(df.loc[ti, "value"])  # left_target/right_target
                val_p = str(df.loc[pi, "value"])  # left_buttonPress/right_buttonPress

                correct = False
                if fb in ("smiley_face",):
                    correct = True
                elif fb in ("sad_face", "non_target"):
                    correct = False
                else:
                    # Fallback to side matching if feedback missing
                    if (val_t.startswith("left") and val_p.startswith("left")) or (val_t.startswith("right") and val_p.startswith("right")):
                        correct = True

                if correct:
                    hits += 1
                    rt_hits.append(max(0.0, p_on - t_on))

        if total_targets == 0:
            return (None, None)
        acc = 100.0 * hits / float(total_targets)
        med_rt = float(np.median(rt_hits)) if rt_hits else None
        return (acc, med_rt)

    def _augment_ccd_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ccd_accuracy"] = np.nan
        df["ccd_response_time"] = np.nan

        for idx, row in df.iterrows():
            subj = str(row.get("participant_id"))
            
            disk_pairs = self._task_index.get(subj, [])
            if not any(t == "contrastChangeDetection" for (t, r) in disk_pairs):
                continue

            rdir = self._release_by_number.get(str(row.get('release_number')))

            acc, rt = self._compute_ccd_metrics(subj, rdir)
            
            if acc is not None:
                if not pd.notna(df.at[idx, "ccd_accuracy"]) or df.at[idx, "ccd_accuracy"] != acc:
                    df.at[idx, "ccd_accuracy"] = round(acc, 2)
            if rt is not None:
                if not pd.notna(df.at[idx, "ccd_response_time"]) or df.at[idx, "ccd_response_time"] != rt:
                    df.at[idx, "ccd_response_time"] = round(rt, 3)

        return df

    # ---------- combined TSV with validation ----------
    def _validate_participants_file(self, p_path: Path, release_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(p_path, sep="\t")
        status = {"available", "unavailable", "caution", "missing"}
        task_cols = []
        for col in df.columns:
            ser = df[col]
            if ser.dtype == object and ser.dropna().astype(str).str.lower().isin(status).any():
                task_cols.append(col)

        for idx, row in df.iterrows():
            pid = str(row["participant_id"])
            subj_dir = release_dir / pid
            exists = subj_dir.exists() and subj_dir.is_dir()
            subj_tasks = set(self._task_index.get(pid, [])) if exists else set()

            if not exists:
                for c in task_cols:
                    if str(row[c]).lower() == "available":
                        df.at[idx, c] = "missing"
                continue

            for c in task_cols:
                val = str(row[c]).lower()
                if val != "available":
                    continue

                base_match = re.match(r"^(.*?)(?:_(\d+))?$", c)
                if base_match:
                    base = base_match.group(1)
                    idx_str = base_match.group(2)
                else:
                    base, idx_str = c, None

                if idx_str is not None:
                    run = idx_str 
                    has_match = any(t == base and r == run for (t, r) in subj_tasks)
                else:
                    has_match = any(t == base for (t, r) in subj_tasks)

                if not has_match:
                    df.at[idx, c] = "missing"

        return df

    def _build_or_load_participants(self) -> pd.DataFrame:
        cp = self._combined_path
        print("Finding participants_combind.tsv on " + str(cp))
        if cp.exists():
            print("Found")
            df = pd.read_csv(cp, sep='\t')
            return df
        print("Not found")
        
        frames: List[pd.DataFrame] = []
        for rdir in self._release_dirs:
            p = rdir / 'participants.tsv'
            print("Loadding" + str(p))
            if p.exists():
                vdf = self._validate_participants_file(p, rdir)
                if not vdf.empty:
                    frames.append(vdf)
        if not frames:
            raise FileNotFoundError('No participants.tsv found in cmi_bids_R* directories')
        combined = pd.concat(frames, ignore_index=True)
        combined['participant_id'] = combined['participant_id'].astype(str)
        combined = self._augment_ccd_metrics(combined)
        combined.to_csv(cp, sep='\t', index=False)
        print("participants_combind saved at " + str(cp))
        return combined

    def _participants(self) -> pd.DataFrame:
        if self._participants_df is None:
            self._participants_df = self._build_or_load_participants()
        return self._participants_df

    # ---------- public API ----------
    def subject_data_dir(self, subject: str) -> Path:
        df = self._participants()
        rows = df[df['participant_id']== str(subject)]
        if not rows.empty and 'release_number' in rows.columns:
            rn = str(rows.iloc[0]['release_number'])
            rdir = self._release_by_number.get(rn)
            if rdir:
                return rdir
        for rdir in self._release_dirs:
            if (rdir / subject).exists():
                return rdir
        return self._release_dirs[0] if self._release_dirs else self._data_dir

    def list_subjects(self) -> List[str]:
        df = self._participants()
        return df['participant_id'].dropna().sort_values().tolist()

    def list_all_tasks(self):
        df = self._participants()
        status = {"available", "unavailable", "caution", "missing"}
        task_cols = []
        for col in df.columns:
            ser = df[col]
            if ser.dtype == object and ser.dropna().str.lower().isin(status).any():
                task_cols.append(col)
                
        merged = set()
        for col in task_cols:
            base = re.sub(r"_\d+$", "", col)
            merged.add(base)
        return sorted(merged)

    def list_tasks(self, subject: str):
        df = self._participants()
        rows = df[df['participant_id'] == str(subject)]
        if rows.empty:
            return []
        avail_bases: Set[str] = set()
        for col in rows.columns:
            val = rows.iloc[0].get(col)
            if isinstance(val, str) and val.lower() == 'available':
                avail_bases |= self._norm_bases(col)
        disk = self._task_index.get(subject, [])
        return sorted([(t, r) for (t, r) in disk if self._norm(t) in avail_bases])

    def filter_subjects_by_dto(self, dto: SubjectFilterDTO) -> List[str]:
        df = self._participants().copy()
        print(df['participant_id'].sort_values().tolist())
        task = dto.task
        if task in df.columns:
            df = df[df[task].str.lower() == 'available']
        else:
            prefix = f"{task}_"
            group_cols = [c for c in df.columns if c.startswith(prefix)]
            print(group_cols)
            if group_cols:
                mask = (
                    df[group_cols]
                    .astype('string')
                    .apply(lambda s: s.str.lower().eq('available'))
                    .any(axis=1)
                )
                df = df[mask]

        print(df['participant_id'].sort_values().tolist())
            
        for name in dto.__dataclass_fields__.keys():
            if name in ('task','subject','run','subject_limit'):
                continue
            val = getattr(dto, name, None)
            if val is None:
                continue
            if name.endswith('_range'):
                col = name[:-6]
                if col in df.columns and isinstance(val, (tuple, list)):
                    lo, hi = val
                    s = pd.to_numeric(df[col], errors='coerce')
                    if lo is not None:
                        df = df[s >= lo]
                    if hi is not None:
                        df = df[s <= hi]
            elif name in df.columns:
                if isinstance(val, (list, tuple)):
                    vals = [v for v in val if v is not None]
                    if vals:
                        df = df[df[name].isin(vals)]
                else:
                    df = df[df[name] == val]
        subjects = df['participant_id'].dropna().tolist()
        limit = getattr(dto, 'subject_limit', None)
        if isinstance(limit, int) and limit > 0:
            subjects = subjects[:limit]
        return sorted(subjects)
