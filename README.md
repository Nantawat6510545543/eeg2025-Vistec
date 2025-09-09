# EEG Workbench

An interactive EEG exploration toolkit built on MNE-Python with Jupyter-based UI.

Main components:
- `eegkit/` package with models, services, and ipywidgets GUI
- Caching in `.eegcache/` for filtered raw and epochs
- Notebooks: `eeg_workbench.ipynb`, `eeg_debuger.ipynb`, etc.

Quickstart (in a notebook):
1. from notebook_utils import reload_classes; EEGController, EEGUI = reload_classes()
2. from eegkit.models.subject_model import EEGSubjectModel
3. sm = EEGSubjectModel(data_dir="/path/to/BIDS")
4. ctrl = EEGController(sm)
5. ui = EEGUI(ctrl); ui.show()

Notes:
- Use conda to create env from `requirements.txt` (explicit conda spec).
- Data layout: BIDS-like, expecting `sub-*/eeg/sub-*_task-*_eeg.set` plus tsv/json sidecars.
- Cache stored under `.eegcache/` at repo root.
