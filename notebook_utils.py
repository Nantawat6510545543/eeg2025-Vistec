def reload_classes():
    import importlib

    ctrl_mod = importlib.import_module('eegkit.controller.eeg_controller')
    gui_mod = importlib.import_module('eegkit.views.gui')

    importlib.reload(ctrl_mod)
    importlib.reload(gui_mod)

    from eegkit.controller.eeg_controller import EEGController
    from eegkit.views.gui import EEGUI

    return EEGController, EEGUI


def reload_data_classes():
    import importlib

    # Configure logging early so subsequent reloads emit logs to notebook output
    try:
        from eegkit.utils.logging_utils import configure_logging
        configure_logging()
    except Exception:
        pass

    subj_mod = importlib.import_module('eegkit.models.subject_model')
    task_mod = importlib.import_module('eegkit.models.task_model')
    loader_mod = importlib.import_module('eegkit.models.task_loader')
    proc_mod = importlib.import_module('eegkit.models.task_processor')

    importlib.reload(subj_mod)
    importlib.reload(task_mod)
    importlib.reload(loader_mod)
    importlib.reload(proc_mod)

    from eegkit.models.subject_model import EEGSubjectModel
    return EEGSubjectModel


def get_model():
    import importlib

    ai_mod = importlib.import_module('eegkit.services.ai_services')
    importlib.reload(ai_mod)

    return ai_mod
