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
    from eegkit.utils.system import configure_logging
    configure_logging()

    subj_mod = importlib.import_module('eegkit.models.subject_model')
    # Updated paths: pipeline package is the canonical home for task model/loader/processor
    pipe_pkg = importlib.import_module('eegkit.models.pipeline')
    task_mod = importlib.import_module('eegkit.models.pipeline.task_model')
    loader_mod = importlib.import_module('eegkit.models.pipeline.task_loader')
    proc_mod = importlib.import_module('eegkit.models.pipeline.task_processor')

    importlib.reload(subj_mod)
    importlib.reload(task_mod)
    importlib.reload(loader_mod)
    importlib.reload(proc_mod)

    from eegkit.models.subject_model import EEGSubjectModel
    return EEGSubjectModel


def get_model():
    import importlib

    # Module name is singular: ai_service
    ai_mod = importlib.import_module('eegkit.services.ai_service')
    importlib.reload(ai_mod)

    return ai_mod
