def reload_classes():
    import importlib

    import eegkit.controller.eeg_controller
    import eegkit.views.gui

    importlib.reload(eegkit.controller.eeg_controller)
    importlib.reload(eegkit.views.gui)

    from eegkit.controller.eeg_controller import EEGController
    from eegkit.views.gui import EEGUI

    return EEGController, EEGUI

def reload_data_classes():
    import importlib

    import eegkit.models.subject_model
    import eegkit.models.task_model
    import eegkit.models.task_loader
    import eegkit.models.task_processor

    importlib.reload(eegkit.models.subject_model)
    importlib.reload(eegkit.models.task_model)
    importlib.reload(eegkit.models.task_loader)
    importlib.reload(eegkit.models.task_processor)

    from eegkit.models.subject_model import EEGSubjectModel
    return EEGSubjectModel
