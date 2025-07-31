
import matplotlib.pyplot as plt
import json
import pathlib

class EEGVisualization:
    def __init__(self, get_raw_func, get_epochs_func, json_path):
        self.get_raw = get_raw_func
        self.get_epochs = get_epochs_func
        self.json_path = json_path
        self.default_kwargs = {}
        self.plot_specs = {}

        self._load_plot_specs()

    def _load_plot_specs(self):

        path = pathlib.Path(self.json_path)
        with open(path) as f:
            spec_file = json.load(f)

        self.default_kwargs = spec_file.get("default_kwargs", {})
        self.plot_specs = spec_file.get("plot_specs", {})

        for key, spec in self.plot_specs.items():
            func_name = f"plot_{key}"
            if hasattr(self, func_name):
                self.plot_specs[key]["function"] = getattr(self, func_name)

    def _validate_and_crop(self, epochs, tmin, tmax):
        tmin_valid = max(epochs.tmin, tmin)
        tmax_valid = min(epochs.tmax, tmax)
        if tmin_valid >= tmax_valid:
            return None, (tmin_valid, tmax_valid)
        return epochs.copy().crop(tmin=tmin_valid, tmax=tmax_valid), (tmin_valid, tmax_valid)

    def _finalize_figure(self, fig, subject, task, run=None, stimulus=None, caption: dict = None, plot_name="EEG Plot", x=15, y=10):
        if not isinstance(fig, plt.Figure):
            return
        fig.set_size_inches(x, y)
        subject_line = f"{subject} - {task}" + (f" - {stimulus}" if stimulus else "") + (f" (Run {run})" if run else "")
        caption_line = ", ".join(f"{k} = {v:.1f}" if isinstance(v, (float, int)) else f"{k} = {v}" for k, v in caption.items()) if caption else ""
        fig.text(0.5, 0.96, plot_name.title(), ha='center', fontsize=18, weight='bold')
        fig.text(0.5, 0.94, subject_line, ha='center', fontsize=14)
        if caption_line:
            fig.text(0.5, 0.92, caption_line, ha='center', fontsize=11)
        fig.subplots_adjust(top=0.90)
        plt.show()

    def plot_sensors(self, subject, task, run=None, **kwargs):
        raw = self.get_raw(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        raw.plot_sensors(show_names=True)

    def plot_time(self, subject, task, run=None, **kwargs):
        raw = self.get_raw(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        fig = raw.plot(duration=kwargs["duration"], start=kwargs["start"], n_channels=kwargs["n_channels"], scalings='auto', show=False, block=True)
        self._finalize_figure(fig, subject, task, run, caption=kwargs, plot_name="Time Domain")

    def plot_frequency(self, subject, task, run=None, **kwargs):
        raw = self.get_raw(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        psd = raw.compute_psd(fmin=kwargs["fmin"], fmax=kwargs["fmax"])
        fig = psd.plot(average=kwargs["average"], spatial_colors=kwargs["spatial_colors"], dB=kwargs["dB"], show=False)
        self._finalize_figure(fig, subject, task, run, caption=kwargs, plot_name="Frequency Domain")

    def plot_conditionwise_psd(self, subject, task, run=None, **kwargs):
        epochs, labels = self.get_epochs(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        if epochs is None:
            print(f"No epochs available for {subject} - {task}" + (f" (Run {run})" if run else ""))
            return
        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                continue
            cropped, crop_info = self._validate_and_crop(condition_epochs, kwargs["tmin"], kwargs["tmax"])
            if cropped is None:
                continue
            psd = cropped.compute_psd(fmin=kwargs["fmin"], fmax=kwargs["fmax"])
            fig = psd.plot(average=kwargs["average"], spatial_colors=True, dB=kwargs["dB"], show=False)
            self._finalize_figure(fig, subject, task, run, condition, caption=kwargs, plot_name="Condition-wise PSD")

    def plot_epochs(self, subject, task, run=None, **kwargs):
        epochs, labels = self.get_epochs(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        if labels is not None:
            self.plot_specs["epochs"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
        if epochs is None:
            return
        if kwargs["stimulus"]:
            if kwargs["stimulus"] not in epochs.event_id:
                return
            epochs = epochs[kwargs["stimulus"]]
        cropped, crop_info = self._validate_and_crop(epochs, kwargs["tmin"], kwargs["tmax"])
        if cropped is None:
            return
        fig = cropped.plot(events=False, n_channels=kwargs["n_channels"], show=False)
        self._finalize_figure(fig, subject, task, run, kwargs["stimulus"], caption=kwargs, plot_name="Epochs")

    def plot_evoked(self, subject, task, run=None, **kwargs):
        epochs, labels = self.get_epochs(subject, task, run, kwargs["l_freq"], kwargs["h_freq"])
        if labels is not None:
            self.plot_specs["epochs"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
        if epochs is None:
            return
        if kwargs["stimulus"]:
            if kwargs["stimulus"] not in epochs.event_id:
                return
            epochs = epochs[kwargs["stimulus"]]
        cropped, crop_info = self._validate_and_crop(epochs, kwargs["tmin"], kwargs["tmax"])
        if cropped is None:
            return
        fig = cropped.average().plot_joint(show=False)
        self._finalize_figure(fig, subject, task, run, kwargs["stimulus"], caption=kwargs, plot_name="Evoked")
