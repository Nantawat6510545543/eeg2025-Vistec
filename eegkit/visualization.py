from .subject import EEGSubjectData
import matplotlib.pyplot as plt

class EEGVisualization:
    def __init__(self, subject_data: EEGSubjectData):
        self.data = subject_data
        self.default_params = {
            "l_freq": {"type": "float", "default": 1.0},
            "h_freq": {"type": "float", "default": 50.0},
        }

        self.plot_specs = self._build_plot_specs()

    def _build_plot_specs(self):
        return {
            "sensors": {
                "function": self.plot_sensors,
                "label": "Sensor Layout",
                "params": {}
            },
            "time": {
                "function": self.plot_time,
                "label": "Time Domain Plot",
                "params": {
                    "duration": {"type": "float", "default": 10.0},
                    "start": {"type": "float", "default": 0.0},
                    "n_channels": {"type": "int", "default": 10}
                },
            },
            "frequency": {
                "function": self.plot_frequency,
                "label": "Frequency Domain",
                "params": {
                    "fmin": {"type": "float", "default": 1.0},
                    "fmax": {"type": "float", "default": 60.0},
                    "average": {"type": "bool", "default": True},
                    "dB": {"type": "bool", "default": True},
                    "spatial_colors": {"type": "bool", "default": False}
                },
            },
            "conditionwise psd": {
                "function": self.plot_conditionwise_psd,
                "label": "Condition-wise PSD",
                "params": {
                    "fmin": {"type": "float", "default": 1.0},
                    "fmax": {"type": "float", "default": 50.0},
                    "tmin": {"type": "float", "default": 0.0},
                    "tmax": {"type": "float", "default": 2.0},
                    "average": {"type": "bool", "default": True},
                    "dB": {"type": "bool", "default": True}
                },
            },
            "epochs": {
                "function": lambda s, t, r=None, **k: self.plot_epochs_or_evoked(s, t, r, mode="epochs", **k),
                "label": "Epoch Plot",
                "params": {
                    "tmin": {"type": "float", "default": 0.0},
                    "tmax": {"type": "float", "default": 2.0},
                    "stimulus": {"type": "dropdown", "default": []},
                    "n_channels": {"type": "int", "default": 10},
                },
            },
            "evoked": {
                "function": lambda s, t, r=None, **k: self.plot_epochs_or_evoked(s, t, r, mode="evoked", **k),
                "label": "Evoked Response",
                "params": {
                    "tmin": {"type": "float", "default": 0.0},
                    "tmax": {"type": "float", "default": 2.0},
                    "stimulus": {"type": "dropdown", "default": []},
                },
            },
        }

    def _validate_and_crop(self, epochs, tmin, tmax):
        start = epochs.tmin
        end = epochs.tmax

        tmin_valid = max(start, tmin) if tmin is not None else start
        tmax_valid = min(end, tmax) if tmax is not None else end

        if tmin_valid >= tmax_valid:
            return None

        cropped = epochs.copy().crop(tmin=tmin_valid, tmax=tmax_valid)
        return cropped

    def _finalize_figure(self, fig, subject, task, run=None, stimulus=None, caption: dict = None, plot_name="EEG Plot"):
        if not isinstance(fig, plt.Figure):
            return

        fig.set_size_inches(15, 12)

        subject_line = f"{subject} - {task}" + (f" - {stimulus}" if stimulus else "") + (f" (Run {run})" if run else "")

        if caption:
            caption_line = ", ".join(f"{k} = {v:.1f}" if isinstance(v, (float, int)) else f"{k} = {v}" for k, v in caption.items())
        else:
            caption_line = ""

        fig.text(0.5, 0.96, plot_name.title(), ha='center', fontsize=18, weight='bold')
        fig.text(0.5, 0.94, subject_line, ha='center', fontsize=14)
        if caption_line:
            fig.text(0.5, 0.92, caption_line, ha='center', fontsize=11)

        fig.subplots_adjust(top=0.90)
        plt.show()

    def plot_sensors(self, subject, task, run=None, **kwargs):
        l_freq = kwargs.get("l_freq", 1)
        h_freq = kwargs.get("h_freq", 50)

        task_data = self.data.get_task(subject, task, run)
        raw = task_data.get_filtered_raw(l_freq, h_freq)
        raw.plot_sensors(show_names=True)

    def plot_time(self, subject, task, run=None, **kwargs):
        l_freq = kwargs.get("l_freq", 1)
        h_freq = kwargs.get("h_freq", 50)
        duration = kwargs.get('duration', 10.0)
        start = kwargs.get('start', 0.0)
        n_channels = kwargs.get('n_channels', 10)

        task_data = self.data.get_task(subject, task, run)
        raw = task_data.get_filtered_raw(l_freq, h_freq)

        fig = raw.plot(
            duration=duration,
            start=start,
            n_channels=n_channels,
            scalings='auto',
            show=False,
            block=True
        )

        caption_dict = {"start": start, "duration": duration}
        self._finalize_figure(fig, subject, task, run, caption=caption_dict, plot_name="Time Domain")

    def plot_frequency(self, subject, task, run=None, **kwargs):
        l_freq = kwargs.get("l_freq", 1)
        h_freq = kwargs.get("h_freq", 50)
        fmin = kwargs.get("fmin", 1)
        fmax = kwargs.get("fmax", 60)
        average = kwargs.get("average", True)
        dB = kwargs.get("dB", True)
        spatial_colors = kwargs.get("spatial_colors", False)

        task_data = self.data.get_task(subject, task, run)
        raw = task_data.get_filtered_raw(l_freq, h_freq)

        psd = raw.compute_psd(fmin=fmin, fmax=fmax)
        fig = psd.plot(
            average=average,
            spatial_colors=spatial_colors,
            dB=dB,
            show=False
        )

        caption_dict = {"l_freq": l_freq, "h_freq": h_freq,"fmin": fmin, "fmax": fmax}
        self._finalize_figure(fig, subject, task, run, caption=caption_dict, plot_name="Frequency Domain")

    def plot_conditionwise_psd(self, subject, task, run=None, **kwargs):
        fmin = kwargs.get("fmin", 1)
        fmax = kwargs.get("fmax", 50)
        tmin = kwargs.get("tmin", None)
        tmax = kwargs.get("tmax", None)
        average = kwargs.get("average", True)
        dB = kwargs.get("dB", True)
        l_freq = kwargs.get("l_freq", 1)
        h_freq = kwargs.get("h_freq", 50)

        task_data = self.data.get_task(subject, task, run)
        epochs, labels = task_data.get_epochs(l_freq=l_freq, h_freq=h_freq)

        if epochs is None:
            print(f"No epochs available for {subject} - {task}" + (f" (Run {run})" if run else ""))
            return

        for condition_name in epochs.event_id:
            condition_epochs = epochs[condition_name]
            if len(condition_epochs) == 0:
                print(f"Skipping condition '{condition_name}' — no valid epochs.")
                continue
            cropped_epochs = self._validate_and_crop(condition_epochs, tmin, tmax)

            if cropped_epochs is None:
                print(f"Skipping {condition_name} — Invalid crop range: tmin={tmin}, tmax={tmax}")
                continue

            psd = cropped_epochs.compute_psd(fmin=fmin, fmax=fmax)
            fig = psd.plot(spatial_colors=True, average=average, dB=dB, show=False)

            caption_dict = {"l_freq": l_freq, "h_freq": h_freq,"tmin": tmin, "tmax": tmax}
            self._finalize_figure(fig, subject, task, run, condition_name, caption=caption_dict, plot_name="Condition-wise PSD")

    def plot_epochs_or_evoked(self, subject, task, run=None, mode='epochs', **kwargs):
        l_freq = kwargs.get("l_freq", 1)
        h_freq = kwargs.get("h_freq", 50)
        stimulus = kwargs.get("stimulus", None)
        n_channels = kwargs.get("n_channels", 20)
        tmin = kwargs.get("tmin", None)
        tmax = kwargs.get("tmax", None)

        task_data = self.data.get_task(subject, task, run)
        epochs, labels = task_data.get_epochs(l_freq=l_freq, h_freq=h_freq)
        self.plot_specs["epochs"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
        self.plot_specs["evoked"]["params"]["stimulus"]["default"] = [None] + sorted(labels)

        if epochs is None:
            print(f"No epochs available for {subject} - {task}" + (f" (Run {run})" if run else ""))
            return

        if stimulus:
            if stimulus not in epochs.event_id:
                print(f"Stimulus '{stimulus}' not found in event_id.")
                return
            epochs = epochs[stimulus]

        cropped_epochs = self._validate_and_crop(epochs, tmin, tmax)
        if cropped_epochs is None:
            print(f"Invalid crop window: tmin={tmin}, tmax={tmax}")
            return

        if mode == 'evoked':
            evoked = cropped_epochs.average()
            fig = evoked.plot(show=False)
        else:
            fig = cropped_epochs.plot(events=False, n_channels=n_channels, show=False)

        caption_dict = {"l_freq": l_freq, "h_freq": h_freq, "tmin": tmin, "tmax": tmax}
        self._finalize_figure(fig, subject, task, run, stimulus, caption=caption_dict, plot_name=mode)
