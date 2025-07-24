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

    def _filter_params(self, plot_type, kwargs):
        spec = self.plot_specs.get(plot_type, {})
        param_defs = {**self.default_params, **spec.get("params", {})}
        return {k: kwargs.get(k, v.get("default")) for k, v in param_defs.items()}

    def _get_raw(self, subject, task, run, l_freq, h_freq):
        task_data = self.data.get_task(subject, task, run)
        return task_data.get_filtered_raw(l_freq, h_freq)

    def _get_epochs(self, subject, task, run, l_freq, h_freq):
        task_data = self.data.get_task(subject, task, run)
        return task_data.get_epochs(l_freq, h_freq)

    def plot_sensors(self, subject, task, run=None, **kwargs):
        params = self._filter_params("sensors", kwargs)
        raw = self._get_raw(subject, task, run, params["l_freq"], params["h_freq"])
        raw.plot_sensors(show_names=True)

    def plot_time(self, subject, task, run=None, **kwargs):
        params = self._filter_params("time", kwargs)
        raw = self._get_raw(subject, task, run, params["l_freq"], params["h_freq"])
        
        fig = raw.plot(
            duration=params["duration"],
            start=params["start"],
            n_channels=params["n_channels"],
            scalings='auto',
            show=False,
            block=True
        )

        self._finalize_figure(
            fig, subject, task, run,
            caption=params,
            plot_name="Time Domain"
        )

    def plot_frequency(self, subject, task, run=None, **kwargs):
        params = self._filter_params("frequency", kwargs)
        raw = self._get_raw(subject, task, run, params["l_freq"], params["h_freq"])

        psd = raw.compute_psd(fmin=params["fmin"], fmax=params["fmax"])
        fig = psd.plot(
            average=params["average"],
            spatial_colors=params["spatial_colors"],
            dB=params["dB"],
            show=False
        )

        self._finalize_figure(
            fig, subject, task, run,
            caption=params,
            plot_name="Frequency Domain"
        )


    def plot_conditionwise_psd(self, subject, task, run=None, **kwargs):
        params = self._filter_params("conditionwise_psd", kwargs)
        epochs, labels = self._get_epochs(subject, task, run, params["l_freq"], params["h_freq"])

        if epochs is None:
            print(f"No epochs available for {subject} - {task}" + (f" (Run {run})" if run else ""))
            return

        for condition in epochs.event_id:
            condition_epochs = epochs[condition]
            if len(condition_epochs) == 0:
                print(f"Skipping condition '{condition}' — no valid epochs.")
                continue

            cropped = self._validate_and_crop(condition_epochs, params["tmin"], params["tmax"])
            if cropped is None:
                print(f"Skipping {condition} — Invalid crop range: tmin={params['tmin']}, tmax={params['tmax']}")
                continue

            psd = cropped.compute_psd(fmin=params["fmin"], fmax=params["fmax"])
            fig = psd.plot(average=params["average"], spatial_colors=True, dB=params["dB"], show=False)

            self._finalize_figure(
                fig, subject, task, run, condition,
                caption=params,
                plot_name="Condition-wise PSD"
            )

    def plot_epochs_or_evoked(self, subject, task, run=None, mode='epochs', **kwargs):
        params = self._filter_params(mode, kwargs)
        epochs, labels = self._get_epochs(subject, task, run, params["l_freq"], params["h_freq"])

        if labels is not None:
            self.plot_specs["epochs"]["params"]["stimulus"]["default"] = [None] + sorted(labels)
            self.plot_specs["evoked"]["params"]["stimulus"]["default"] = [None] + sorted(labels)

        if epochs is None:
            print(f"No epochs available for {subject} - {task}" + (f" (Run {run})" if run else ""))
            return

        if params["stimulus"]:
            if params["stimulus"] not in epochs.event_id:
                print(f"Stimulus '{params['stimulus']}' not found in event_id.")
                return
            epochs = epochs[params["stimulus"]]

        cropped = self._validate_and_crop(epochs, params["tmin"], params["tmax"])
        if cropped is None:
            print(f"Invalid crop window: tmin={params['tmin']}, tmax={params['tmax']}")
            return

        if mode == 'evoked':
            fig = cropped.average().plot(show=False)
        else:
            fig = cropped.plot(events=False, n_channels=params["n_channels"], show=False)

        self._finalize_figure(
            fig, subject, task, run, params["stimulus"],
            caption=params,
            plot_name=mode
        )