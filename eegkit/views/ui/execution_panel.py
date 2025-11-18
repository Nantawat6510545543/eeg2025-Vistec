"""Execution panel: run buttons and output area for results."""
from __future__ import annotations

import json

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


class ExecutionPanel:
    """Run buttons and a shared output area for controller results."""

    def __init__(self) -> None:
        """Create buttons and an output container."""
        self.tmux_button = widgets.Button(description="Run on Tmux", button_style="success")
        self.inline_button = widgets.Button(description="Run Inline", button_style="success")
        self.output = widgets.Output()
        self.container = widgets.VBox([self.tmux_button, self.inline_button, self.output])

    def render_result(self, result) -> None:
        """Render a result object into the output area.

        Does not clear existing content; callers can clear/print a header first.
        Supports DataFrame, matplotlib Figure, list[Figure], dict/list (JSON), str, or generic fallback.
        """
        with self.output:
            if isinstance(result, pd.DataFrame):
                display(result)
            elif isinstance(result, plt.Figure):
                display(result)
            elif isinstance(result, list) and all(isinstance(fig, plt.Figure) for fig in result):
                for fig in result:
                    display(fig)
            elif isinstance(result, (dict, list)):
                print(json.dumps(result, indent=2))
            elif isinstance(result, str):
                print(result)
            elif result is not None:
                print("Output:", result)
