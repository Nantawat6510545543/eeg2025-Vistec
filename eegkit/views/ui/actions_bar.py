"""Actions bar for EEG GUI: mode, description, action, and view toggle."""
from __future__ import annotations

from typing import Dict

import ipywidgets as widgets


class ActionsBar:
    """Compact bar with mode selector, action selector, and view toggle."""

    def __init__(self, specs: Dict[str, dict], mode_info: Dict[str, str] | None = None) -> None:
        """Initialize widgets from available specs and optional mode descriptions."""
        self.specs = specs or {}
        self.mode_info = mode_info or {}

        self.mode_selector = widgets.ToggleButtons(options=list(self.specs.keys()), description="Mode:")
        self.mode_description = widgets.HTML(value="")
        self.action_selector = widgets.ToggleButtons(description="Action:")
        self.param_view_toggle = widgets.Checkbox(
            value=False,
            description="Show all groups",
            indent=False,
            layout=widgets.Layout(width="auto")
        )

        self.container = widgets.VBox([
            self.mode_selector,
            self.mode_description,
            self.action_selector,
            self.param_view_toggle,
        ])

    def update_actions(self) -> None:
        """Refresh available actions based on the selected mode."""
        group = self.mode_selector.value
        if not group:
            self.action_selector.options = []
            return
        self.action_selector.options = list(self.specs.get(group, {}).keys())
        if self.action_selector.options and self.action_selector.value not in self.action_selector.options:
            self.action_selector.value = self.action_selector.options[0]

    def update_mode_description(self) -> None:
        """Update the mode description helper text under the mode selector."""
        group = self.mode_selector.value
        desc = (self.mode_info or {}).get(group, "")
        if desc:
            self.mode_description.value = f"<div style='color:#666;font-size:12px;margin:4px 0 8px 0'>{desc}</div>"
        else:
            self.mode_description.value = ""

    def set_specs(self, specs: Dict[str, dict], mode_info: Dict[str, str] | None = None) -> None:
        """Replace the available specs and optionally mode descriptions, updating UI."""
        self.specs = specs or {}
        if mode_info is not None:
            self.mode_info = mode_info or {}
        self.mode_selector.options = list(self.specs.keys())
        self.update_actions()
        self.update_mode_description()
