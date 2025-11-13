"""Simple progress area for long UI operations."""
from __future__ import annotations

import ipywidgets as widgets
from IPython.display import display


class ProgressArea:
    def __init__(self) -> None:
        self.label = widgets.HTML(value="")
        self.progress = widgets.IntProgress(value=0, min=0, max=1,
                                            description="0/0",
                                            layout=widgets.Layout(width="100%", height="26px"))
        self.text = widgets.HTML(value="")
        self.container = widgets.VBox([self.label, self.progress, self.text],
                                      layout=widgets.Layout(width="100%"))

    def begin(self, total: int, title: str = "Working…") -> None:
        self.label.value = f"<b>{title}</b>"
        self.progress.max = max(1, int(total))
        self.progress.value = 0
        self.progress.bar_style = ""
        self.progress.description = f"0/{self.progress.max}"
        self.text.value = ""
        display(self.container)

    def update(self, current: int, total: int | None = None, group: str | None = None, key: str | None = None) -> None:
        if total is not None:
            self.progress.max = max(self.progress.max, int(total))
        self.progress.value = int(current)
        self.progress.description = f"{self.progress.value}/{self.progress.max}"
        if group and key:
            self.text.value = f"Building <code>{group}</code> → <code>{key}</code> ({self.progress.description})"

    def finish(self, summary: str = "Ready.") -> None:
        self.progress.bar_style = "success"
        self.progress.description = f"{self.progress.value}/{self.progress.max}"
        self.text.value = summary
