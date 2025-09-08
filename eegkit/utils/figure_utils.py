import matplotlib.pyplot as plt
from ..models import BaseTaskDTO


def finalize_figure(fig: plt.Figure, task_dto: BaseTaskDTO, stimulus=None, caption: dict=None,
                    plot_name="EEG Plot", x=15, y=10) -> plt.Figure:
    fig.set_size_inches(x, y)

    subject_line = f"{task_dto}" + (f" - {stimulus}" if stimulus else "")
    caption_line = ", ".join(
        f"{k} = {v:.1f}" if isinstance(v, (float, int)) else f"{k} = {v}"
        for k, v in (caption or {}).items()
    )

    # Use suptitle with constrained layout
    fig.suptitle(plot_name.title(), fontsize=18, fontweight='bold')
    fig.text(0.5, 0.94, subject_line, ha='center', fontsize=14)
    if caption_line:
        fig.text(0.5, 0.92, caption_line, ha='center', fontsize=11)

    # Optional: fine-tune spacing while staying compatible
    if hasattr(fig, "set_constrained_layout_pads"):
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    return fig

