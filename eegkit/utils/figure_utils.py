import matplotlib.pyplot as plt
from ..models import BaseTaskDTO


def finalize_figure(fig: plt.Figure, task_dto: BaseTaskDTO, stimulus=None, caption: dict = None, plot_name="EEG Plot", x=15,
                    y=10) -> plt.Figure:
    fig.set_size_inches(x, y)
    subject_line = f"{task_dto}" + (f" - {stimulus}" if stimulus else "")
    caption_line = ", ".join(f"{k} = {v:.1f}" if isinstance(v, (float, int)) else f"{k} = {v}" for k, v in
                             caption.items()) if caption else ""
    fig.text(0.5, 0.96, plot_name.title(), ha='center', fontsize=18, weight='bold')
    fig.text(0.5, 0.94, subject_line, ha='center', fontsize=14)
    if caption_line:
        fig.text(0.5, 0.92, caption_line, ha='center', fontsize=11)
    fig.subplots_adjust(top=0.90)
    return fig
