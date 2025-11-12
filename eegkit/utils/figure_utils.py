from __future__ import annotations

import matplotlib.pyplot as plt
import textwrap


def finalize_figure(fig: plt.Figure, task_dto, stimulus=None, caption_line: str = None,
                    plot_name="EEG Plot", x=15, y=10, max_line_chars: int = 110,
                    title_y: float = 0.995, gap_under_title: float = 0.030,
                    subject_line_spacing: float = 0.022, caption_line_spacing: float = 0.016,
                    extra_header_pad: float = 0.05) -> plt.Figure:
    """Finalize figure with wrapped header text using fixed positioning.

    extra_header_pad: additional vertical space (figure fraction) inserted below the
        lowest header text (after subject+caption) before subplots start.
    """
    fig.set_size_inches(x, y)
    subject_line = f"{task_dto}" + (f" - {stimulus}" if stimulus else "")

    def _wrap(text: str) -> str:
        if not text:
            return ""
        if len(text) <= max_line_chars:
            return text
        return "\n".join(textwrap.wrap(text, width=max_line_chars))

    subject_wrapped = _wrap(subject_line)
    caption_wrapped = _wrap(caption_line)

    # Disable constrained layout adjustments that would move our absolute text
    if getattr(fig, 'get_constrained_layout', lambda: False)():
        fig.set_constrained_layout(False)
    # Disable new layout engine (Matplotlib >=3.8) to allow subplots_adjust
    if hasattr(fig, 'get_layout_engine') and hasattr(fig, 'set_layout_engine'):
        if fig.get_layout_engine() is not None:
            fig.set_layout_engine(None)

    # Title fixed at title_y
    fig.text(0.5, title_y, plot_name.title(), ha='center', va='top', fontsize=18, fontweight='bold')

    n_subject_lines = subject_wrapped.count('\n') + 1 if subject_wrapped else 0
    n_caption_lines = caption_wrapped.count('\n') + 1 if caption_wrapped else 0

    subject_top_y = title_y - gap_under_title
    if subject_wrapped:
        fig.text(0.5, subject_top_y, subject_wrapped, ha='center', va='top', fontsize=14)

    # Inter-block gap between subject block and caption block
    inter_block_gap = 0.006

    if caption_wrapped:
        caption_top_y = subject_top_y - (n_subject_lines * subject_line_spacing) - inter_block_gap
        fig.text(0.5, caption_top_y, caption_wrapped, ha='center', va='top', fontsize=11)
    else:
        caption_top_y = subject_top_y - (n_subject_lines * subject_line_spacing)

    # Compute bottom of header region to reserve space for axes
    if subject_wrapped:
        subject_block_bottom = subject_top_y - (n_subject_lines * subject_line_spacing) + subject_line_spacing
    else:
        subject_block_bottom = subject_top_y
    if caption_wrapped:
        caption_block_bottom = caption_top_y - (n_caption_lines * caption_line_spacing) + caption_line_spacing
    else:
        caption_block_bottom = subject_block_bottom

    if subject_wrapped or caption_wrapped:
        # Take the lower (smaller y) bottom to ensure all header text is above axes
        bottom_header_y = min(subject_block_bottom, caption_block_bottom)
    else:
        bottom_header_y = title_y - gap_under_title

    # Increase whitespace below header
    buffer = 0.01 + extra_header_pad
    new_top = max(0.30, bottom_header_y - buffer)

    # Try subplots_adjust; if incompatible, manually shift axes
    applied = False
    try:
        fig.subplots_adjust(top=new_top)
        applied = True
    except Exception:
        applied = False
    if not applied:
        # Manual adjustment: scale each axes position so that its top <= new_top
        for ax in fig.axes:
            pos = ax.get_position()
            # Only adjust if current top is above new_top
            current_top = pos.y1
            if current_top > new_top:
                # Keep bottom the same, shrink height
                new_height = max(0.01, new_top - pos.y0)
                ax.set_position([pos.x0, pos.y0, pos.width, new_height])

    return fig
