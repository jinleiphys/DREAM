"""Optional plotting for reduced-basis verification results."""

import numpy as np


def plot_rank_sweep(result, path):
    """Plot maximum chi2_emu versus synchronized time, with rank labels.

    The single panel compares forward and likelihood value-and-gradient costs.
    Zero errors are shown at the smallest positive floating-point number. The
    saved path selects the export format, including PDF, SVG, or PNG.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    style = {
        'font.family': 'sans-serif', 'font.size': 8, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.linewidth': 0.8, 'axes.grid': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.bottom': True, 'axes.spines.left': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    }
    with mpl.rc_context(style):
        fig = Figure(figsize=(3.5, 2.8), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        error = np.maximum(result['chi2_max'], np.finfo(float).tiny)
        for field, label, color, marker in (
            ('t_forward', 'Forward', '#6BAA7A', 'o'),
            ('t_grad', 'Value and gradient', '#E58CA3', 's'),
        ):
            times = np.asarray(result[field]) * 1000
            ax.loglog(times, error, color=color, marker=marker, ms=4, lw=1.1, label=label)
        for index, (rank, time, value) in enumerate(zip(result['rank'], result['t_forward'], error)):
            ax.annotate(str(rank), (time * 1000, value), xytext=(4, 6 + 12 * (index % 2)),
                        textcoords='offset points', fontsize=7)
        ax.axhline(1, color='#606060', lw=0.8, linestyle=':')
        ax.set(xlabel='Time per evaluation (ms)', ylabel=r'Maximum $\chi^2_{\mathrm{emu}}$')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(False, which='both')
        ax.margins(x=0.15, y=0.15)
        fig.savefig(path, dpi=300)
        fig.clear()
