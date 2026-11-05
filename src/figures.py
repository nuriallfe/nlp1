"""Helpers to keep the figures and tables of the experiments in files (for the blog and the slides)."""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt


@contextmanager
def saving_figure(path, dpi=200):
    """
    Inside the `with` block, every plt.show() first saves the current figure to `path` and then shows it.
    It lets us save the plots of functions that draw and show the figure themselves (e.g. analysis.plot_box_and_points).

        with saving_figure("images/my_plot.png"):
            analysis.plot_box_and_points(df)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    original_show = plt.show

    def show_and_save(*args, **kwargs):
        plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight")
        original_show(*args, **kwargs)

    plt.show = show_and_save
    try:
        yield
    finally:
        plt.show = original_show


def save_figure(fig, path, dpi=200):
    """Save a matplotlib figure (creating the folder if needed) and return it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def save_table(df, path):
    """Save a DataFrame as CSV (creating the folder if needed) and return it, so it can still be displayed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
