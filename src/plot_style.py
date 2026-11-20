"""
Shared chart styling, so every figure in the report looks the same.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
# one fixed colour per feature group, used wherever groups appear
GROUP_COLORS = {"PE header": "#2a78d6", "PE sections": "#eb6834",
                "Imported DLLs": "#1baf7a", "API calls": "#eda100"}
# one fixed colour per model, used in the comparison figures
MODEL_COLORS = {"Random Forest": "#2a78d6", "SVM": "#eb6834", "Gradient Boosting": "#1baf7a"}
# light-to-dark single-hue ramp for heatmaps
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                  "#256abf", "#184f95", "#0d366b"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 1,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def value_grid(ax, axis="x"):
    """Hairline gridlines behind the marks, on the value axis only."""
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.spines["left" if axis == "x" else "bottom"].set_visible(True)
    ax.tick_params(length=0)
