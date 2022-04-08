from typing import Tuple, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt

# type annotation for convenience
NUMBER = Union[float, int]
PR_Type = Union[Tuple[NUMBER], List[NUMBER], np.ndarray]


def plot_auroc(
    title: str,
    data: List[Tuple[str, PR_Type, PR_Type, float]],
    xrange: Optional[Tuple[NUMBER, NUMBER]] = None,
    xscale: Optional[str] = "linear",
    yrange: Optional[Tuple[NUMBER, NUMBER]] = None,
    yscale: Optional[str] = "linear"
) -> None:
    """
    Plot multiple ROC curves and calculate corresponding AUCs.
    :param title: Title of the plot
    :param data: Tuple of (model name, FPR, TPR, AUC score) values, model name is used as a legend
    :param xrange: Range of x-axis
    :param xscale: Scale x-axis: "linear", "log"
    :param yrange: Range of y-axis
    :param yscale: Scale y-axis: "linear", "log"
    :return: None
    """
    if len(data) < 1:
        raise ValueError(f"data should contain at least one pair of (FPR, TPR), got {data}")

    colors = ["r", "g", "b", "y", "c", "m", "k"]
    for i, (name, fpr, tpr, auc) in enumerate(data):
        plt.plot(fpr, tpr, label=f"{name} (area={auc})", color=colors[i])

    plt.title(title)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if xrange is not None:
        plt.xlim(*xrange)
    if xscale is not None:
        plt.xscale(xscale)
    if yrange is not None:
        plt.ylim(*yrange)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # generate some random data to test
    x = np.linspace(0, 10)
    metrics = []
    metrics.append(("AlexNet", x, np.sin(x) + x + np.random.randn(50), 0.5))
    metrics.append(("VGG", x, np.sin(x) + 2 * x + np.random.randn(50), 0.6))
    metrics.append(("ResNet", x, np.sin(x) + 3 * x + np.random.randn(50), 0.7))
    metrics.append(("VIT", x, np.sin(x) + 4 + np.random.randn(50), 0.8))
    plot_auroc("ROC Curve on LFW Test Set", metrics)

