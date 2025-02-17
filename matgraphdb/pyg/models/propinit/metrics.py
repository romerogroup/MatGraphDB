import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_COLORS = ["#0A9F9D", "#CEB175", "#E54E21", "#6C8645", "#C18748"]
DEFAULT_COLORS.extend(
    ["#C52E19", "#AC9765", "#54D8B1", "#b67c3b", "#175149", "#AF4E24"]
)
DEFAULT_COLORS.extend(["#FBA72A", "#D3D4D8", "#CB7A5C", "#5785C1"])
DEFAULT_COLORS.extend(["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"])
DEFAULT_COLORS.extend(["#ECCBAE", "#046C9A", "#D69C4E", "#ABDDDE", "#000000"])


class ROCCurve:
    def __init__(self, colors=DEFAULT_COLORS):
        self.curves = {}
        self.fig, self.ax = plt.subplots(
            figsize=(8, 6)
        )  # Use subplot for better control
        self.colors = colors  # Custom colors list

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel("False Positive Rate")
        self.ax.set_ylabel("True Positive Rate")
        self.ax.set_title("Receiver Operating Characteristic (ROC) Curve")

        self.line_styles = {
            "main": "-",  # Solid line for main model
            "baseline": "--",  # Dashed line for baselines
        }
        self.split_colors = {}  # Dictionary to track color assignments per split

    def add_curve(self, pred, target, label, split_label, is_baseline=False):
        """Adds a curve to the plot. Determines color and linestyle based on split and model type."""
        fpr, tpr, _ = roc_curve(target.cpu().detach().numpy(), pred.cpu().detach().numpy())
        auc_score = roc_auc_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy())

        # Assign consistent colors per split
        if split_label not in self.split_colors:
            self.split_colors[split_label] = self.colors[
                len(self.split_colors) % len(self.colors)
            ]

        color = self.split_colors[split_label]
        linestyle = (
            self.line_styles["baseline"] if is_baseline else self.line_styles["main"]
        )

        self.curves[label] = (fpr, tpr, auc_score, color, linestyle)

    def plot(self):
        for label, (fpr, tpr, auc_score, color, linestyle) in self.curves.items():
            self.ax.plot(
                fpr,
                tpr,
                color=color,
                linestyle=linestyle,
                lw=2,
                label=f"{label} (AUC = {auc_score:.2f})",
            )

        # Add diagonal reference line once
        self.ax.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Random Chance"
        )

        # Move legend outside the plot
        self.ax.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        self.fig.tight_layout()

    def save(self, save_path):
        self.fig.savefig(save_path, bbox_inches="tight")

    def close(self):
        plt.close(self.fig)


class LearningCurve:
    def __init__(self, colors=DEFAULT_COLORS):
        self.curves = {}
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.colors = colors  # Custom colors list

        self.ax.set_xlabel("Epoch")
        self.ax.set_title("Learning Curve")

        self.line_styles = {
            "main": "-",  # Solid line for main model
            "baseline": "--",  # Dashed line for baselines
        }
        self.split_colors = {}  # Dictionary to track color assignments per split

    def add_curve(self, epochs, values, label, split_label, is_baseline=False):
        """Adds a curve to the plot. Assigns consistent color per split and line style per model type."""
        if split_label not in self.split_colors:
            self.split_colors[split_label] = self.colors[
                len(self.split_colors) % len(self.colors)
            ]

        color = self.split_colors[split_label]
        linestyle = (
            self.line_styles["baseline"] if is_baseline else self.line_styles["main"]
        )

        self.curves[label] = (epochs, values, color, linestyle)

    def plot(self):
        for label, (epochs, values, color, linestyle) in self.curves.items():
            self.ax.plot(
                epochs, values, color=color, linestyle=linestyle, lw=2, label=label
            )

        self.ax.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        self.fig.tight_layout()
        
    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel)

    def save(self, save_path):
        self.fig.savefig(save_path, bbox_inches="tight")

    def close(self):
        plt.close(self.fig)


def plot_pca(
    embeddings,
    save_dir=None,
    n_components=1,
    n_nodes_per_type=None,
    node_labels_per_type=None,
    figsize=(10, 8),
    colors=DEFAULT_COLORS,
    close=True,
    save_name='embeddings_pca_grid.png'
):
    """
    Args:
        embeddings: Embeddings to visualize.
        save_path: Path where plots will be saved.
        pca_components_to_plot (list of tuples):
            List of (x_comp, y_comp) pairs, 0-based indexing for PCA components.
    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    if n_nodes_per_type is None:
        n_nodes_per_type = {}
    if node_labels_per_type is None:
        node_labels_per_type = {}
    # Fit PCA with enough components
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    embeddings_pca_per_type = {}
    count = 0
    for i, (node_type, n_nodes) in enumerate(n_nodes_per_type.items()):
        embeddings_pca_per_type[node_type] = embeddings_pca[count:count+n_nodes, :]
        count += n_nodes
    
    if len(embeddings_pca_per_type) == 0:
        embeddings_pca_per_type['all'] = embeddings_pca
    
    # 6. For each pair of PCA components, create plots
    figs_and_axes = []
    # Get unique x and y components
    x_comps = np.arange(1, n_components + 1)
    y_comps = np.arange(1, n_components + 1)

    # Create subplot grid
    n_rows = len(y_comps)
    n_cols = len(x_comps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Create plots for each component combination
    
    for x_comp in x_comps:
        for y_comp in y_comps:
            for i, (node_type, embeddings_pca) in enumerate(embeddings_pca_per_type.items()):
            
                # Get row and column index for subplot
                row_idx = y_comp - 1  # Subtract 1 since PCA components are 1-based but array indices are 0-based
                col_idx = x_comp - 1
                
                ax = axes[row_idx, col_idx]
                ax.scatter(
                    embeddings_pca[:, x_comp - 1],  # Also adjust indices for embeddings_pca
                    embeddings_pca[:, y_comp - 1],
                    c=colors[i % len(colors)],
                    alpha=0.6,
                    label=node_type
                )
                
            # Add text labels for each point
            for j, coords in enumerate(embeddings_pca):
                if node_type not in node_labels_per_type:
                    continue
                node_label = node_labels_per_type[node_type][j]
                ax.annotate(f'{node_label}', 
                          (coords[x_comp - 1], coords[y_comp - 1]),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=8,
                          alpha=0.5)
                
            ax.set_xlabel(f"PCA {x_comp}")
            ax.set_ylabel(f"PCA {y_comp}")
            ax.set_title(f"PCA {x_comp} vs {y_comp}")
            ax.legend()

    plt.tight_layout()

    if save_dir is not None:
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)

    if close:
        plt.close(fig)
    return fig, axes
