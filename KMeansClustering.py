# last updated: 3/21/2025
import os
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist  # Import cdist

np.random.seed(42)
random.seed(42)


class KMeansClustering:
    def __init__(self, df_with_label, df_wo_label, col_name):

        self.df_with_label = df_with_label
        self.df_wo_label = df_wo_label
        self.col_name = col_name
        self.kmeans = None  # Initialize kmeans attribute

    def compute_kmeans(self, k):
        # ensure removing any previously added columns to avoid errors
        self.df_with_label.drop(
            columns=["dbscan", "cluster"], errors="ignore", inplace=True
        )
        X = self.df_wo_label
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Added n_init=10 to suppress warning
        kmeans.fit(X)

        # --- THIS IS THE FIX ---
        # Store the kmeans object so we can access its centers later
        self.kmeans = kmeans
        # ---------------------

        # add a new column 'cluster' to the DataFrame and rename the labels
        self.df_with_label["cluster"] = kmeans.labels_
        self.df_with_label["cluster"] = self.df_with_label["cluster"].apply(
            lambda x: f"C{x + 1}"
        )

        # group by clusters and count occurrences of each label
        cluster_label_counts = (
            self.df_with_label.groupby("cluster")[self.col_name]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Convert keys to int if they are not, for correct sorting
        cluster_label_counts.index = cluster_label_counts.index.map(
            lambda x: (
                int(x[1:]) if isinstance(x, str) and x.startswith("C") else x
            )
        )
        cluster_label_counts = cluster_label_counts.sort_index()
        # Convert index back to string "C" format
        cluster_label_counts.index = cluster_label_counts.index.map(
            lambda x: f"C{x}"
        )

        return self.df_with_label, cluster_label_counts.to_dict(orient="index")

    def evaluate_k(self, k, bad_sample_index):
        updated_df, cluster_label_counts = self.compute_kmeans(k)

        # find which cluster the bad sample belongs to
        target_cluster = updated_df.loc[bad_sample_index, "cluster"]
        print(f"target cluster {target_cluster}")
        # get counts for the target cluster
        counts_in_target_cluster = cluster_label_counts.get(target_cluster, {})
        good_in_cluster = counts_in_target_cluster.get(0, 0)  # 0 is good
        bad_in_cluster = counts_in_target_cluster.get(1, 0)  # 1 is bad
        print(f"Good in cluster: {good_in_cluster}")
        print(f"Bad in cluster: {bad_in_cluster}")
        # get total counts
        total_good = sum(
            counts.get(0, 0) for counts in cluster_label_counts.values()
        )
        total_bad = sum(counts.get(1, 0) for counts in cluster_label_counts.values())
        print(f"Total good: {total_good}")
        print(f"Total bad: {total_bad}")

        # calculate ratios
        ratio_good = good_in_cluster / total_good if total_good > 0 else 0
        ratio_bad = bad_in_cluster / total_bad if total_bad > 0 else 0

        print(f"Ratio good: {ratio_good}")
        print(f"Ratio bad: {ratio_bad}")

        # scoring: high score is bad, low score is good
        # score = ratio_bad - ratio_good
        score = ratio_good - ratio_bad  # want to maximize good, minimize bad

        print(f"Score: {score}")

        # filter dataset to only the target cluster
        cluster_data = updated_df[updated_df["cluster"] == target_cluster]
        features_in_cluster = cluster_data.drop(
            columns=[self.col_name, "cluster"], errors="ignore"
        )

        # recalculate min/max boundaries
        if features_in_cluster.empty:
            # Handle empty cluster case: return original boundaries
            print(f"Warning: Cluster {target_cluster} is empty. Using original boundaries.")
            new_min_values = self.df_wo_label.min()
            new_max_values = self.df_wo_label.max()
        else:
            new_min_values = features_in_cluster.min()
            new_max_values = features_in_cluster.max()

        print(f"For k={k}: Sample in {target_cluster}, score = {score:.4f}")
        return score, target_cluster, new_min_values, new_max_values, cluster_label_counts

    def plot_kmeans_results(self, cluster_label_counts, dataset_name, target_cluster, k):
        title_name = dataset_name.split("_")[0] + "_kmeans"
        plt.figure(figsize=(6, 4))

        # Ensure 'cluster_label_counts' is a DataFrame for plotting
        if isinstance(cluster_label_counts, dict):
            cluster_label_counts = pd.DataFrame.from_dict(cluster_label_counts, orient='index').fillna(0)

        # Ensure columns 0 and 1 exist even if no samples are present
        if 0 not in cluster_label_counts.columns:
            cluster_label_counts[0] = 0
        if 1 not in cluster_label_counts.columns:
            cluster_label_counts[1] = 0

        # Sort columns to ensure '0' (Good) is always plotted first (Blue)
        cluster_label_counts = cluster_label_counts.sort_index(axis=1)
        colors = ["blue", "red"]  # 0 (Good) is blue, 1 (Bad) is red

        ax = cluster_label_counts.plot(kind="bar", color=colors, stacked=True)
        plt.title(
            f"K-Means Clustering (k = {k})",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Cluster", fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")

        # Make sure xticks match the sorted index
        plt.xticks(
            range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0
        )

        # Highlight the target cluster
        try:
            target_idx = cluster_label_counts.index.get_loc(target_cluster)
            ax.patches[target_idx].set_edgecolor("black")
            ax.patches[target_idx].set_linewidth(2)
            # Also highlight the stacked bar (for 'bad' samples) if it exists
            if 1 in cluster_label_counts.columns:
                ax.patches[target_idx + len(cluster_label_counts)].set_edgecolor("black")
                ax.patches[target_idx + len(cluster_label_counts)].set_linewidth(2)

        except KeyError:
            print(f"Could not find target cluster {target_cluster} in index for plotting.")
        except IndexError:
            print("IndexError while trying to highlight cluster.")

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=["Good (0)", "Bad (1)"],
            title="Label",
            fontsize=12,
        )
        plt.tight_layout(pad=2.0)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        output_path = os.path.join(
            "plots", f"{title_name}_k{k}_target_{target_cluster}.png"
        )
        plt.savefig(output_path)
        plt.close()

    def calculate_distance_per_cluster(self, cluster_name, sample, label):
        cluster_data = self.df_with_label[
            self.df_with_label["cluster"] == cluster_name
            ]
        sample_features = sample.drop(columns=[label], errors="ignore")
        cluster_features = cluster_data.drop(columns=[label, "cluster"], errors="ignore")

        if cluster_features.empty:
            print(f"No other samples in cluster {cluster_name} to calculate distance.")
            return pd.DataFrame(columns=cluster_data.columns.tolist() + ["distance"])

        distances = cdist(sample_features, cluster_features, metric="euclidean").flatten()

        # Use .loc to avoid SettingWithCopyWarning
        cluster_data_copy = cluster_data.copy()
        cluster_data_copy["distance"] = distances

        ordered_data = cluster_data_copy.sort_values(by="distance")
        return ordered_data

    def generate_distance_plot(self, ordered_data, label, cluster_name, plot_name):
        if ordered_data.empty:
            print(f"Skipping distance plot for empty cluster {cluster_name}.")
            return

        colors = ordered_data[label].map({0: "blue", 1: "red"})
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(ordered_data)),
            ordered_data["distance"],
            color=colors.values,
        )
        plt.title(
            f"Distance from Instance to Samples in Cluster {cluster_name}",
            fontsize=16,
            fontweight="bold",
        )
        # add legends
        handles = [
            Line2D(
                [0], [0], color="blue", linestyle="-", linewidth=10, label="Good: Blue"
            ),
            Line2D(
                [0], [0], color="red", linestyle="-", linewidth=10, label="Bad: Red"
            ),
        ]
        plt.legend(handles=handles, fontsize=12)
        plt.xlabel("Number of Samples", fontsize=14, fontweight="bold")
        plt.ylabel("Distance from the Instance", fontsize=14, fontweight="bold")
        plt.tight_layout(pad=2.0)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        output_path = os.path.join("plots", plot_name)
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()