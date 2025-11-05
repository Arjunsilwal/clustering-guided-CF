# last updated: 3/21/2025
import os
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(42)
random.seed(42)


class KMeansClustering:
    def __init__(self, df_with_label, df_wo_label, col_name):

        self.df_with_label = df_with_label
        self.df_wo_label = df_wo_label
        self.col_name = col_name

    def compute_kmeans(self, k):
        # ensure removing any previously added columns to avoid errors
        self.df_with_label.drop(
            columns=["dbscan", "cluster"], errors="ignore", inplace=True
        )
        X = self.df_wo_label
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        # add a new column 'cluster' to the DataFrame and rename the labels
        self.df_with_label["cluster"] = kmeans.labels_
        self.df_with_label["cluster"] = self.df_with_label["cluster"].apply(
            lambda x: f"C{x + 1}"
        )

        # group by clusters and count occurrences of each label
        cluster_label_counts = (
            self.df_with_label.groupby(["cluster", self.col_name])
            .size()
            .unstack(fill_value=0)
        )

        return self.df_with_label, cluster_label_counts

    def evaluate_k(self, k, bad_sample_index):
        # for a given k and a chosen sample, find the cluster the sample belongs to.
        # score S = (# of desirable in cluster/total desirable) - lambda * (# of undesirable in lucster/total undesirable)
        # get the min and max boundaries from that cluster to update the min and max boundaries for GAs
        updated_df, cluster_label_counts = self.compute_kmeans(k)
        try:
            target_cluster = updated_df.loc[bad_sample_index, "cluster"]
        except KeyError:
            raise ValueError(
                f"Bad sample index {bad_sample_index} not found in the dataset."
            )

        cluster_data = updated_df[updated_df["cluster"] == target_cluster]
        print("target cluster", target_cluster)
        num_good_in_cluster = (cluster_data[self.col_name] == 0).sum()
        print(f"Good in cluster: {num_good_in_cluster}")
        num_bad_in_cluster = (cluster_data[self.col_name] == 1).sum()
        print(f"Bad in cluster: {num_bad_in_cluster}")
        # count totals in the entire dataset
        total_good = (updated_df[self.col_name] == 0).sum()
        print(f"Total good: {total_good}")
        total_bad = (updated_df[self.col_name] == 1).sum()
        print(f"Total bad: {total_bad}")
        # compute ratios
        ratio_good = num_good_in_cluster / total_good if total_good > 0 else 0
        print(f"Ratio good: {ratio_good}")
        ratio_bad = num_bad_in_cluster / total_bad if total_bad > 0 else 0
        print(f"Ratio bad: {ratio_bad}")
        score = ratio_good - ratio_bad
        print(f"Score: {score}")
        # compute new boundaries based only on the rows in this cluster
        cluster_rows = cluster_data.drop(
            columns=[self.col_name, "cluster"], errors="ignore"
        )
        new_min_values = cluster_rows.min()
        new_max_values = cluster_rows.max()
        print(f"For k={k}: Sample in {target_cluster}, score = {score:.4f}")
        return (
            score,
            target_cluster,
            new_min_values,
            new_max_values,
            cluster_label_counts,
        )

    def calculate_distance_per_cluster(self, cluster_label, sample, label):
        cluster_data = self.df_with_label[
            self.df_with_label["cluster"] == cluster_label
        ]
        rest_data_wo_label = cluster_data.drop(columns=[label, "cluster"]).values
        sample_wo_label = sample.drop(columns=[label]).values
        distances = np.linalg.norm(rest_data_wo_label - sample_wo_label, axis=1)
        cluster_data["distance"] = distances
        group_blue = cluster_data[cluster_data[label] == 0].sort_values(by="distance")
        group_red = cluster_data[cluster_data[label] == 1].sort_values(by="distance")
        ordered_df = pd.concat([group_red, group_blue])
        return ordered_df

    @staticmethod
    def plot_kmeans_results(cluster_label_counts, dataset_name, label_name, k):
        title_name = dataset_name.split("_")[0] + "_" + label_name
        plt.figure(figsize=(6, 4))
        colors = ["blue", "red"]

        ax = cluster_label_counts.plot(kind="bar", color=colors)
        plt.title(f"K = {k}", fontsize=16, fontweight="bold")
        plt.xlabel("KMeans Clustering", fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")
        plt.xticks(
            range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0
        )

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Good: Blue", "Bad: Red"], fontsize=12)

        if not os.path.exists("kmeans"):
            os.makedirs("kmeans")

        plot_filename = os.path.join("kmeans", f"{title_name}_k{k}_plot.png")
        plt.savefig(plot_filename)
        plt.close()

    @staticmethod
    def generate_distance_plot(ordered_data, label, cluster_label, plot_name):
        plt.figure(figsize=(8, 6))
        ordered_distance = ordered_data["distance"].values
        ordered_label = ordered_data[[label]]
        colors = ["blue" if val == 0 else "red" for val in ordered_label[label]]
        plt.bar(range(len(ordered_distance)), ordered_distance, color=colors)
        plt.title(f"{cluster_label} Distances", fontsize=14, fontweight="bold")

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

    # def calculate_distance(self, label):
    #     cluster1 = self.df_with_label[self.df_with_label["cluster"] == "C1"]
    #     # sample_index = 525  # example fixed sample index
    #     sample = cluster1.loc[[sample_index]]
    #     cluster1 = cluster1.drop(index=sample_index)
    #     unique_clusters = self.df_with_label["cluster"].unique()
    #     results = {}
    #     for cluster_label in unique_clusters:
    #         ordered_df = self.calculate_distance_per_cluster(
    #             cluster_label, sample, label
    #         )
    #         results[cluster_label] = {
    #             "ordered_df": ordered_df,
    #             "sample_index": sample_index,
    #         }
    #     return results
