# last updated: 4/5/2025
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# import random
# from DataProcessing import DataProcessing
# from sklearn.cluster import AgglomerativeClustering


class HierarchicalClustering:
    def __init__(self, df, original_label, desired_label, label_col, bad_sample_idx):
        self.df = df
        self.original_label = original_label
        self.desired_label = desired_label
        self.label_col = label_col
        self.bad_sample_idx = bad_sample_idx
        self.df_features = df.drop(columns=[label_col])
        self.linkage_matrix = None

    def perform_clustering(
        self, method="ward"
    ):  # ‘ward’ minimizes the variance of the clusters being merged.
        drop_cols = [self.label_col]
        if "temp_cluster" in self.df.columns:
            drop_cols.append("temp_cluster")
        self.linkage_matrix = linkage(self.df_features, method=method)
        self.df_features = self.df.drop(columns=drop_cols)

    def find_optimal_cluster(self):
        best_score = -np.inf
        optimal_cluster = None
        optimal_good_count = 0
        optimal_bad_count = 0
        n_samples = len(self.df)

        # max_good = 0
        # optimal_cluster = None
        # optimal_cluster_bad_count = 0
        # n_samples = len(self.df)
        fallback_cluster = None
        fallback_good_count = 0

        for n_clusters in range(n_samples, 1, -1):
            labels = fcluster(self.linkage_matrix, t=n_clusters, criterion="maxclust")
            # fcluster() is to form flat clusters from the hierarchical clustering defined by the given linkage matrix.
            # maxclust is to find a minimum threshold r so that the cophenetic distance between any two original observations
            # in the same flat cluster is no more than r and no more than t flat clusters are formed.
            self.df["temp_cluster"] = labels
            bad_sample_cluster = self.df.loc[self.bad_sample_idx, "temp_cluster"]

            for cluster_id in np.unique(labels):
                if cluster_id != bad_sample_cluster:
                    continue

                cluster_data = self.df[self.df["temp_cluster"] == cluster_id]
                num_good = (cluster_data[self.label_col] == self.desired_label).sum()
                num_bad = (cluster_data[self.label_col] == self.original_label).sum()

                score = num_good / (num_bad + 1)

                if score > best_score:
                    best_score = score
                    optimal_cluster = cluster_data.copy()
                    optimal_good_count = num_good
                    optimal_bad_count = num_bad

            # for cluster_id in np.unique(labels):
            #     cluster_data = self.df[self.df["temp_cluster"] == cluster_id]
            #     good_count = (cluster_data[self.label_col] == self.desired_label).sum()

            #     # Store best overall cluster of good samples (for fallback)
            #     if good_count > fallback_good_count:
            #         fallback_good_count = good_count
            #         fallback_cluster = cluster_data.copy()

            #     if cluster_id == bad_sample_cluster and good_count > 0:
            #         if good_count >= max_good:
            #             max_good = good_count
            #             optimal_cluster = cluster_data.copy()
            #             optimal_cluster_bad_count = (
            #                 cluster_data[self.label_col] == self.original_label
            #             ).sum()

        if optimal_cluster is not None:
            # print(f"Using cluster with bad sample and {max_good} good samples.")
            # print(f"Bad sample count in this cluster: {optimal_cluster_bad_count}")
            print("optimal good samples: ", optimal_good_count)
            print("optimal bad samples: ", optimal_bad_count)
            return optimal_cluster, optimal_good_count, optimal_bad_count

        # Fallback: No good samples with bad sample
        if fallback_cluster is not None:
            print(f"No good samples found with bad sample.")
            print(f"Using fallback cluster with {fallback_good_count} good samples.")

            bad_sample_vec = self.df_features.loc[self.bad_sample_idx].values
            fallback_center = (
                fallback_cluster.drop(columns=[self.label_col, "temp_cluster"])
                .mean()
                .values
            )
            dist = euclidean(bad_sample_vec, fallback_center)
            print(f"Distance from bad sample to fallback cluster center: {dist:.4f}")

            return fallback_cluster, fallback_good_count, None

        raise ValueError("No cluster with good samples found at all.")

    def update_boundaries(self, optimal_cluster):
        feature_data = optimal_cluster.drop(columns=[self.label_col, "temp_cluster"])
        new_min = feature_data.min()
        new_max = feature_data.max()
        return new_min, new_max

    def visualize_dendrogram(self):
        plt.figure(figsize=(20, 20))
        dendrogram(
            self.linkage_matrix,
            labels=[
                f"Good-{idx}" if label == self.desired_label else f"Bad-{idx}"
                for idx, label in enumerate(self.df[self.label_col])
            ],
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0,
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.tight_layout()

        if not os.path.exists("hierarchical_plots"):
            os.makedirs("hierarchical_plots")
        plt.savefig(
            os.path.join("hierarchical_plots", f"dendrogram_{self.bad_sample_idx}.png")
        )
        plt.close()

    def export_merge_log_to_excel(self):
        output_file = f"merge_log_{self.bad_sample_idx}.csv"
        n_samples = len(self.df)
        clusters = {i: [i] for i in range(n_samples)}
        log_rows = []
        for step, (idx1, idx2, dist, sample_count) in enumerate(self.linkage_matrix):
            idx1, idx2 = int(idx1), int(idx2)
            new_cluster_id = n_samples + step
            merged_cluster = clusters[idx1] + clusters[idx2]
            clusters[new_cluster_id] = merged_cluster
            contains_bad = self.bad_sample_idx in merged_cluster
            labels = self.df.loc[merged_cluster, self.label_col]
            num_good = (labels == self.desired_label).sum()
            num_bad = (labels == self.original_label).sum()
            log_rows.append(
                {
                    "Step": step + 1,
                    "Cluster ID": new_cluster_id,
                    "Merged Nodes": f"{idx1}, {idx2}",
                    "Contains Bad Sample": contains_bad,
                    "Num Good": num_good,
                    "Num Bad": num_bad,
                    "Total Samples": len(merged_cluster),
                    "Distance": dist,
                }
            )
            log_df = pd.DataFrame(log_rows)
            log_df.to_csv(output_file)
            # print(f"Merge log saved to {output_file}")

    def run_analysis(self):
        self.perform_clustering()
        optimal_cluster, max_good_samples, optimal_cluster_bad_count = (
            self.find_optimal_cluster()
        )
        new_min, new_max = self.update_boundaries(optimal_cluster)
        # self.visualize_dendrogram()
        # self.export_merge_log_to_excel()

        print(f"Max number of good samples grouped with bad sample: {max_good_samples}")
        print("Updated feature min boundaries: ")
        print(new_min)
        print("Updated feature max boundaries: ")
        print(new_max)
        return optimal_cluster, new_min, new_max


# neighborhood-based bounding
class LocalKNNSelection:
    def __init__(self, df, label_col, bad_sample_idx, desired_label, k):
        self.df = df
        self.label_col = label_col
        self.bad_sample_idx = bad_sample_idx
        self.desired_label = desired_label
        self.k = k

    def get_knn_bounds(self):
        # Separate features and target
        df_features = self.df.drop(columns=[self.label_col])
        good_samples = self.df[self.df[self.label_col] == self.desired_label]
        good_features = good_samples.drop(columns=[self.label_col])

        bad_sample = df_features.loc[self.bad_sample_idx].values.reshape(1, -1)

        # Compute distances to all good samples
        distances = cdist(bad_sample, good_features.values, metric="euclidean")[0]
        nearest_indices = np.argsort(distances)[: self.k]

        nearest_good_samples = good_features.iloc[nearest_indices]
        min_bounds = nearest_good_samples.min()
        max_bounds = nearest_good_samples.max()

        print(f"Using KNN with k={self.k} to define bounds from nearby good samples.")
        return min_bounds, max_bounds


# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)


# file_name = "breast_cancer.csv"
# extracted_data_name = file_name.split(".")[0]
# class_label = "Diagnosis"
# mapping_values = {"M": 1, "B": 0}

# data_processing = DataProcessing(file_name, class_label, mapping_values)
# data_wo_label, target, min_values, max_values = data_processing.load_and_process()
# print("min values: ", min_values)
# print("max values: ", max_values)
# combined_data = pd.concat([data_wo_label, target], axis=1)

# bad_samples = combined_data[combined_data[class_label] == 1]
# bad_sample_idx = random.choice(bad_samples.index.tolist())
# print(f"bad_sample_index: {bad_sample_idx}")

# hc_search = HierarchicalClustering(
#     combined_data, label_col="Diagnosis", bad_sample_idx=bad_sample_idx
# )
# optimal_cluster, new_min_boundaries, new_max_boundaries = hc_search.run_analysis()


# # setting distance_threshold=0 ensures we compute the full tree.
# # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# model = AgglomerativeClustering(n_clusters=2)
# model = model.fit(data_wo_label)
# cluster_labels = model.labels_
# data_wo_label["cluster"] = cluster_labels
# data_wo_label["diagnosis"] = target
# print(data_wo_label.head())

# from sklearn.metrics import adjusted_rand_score, accuracy_score

# print("Adjusted Rand Index:", adjusted_rand_score(target, cluster_labels))


# def plot_agglomerative_results(data_wo_label, dataset_name, label_name, n_clusters):
#     # Combine labels into a DataFrame
#     # df = pd.DataFrame({"Cluster": predicted_labels, "ActualLabel": actual_labels})
#     df = data_wo_label[["cluster", "diagnosis"]]
#     # Count good (0) and bad (1) samples within each cluster
#     cluster_label_counts = (
#         df.groupby("cluster")["diagnosis"].value_counts().unstack(fill_value=0)
#     )

#     title_name = dataset_name.split("_")[0] + "_" + label_name
#     plt.figure(figsize=(6, 4))
#     colors = ["blue", "red"]

#     ax = cluster_label_counts.plot(kind="bar", color=colors)
#     plt.title(
#         f"Agglomerative Clustering (n_clusters = {n_clusters})",
#         fontsize=16,
#         fontweight="bold",
#     )
#     plt.xlabel("Cluster", fontsize=14, fontweight="bold")
#     plt.ylabel("Count", fontsize=14, fontweight="bold")
#     plt.xticks(range(len(cluster_label_counts)), cluster_label_counts.index, rotation=0)

#     handles, _ = ax.get_legend_handles_labels()
#     ax.legend(handles, ["Good: Blue", "Bad: Red"], fontsize=12)

#     if not os.path.exists("agglomerative"):
#         os.makedirs("agglomerative")

#     plot_filename = os.path.join(
#         "agglomerative", f"{title_name}_n_clusters{n_clusters}_plot.png"
#     )
#     plt.savefig(plot_filename)
#     plt.close()
