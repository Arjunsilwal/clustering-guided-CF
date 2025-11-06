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

        # [NEW] Attributes to store results
        self.df_with_clusters = None  # Stores the final DataFrame with cluster assignments
        self.best_n_clusters = None  # Stores the optimal number of clusters
        self.best_cluster_label = None  # Stores the label of the cluster (e.g., 1, 2, 3)

    def perform_clustering(
            self, method="ward"
    ):  # ‘ward’ minimizes the variance of the clusters being merged.
        drop_cols = [self.label_col]
        if "temp_cluster" in self.df.columns:
            drop_cols.append("temp_cluster")

        # [MODIFIED] Use df_features, not a dropped version
        self.linkage_matrix = linkage(self.df_features, method=method)
        # self.df_features = self.df.drop(columns=drop_cols) # This line seemed to be misplaced

    def find_optimal_cluster(self):
        best_score = -np.inf
        optimal_cluster = None
        optimal_new_min = None
        optimal_new_max = None

        # [MODIFIED] Store best results as attributes
        optimal_df_with_clusters = None
        optimal_n_clusters = None
        optimal_cluster_label = None

        # we can vary the number of clusters
        for n_clusters in range(2, 7):
            # assign cluster labels to each sample
            cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion="maxclust")
            df_with_clusters = self.df.copy()
            df_with_clusters["temp_cluster"] = cluster_labels

            # find the cluster of the bad sample
            target_cluster_label = df_with_clusters.loc[self.bad_sample_idx][
                "temp_cluster"
            ]
            target_cluster_data = df_with_clusters[
                df_with_clusters["temp_cluster"] == target_cluster_label
                ]

            # check label distribution in the target cluster
            label_counts = target_cluster_data[self.label_col].value_counts()
            good_in_cluster = label_counts.get(self.desired_label, 0)
            bad_in_cluster = label_counts.get(self.original_label, 0)

            # scoring
            total_good = (self.df[self.label_col] == self.desired_label).sum()
            total_bad = (self.df[self.label_col] == self.original_label).sum()

            # Avoid division by zero if total_good or total_bad is 0
            ratio_good = good_in_cluster / total_good if total_good > 0 else 0
            ratio_bad = bad_in_cluster / total_bad if total_bad > 0 else 0

            # score = ratio_good - ratio_bad  # (original score)

            # [MODIFIED] Score from K-Means: prioritize finding good samples, penalize picking up bad ones
            # If good_in_cluster is 0, score will be negative
            # If good_in_cluster > 0, score will be positive
            # This score prioritizes finding *any* good samples over finding none.
            if good_in_cluster > 0:
                score = ratio_good - (ratio_bad * 0.5)  # Prioritize good, lightly penalize bad
            else:
                score = -ratio_bad  # If no good, bigger negative score is worse

            print(
                f"  n_clusters={n_clusters}, target_cluster={target_cluster_label}, good={good_in_cluster}, bad={bad_in_cluster}, score={score:.4f}")

            if score > best_score:
                best_score = score
                optimal_cluster = target_cluster_data
                optimal_new_min = target_cluster_data.drop(
                    columns=[self.label_col, "temp_cluster"]
                ).min()
                optimal_new_max = target_cluster_data.drop(
                    columns=[self.label_col, "temp_cluster"]
                ).max()

                # [MODIFIED] Save the best results
                optimal_df_with_clusters = df_with_clusters
                optimal_n_clusters = n_clusters
                optimal_cluster_label = target_cluster_label

        # [MODIFIED] Store best results as attributes
        self.df_with_clusters = optimal_df_with_clusters
        self.best_n_clusters = optimal_n_clusters
        self.best_cluster_label = optimal_cluster_label

        print(
            f"\n: Best n_clusters: {self.best_n_clusters} (Cluster {self.best_cluster_label}) with score = {best_score:.4f}")

        return optimal_cluster, optimal_new_min, optimal_new_max

    def plot_dendrogram(self, dataset_name, label_name):
        # plot dendrogram
        title_name = dataset_name.split("_")[0] + "_" + label_name
        plt.figure(figsize=(10, 7))
        dendrogram(self.linkage_matrix)
        plt.title(f"Dendrogram for {title_name}")
        plt.xlabel("Samples")
        plt.ylabel("Distance")

        if not os.path.exists("dendrograms"):
            os.makedirs("dendrograms")
        plt.savefig(f"dendrograms/dendrogram_{title_name}.png")
        plt.close()


class LocalKNNSelection:
    def __init__(self, df, label_col, bad_sample_idx):
        self.df = df  # This has labels
        self.label_col = label_col
        self.bad_sample_idx = bad_sample_idx
        self.df_features = df.drop(columns=[label_col])  # This is just features

    def find_knn_boundaries(self, k):
        bad_sample_features = self.df_features.loc[[self.bad_sample_idx]]

        # Calculate distances from the bad sample to all other samples
        distances = cdist(bad_sample_features, self.df_features, metric="euclidean")[0]

        # Get indices of the k+1 nearest neighbors (include the sample itself, then drop it)
        nearest_indices = np.argsort(distances)[1: k + 1]

        # Get the feature data for these neighbors
        knn_features = self.df_features.iloc[nearest_indices]

        # Combine with the original bad sample to create boundaries
        combined_features = pd.concat([bad_sample_features, knn_features])

        # Recalculate feature bounds (min and max)
        new_min_values = combined_features.min()
        new_max_values = combined_features.max()

        # [NEW] Also return the neighbors themselves for feasibility check
        knn_data_with_labels = self.df.iloc[nearest_indices]

        return new_min_values, new_max_values, knn_data_with_labels

    # [NEW] Method for fallback
    def find_next_viable_seed(self, k_to_skip, desired_label):
        """
        Finds the closest "good" sample, skipping the first k_to_skip neighbors.
        """
        bad_sample_features = self.df_features.loc[[self.bad_sample_idx]]

        # Calculate distances from the bad sample to all other samples
        distances = cdist(bad_sample_features, self.df_features, metric="euclidean")[0]

        # Get indices of *all* neighbors, sorted by distance
        all_nearest_indices = np.argsort(distances)

        # Iterate, starting *after* the initial k neighbors we already checked
        # Start from k_to_skip + 1 (to skip the original k neighbors + the sample itself)
        for idx in all_nearest_indices[k_to_skip + 1:]:
            # Get the label of this neighbor
            neighbor_label = self.df.loc[idx, self.label_col]

            if neighbor_label == desired_label:
                # This is the first "good" sample found outside the initial k-group
                seed_instance_features = self.df_features.loc[[idx]]
                return seed_instance_features

        # If no viable sample is found in the entire dataset
        raise Exception("Fallback Failed: No viable samples found in the entire dataset.")