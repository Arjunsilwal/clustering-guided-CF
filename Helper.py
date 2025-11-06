# last upated: 4/12/2025
# helper.py
import os
import pandas as pd
import numpy as np  # Make sure numpy is imported
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from scipy.spatial.distance import cdist  # [NEW] Import cdist


class Helper:
    # [MOVED] This function was in breast_cancer_main.py
    @staticmethod
    def find_nearest_viable_cluster(empty_cluster_label, all_centroids, all_cluster_labels, label_counts,
                                    desired_label):
        """
        Finds the cluster that is (1) viable (contains desired_label) and
        (2) closest to the empty cluster.
        """
        print(f":   Finding nearest viable neighbor to {empty_cluster_label}...")

        # 1. Get the centroid of the empty cluster
        # We must parse the cluster index from the label (e.g., "C3" -> 2)
        empty_cluster_index = all_cluster_labels.index(empty_cluster_label)
        empty_centroid = all_centroids[empty_cluster_index]

        viable_clusters = []
        for cluster_name, counts in label_counts.items():
            if cluster_name != empty_cluster_label and counts.get(desired_label, 0) > 0:
                viable_clusters.append(cluster_name)

        if not viable_clusters:
            raise Exception("Fallback Failed: No other clusters contain any desirable samples.")

        min_dist = float('inf')
        best_fallback_cluster = None

        # 2. Find the closest centroid among the viable clusters
        for cluster_name in viable_clusters:
            cluster_index = all_cluster_labels.index(cluster_name)
            viable_centroid = all_centroids[cluster_index]

            # Calculate Euclidean distance
            dist = np.linalg.norm(empty_centroid - viable_centroid)

            if dist < min_dist:
                min_dist = dist
                best_fallback_cluster = cluster_name

        print(f":   Nearest viable cluster is {best_fallback_cluster} (Distance: {min_dist:.2f})")
        return best_fallback_cluster

    # [MOVED] This function was in breast_cancer_main.py
    @staticmethod
    def find_seed_and_boundaries(fallback_cluster_name, clustered_df, data_wo_label, original_instance_index,
                                 desired_label,
                                 class_label):
        """
        Finds the closest "seed" instance from the fallback cluster and
        defines new search boundaries based on it.
        """
        print(f":   Finding seed instance in {fallback_cluster_name}...")

        # 1. Get the original instance's features as a 2D DataFrame
        original_instance_features = data_wo_label.loc[[original_instance_index]]

        # 2. Get all viable samples from the fallback cluster
        viable_samples = clustered_df[
            (clustered_df['cluster'] == fallback_cluster_name) &
            (clustered_df[class_label] == desired_label)
            ]

        if viable_samples.empty:
            raise Exception(
                f"Fallback Failed: Viable cluster {fallback_cluster_name} mysteriously has no desired samples.")

        # Get just the features of these viable samples
        viable_samples_features = viable_samples.drop(columns=[class_label, 'cluster'])

        # 3. Calculate distances from original instance to all viable samples
        distances = cdist(original_instance_features, viable_samples_features, metric='euclidean')

        # 4. Get the iloc (integer position) of the closest sample
        closest_sample_iloc = distances.argmin()

        # 5. Get the seed instance's features (as a 2D DataFrame)
        seed_instance = viable_samples_features.iloc[[closest_sample_iloc]]

        print(f":   Seed instance found (original index: {seed_instance.index.values})")

        # 6. Define new boundaries as a "bounding box" containing both points
        combined_features = pd.concat([original_instance_features, seed_instance])
        fallback_min_values = combined_features.min()
        fallback_max_values = combined_features.max()

        return fallback_min_values, fallback_max_values

    # [MOVED] This function was in hierarchical_cluster_main.py
    @staticmethod
    def find_nearest_viable_sample(df_all_features, df_all_labels, original_instance_index, empty_cluster_data,
                                   desired_label, class_label):
        """
        Finds the single closest "good" sample from the entire dataset,
        excluding those in the empty cluster.
        """
        print(f":   Finding nearest viable sample (outside empty cluster)...")

        # 1. Get the original instance's features (as a 2D DataFrame)
        original_instance_features = df_all_features.loc[[original_instance_index]]

        # 2. Get all viable samples from *outside* the empty cluster
        # Get indices of samples *not* in the empty cluster
        outside_indices = df_all_features.index.difference(empty_cluster_data.index)

        # Filter all data to get viable samples from outside
        viable_samples_outside = df_all_labels[
            (df_all_labels.index.isin(outside_indices)) &
            (df_all_labels[class_label] == desired_label)
            ]

        if viable_samples_outside.empty:
            raise Exception("Fallback Failed: No viable samples found in the entire dataset.")

        # 3. Get just the features of these viable samples
        viable_features_outside = df_all_features.loc[viable_samples_outside.index]

        # 4. Calculate distances from original instance to all viable samples
        distances = cdist(original_instance_features, viable_features_outside, metric='euclidean')

        # 5. Get the iloc (integer position) of the closest sample
        closest_sample_iloc = distances.argmin()

        # 6. Get the seed instance's features (as a 2D DataFrame)
        seed_instance_features = viable_features_outside.iloc[[closest_sample_iloc]]

        print(f":   Nearest viable seed instance found (original index: {seed_instance_features.index.values})")

        return seed_instance_features

    # [MOVED] This function was in hierarchical_cluster_main.py & knn_bound_main.py
    @staticmethod
    def find_seed_and_boundaries_from_sample(original_instance_features, seed_instance_features):
        """
        Creates a bounding box from an original instance and a seed instance.
        """
        print(f":   Defining new search boundaries...")

        # 1. Define new boundaries as a "bounding box" containing both points
        combined_features = pd.concat([original_instance_features, seed_instance_features])
        fallback_min_values = combined_features.min()
        fallback_max_values = combined_features.max()

        return fallback_min_values, fallback_max_values

    @staticmethod
    def save_hist_FX(
            hist_F,
            hist_X,
            extracted_data_name,
            cluster_mode,
            pop_size,
            sample_idx,
            data_wo_label,
    ):
        col_list = data_wo_label.columns.tolist()
        data = []
        for f_arr, x_list in zip(hist_F, hist_X):
            for f, x in zip(f_arr, x_list):
                x_dict = dict(zip(col_list, x))
                row = {"(F1, F2)": f, "X": x_dict}
                data.append(row)
        if not os.path.exists("hist_FX"):
            os.makedirs("hist_FX")

        df = pd.DataFrame(data)
        df.to_csv(
            f"hist_FX/hist_FX_{extracted_data_name}_{cluster_mode}_{pop_size}_{sample_idx}.csv",
            index=False,
        )
        print("hist_FX is saved to CSV.")

    @staticmethod
    def plot_scatter(pareto_front, problem, sample_idx, seed, cluster_mode, k=None):
        """
        Plots the Pareto front.

        Args:
            pareto_front (np.array): The final Pareto front (F from res).
            problem (CFProblem): The problem instance, to get all_f1/all_f2.
            sample_idx (int): The index of the sample.
            seed (int): The random seed.
            cluster_mode (str): The name of the clustering mode (for file naming).
            k (int, optional): The value of k if using knn.
        """
        if not os.path.exists("scatter_plots_files"):
            os.makedirs("scatter_plots_files")

        # Set the plot title
        plot_title = f"Pareto Front (Sample: {sample_idx}, Seed: {seed})"

        # Use a new plot object
        plot = Scatter(title=plot_title, labels=["Error", "Distance"])

        # Access problem.all_f1 and problem.all_f2
        # Add all evaluated solutions in grey
        if hasattr(problem, 'all_f1') and hasattr(problem, 'all_f2') and len(problem.all_f1) > 0:
            all_solutions = np.column_stack([problem.all_f1, problem.all_f2])
            plot.add(all_solutions, s=10, color="grey", alpha=0.3, label="History")

        # Plot the final pareto_front in red
        plot.add(pareto_front, s=30, color="red", alpha=0.8, label="Pareto Front")

        # Add legend
        plot.legend = True

        # Construct file path
        file_name = f"scatter_{cluster_mode}_{sample_idx}_seed{seed}"
        if k is not None:
            file_name += f"_k{k}"

        plot_path = os.path.join("scatter_plots_files", f"{file_name}.png")

        plot.save(plot_path, dpi=300)
        print(f"Scatter plot is saved to {plot_path}")

    @staticmethod
    def log_avg_hv_per_row(avg_hv, extracted_data_name, cluster_mode, k=None):
        if not os.path.exists("hv_logs"):
            os.makedirs("hv_logs")

        file_name = f"hv_logs/avg_hv_logs_{extracted_data_name}.csv"
        file_exists = os.path.isfile(file_name)

        # Create a dictionary for the new log entry
        log_data = {"cluster_mode": cluster_mode}
        if k is not None:
            log_data["k"] = k
        log_data.update(
            {f"gen_{i + 1}": hv for i, hv in enumerate(avg_hv)}
        )

        # Convert to DataFrame
        new_log_df = pd.DataFrame([log_data])

        # Safer file writing
        if file_exists:
            try:
                existing_log_df = pd.read_csv(file_name)
                new_log_df = pd.concat([existing_log_df, new_log_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                # File was empty, new_log_df is already correct
                pass

        # Write/Overwrite with updated data
        new_log_df.to_csv(file_name, mode='w', header=True, index=False)
        print(f"Hypervolume log saved to {file_name}")

    @staticmethod
    def plot_combined_pareto_front(
            sample_idx, seed, res_without, res_with, extracted_data_name, cluster_mode, k=None
    ):
        if not os.path.exists("pareto_combined"):
            os.makedirs("pareto_combined")

        plt.figure(figsize=(8, 6))
        plt.scatter(
            res_without[:, 0],
            res_without[:, 1],
            c="blue",
            label="Without Clustering",
            alpha=0.7,
        )
        plt.scatter(
            res_with[:, 0],
            res_with[:, 1],
            c="red",
            label=f"With {cluster_mode}",
            alpha=0.7,
        )
        plt.title(
            f"Pareto Front Comparison (Sample: {sample_idx}, Seed: {seed})",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("F1: Error", fontsize=14, fontweight="bold")
        plt.ylabel("F2: Distance", fontsize=14, fontweight="bold")
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout(pad=2.0)

        if k:
            plt.savefig(
                f"pareto_combined/pareto_comparison_sample_{sample_idx}_seed_{seed}_k={k}.png",
                dpi=300,
            )
        else:
            plt.savefig(
                f"pareto_combined/pareto_comparison_sample_{sample_idx}_seed_{seed}.png",
                dpi=300,
            )
        plt.close()

    @staticmethod
    def plot_combined_avg_hv(
            avg_hv_without, avg_hv_with, sample_idx, cluster_mode, k=None
    ):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
        gens = range(1, len(avg_hv_without) + 1)

        axs[0].plot(gens, avg_hv_without, color="red")
        axs[0].set_title("Without Clustering")
        axs[0].set_xlabel("Generation")
        axs[0].set_ylabel("Average Hypervolume")
        axs[0].grid(True)

        axs[1].plot(gens, avg_hv_with, color="blue")
        axs[1].set_title(f"With {cluster_mode}")
        axs[1].set_xlabel("Generation")
        axs[1].grid(True)

        fig.suptitle(f"AVG HV Convergence (Sample: {sample_idx})", fontsize=14)
        plt.tight_layout()

        if not os.path.exists("hv_plots"):
            os.makedirs("hv_plots")

        if k:
            plt.savefig(
                f"hv_plots/avg_hv_convergence_sample_{sample_idx}_k={k}.png", dpi=300
            )
        else:
            plt.savefig(f"hv_plots/avg_hv_convergence_sample_{sample_idx}.png", dpi=300)

        plt.close()