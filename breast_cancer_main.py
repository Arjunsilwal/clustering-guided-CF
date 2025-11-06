# main.py

import timeit
import pandas as pd
import numpy as np
# [REMOVED] cdist is no longer needed here
from DataProcessing import DataProcessing
from ModelTrainer import ModelTrainer
from CFGenerator import CFGenerator
from CFProblem import CFProblem
from KMeansClustering import KMeansClustering
from Helper import Helper  # [MODIFIED] Import Helper
import random

# good is 0
# bad is 1

# [REMOVED] All helper functions moved to Helper.py
# def find_nearest_viable_cluster(...):
# def find_seed_and_boundaries(...):


def main():
    start = timeit.default_timer()
    file_name = "breast_cancer.csv"
    extracted_data_name = file_name.split(".")[0]
    class_label = "Diagnosis"
    mapping_values = {"M": 1, "B": 0}
    seed = 42

    # [MODIFIED] Use the same sample index for consistent testing
    bad_sample_index = 368

    data_processing = DataProcessing(file_name, class_label, mapping_values)
    data_wo_label, target, min_values, max_values = data_processing.load_and_process()

    combined_data = pd.concat([data_wo_label, target], axis=1)

    # modify the model choice below - available options are: KNN, DTC, SVC, MLP, NBC
    model_trainer = ModelTrainer(
        dataset=data_wo_label,
        target=target,
        class_label=class_label,
        dataset_name=extracted_data_name,
        model_choice="SVC",
    )
    model = model_trainer.train_model()

    original_label = 1  # 1 is malignant
    desired_label = 0  # 0 is benign

    # cf generation without clustering
    print(f"Original min: {min_values}, Original max: {max_values}")

    bad_samples = combined_data[combined_data[class_label] == original_label]
    if bad_samples.empty:
        raise ValueError("No bad samples found in the dataset.")

    # [MODIFIED] Check if our fixed sample is valid
    if bad_sample_index not in bad_samples.index:
        print(f"Warning: Sample {bad_sample_index} is not a 'bad' sample. Choosing a random one.")
        bad_sample_index = random.choice(bad_samples.index.tolist())

    print(f"bad_sample_index: {bad_sample_index}")

    scores = {}
    boundaries = {}
    target_clusters = {}

    kmeans_clustering = KMeansClustering(
        df_with_label=combined_data.copy(),
        df_wo_label=data_wo_label.copy(),
        col_name=class_label,
    )
    for k in range(2, 7):
        score, target_cluster, new_min, new_max, cluster_label_counts = (
            kmeans_clustering.evaluate_k(k, bad_sample_index)
        )

        kmeans_clustering.plot_kmeans_results(
            cluster_label_counts, extracted_data_name, target_cluster, k
        )
        scores[k] = score
        boundaries[k] = (new_min, new_max)
        target_clusters[k] = target_cluster

    # choose the k with the highest score
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    best_cluster = target_clusters[best_k]
    print(f"Best k: {best_k} with score = {best_score:.4f} in cluster {best_cluster}")
    # use the new boundaries from the best cluster
    best_new_min, best_new_max = boundaries[best_k]
    print(f"New min: {best_new_min}, New max: {best_new_max}")
    # update dataset name for clustering
    clustered_dataset_name = extracted_data_name + f"_cluster_k{best_k}"

    sample = combined_data.loc[[bad_sample_index]]
    ordered_data = kmeans_clustering.calculate_distance_per_cluster(
        best_cluster, sample, class_label
    )
    plot_name = f"distance_plot_{best_cluster}_k{best_k}.png"
    kmeans_clustering.generate_distance_plot(ordered_data, class_label, best_cluster, plot_name)

    # --- START: AHS Stage 1 (Feasibility Check) ---
    final_clustered_df, final_label_counts = kmeans_clustering.compute_kmeans(k=best_k)
    cluster_counts = final_label_counts.get(best_cluster, {})
    good_samples_in_cluster = cluster_counts.get(desired_label, 0)

    if good_samples_in_cluster > 0:
        # --- STRATEGY 1: VIABLE CLUSTER (Run as normal) ---
        print(f"\n: Cluster {best_cluster} is viable ({good_samples_in_cluster} good samples found).")
        print(": Running standard constrained search...")

        cf_gen_cluster = CFGenerator(
            CFProblem,
            model,
            data_wo_label,
            clustered_dataset_name,
            "kmeans",
            best_new_min,
            best_new_max,
            seed
        )

        print(f": Generating counterfactuals for sample {bad_sample_index}...")
        cf_gen_cluster.generate_counterfactuals(
            desired_label,
            bad_sample_index
        )

    else:
        # --- STRATEGY 2: EMPTY-CLASS CLUSTER (Trigger Fallback) ---
        print(f"\n: WARNING: Cluster {best_cluster} is an 'Empty-Class Cluster'.")
        print(f": No good samples found. Initiating Stage 2 Fallback Strategy...")

        try:
            # 1. Inter-Cluster Seeding
            all_cluster_labels = sorted(list(final_label_counts.keys()))

            # [MODIFIED] Call Helper method
            fallback_cluster_name = Helper.find_nearest_viable_cluster(
                best_cluster,
                kmeans_clustering.kmeans.cluster_centers_,
                all_cluster_labels,
                final_label_counts,
                desired_label
            )

            # 2. Target Identification & Boundary Definition
            # [MODIFIED] Call Helper method
            fallback_min, fallback_max = Helper.find_seed_and_boundaries(
                fallback_cluster_name,
                final_clustered_df,
                data_wo_label,
                bad_sample_index,
                desired_label,
                class_label
            )

            print(":   New search boundaries defined. Running guided optimization...")
            fallback_dataset_name = clustered_dataset_name + "_fallback"

            # 3. Guided Optimization
            cf_gen_fallback = CFGenerator(
                CFProblem,
                model,
                data_wo_label,
                fallback_dataset_name,
                "kmeans_fallback",
                fallback_min,
                fallback_max,
                seed
            )

            print(f":   Generating counterfactuals for sample {bad_sample_index}...")
            cf_gen_fallback.generate_counterfactuals(
                desired_label,
                bad_sample_index
            )

        except Exception as e:
            print(f":   FALLBACK FAILED: {e}")
    # --- END: AHS Logic ---

    stop = timeit.default_timer()
    time_minutes = (stop - start) / 60
    print(f"\nTotal time: {time_minutes:.4f} minutes")


if __name__ == "__main__":
    main()