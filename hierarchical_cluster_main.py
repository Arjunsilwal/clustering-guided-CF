# last updated: 4/6/2025
# main.py
# this script is updated version of main with hierarchical clustering
# [MODIFIED] Now includes AHS Fallback Logic

import timeit
import pandas as pd
import numpy as np
# [REMOVED] cdist is no longer needed here
from DataProcessing import DataProcessing
from ModelTrainer import ModelTrainer
from CFGenerator import CFGenerator
from CFProblem import CFProblem
from HierarchicalClustering import HierarchicalClustering
from Helper import Helper  # [MODIFIED] Import Helper class

import random

# good is 0
# bad is 1

# [REMOVED] All helper functions moved to Helper.py
# def find_nearest_viable_sample(...):


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

    # [MODIFIED] Pass correct data to trainer
    model_trainer = ModelTrainer(
        dataset=data_wo_label,  # Use data_wo_label
        target=target,  # Use target
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

    cluster_method = "hierarchical"

    # --- Hierarchical Clustering ---
    print("\n: --- Starting Hierarchical Clustering ---")
    h_cluster = HierarchicalClustering(
        combined_data,
        original_label,
        desired_label,
        class_label,
        bad_sample_index,
    )
    h_cluster.perform_clustering(method="ward")

    (
        best_cluster_data,
        best_new_min,
        best_new_max,
    ) = h_cluster.find_optimal_cluster()
    # --- END Hierarchical Clustering ---

    # --- START: AHS Stage 1 (Feasibility Check) ---
    good_samples_in_cluster = best_cluster_data[
        best_cluster_data[class_label] == desired_label
        ].shape[0]

    clustered_dataset_name = extracted_data_name + f"_hierarchical_n{h_cluster.best_n_clusters}"

    if good_samples_in_cluster > 0:
        # --- STRATEGY 1: VIABLE CLUSTER (Run as normal) ---
        print(f"\n: Cluster {h_cluster.best_cluster_label} is viable ({good_samples_in_cluster} good samples found).")
        print(": Running standard constrained search...")

        cf_gen_cluster = CFGenerator(
            CFProblem,
            model,
            data_wo_label,
            clustered_dataset_name,
            cluster_method,
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
        print(f"\n: WARNING: Cluster {h_cluster.best_cluster_label} is an 'Empty-Class Cluster'.")
        print(": No good samples found. Initiating Stage 2 Fallback Strategy...")

        try:
            # 1. Find nearest viable sample
            # [MODIFIED] Call Helper method
            seed_instance_features = Helper.find_nearest_viable_sample(
                data_wo_label,
                combined_data,
                bad_sample_index,
                best_cluster_data,
                desired_label,
                class_label
            )

            # 2. Target Identification & Boundary Definition
            original_instance_features = data_wo_label.loc[[bad_sample_index]]

            # [MODIFIED] Call the centralized function from the Helper class
            fallback_min, fallback_max = Helper.find_seed_and_boundaries_from_sample(
                original_instance_features,
                seed_instance_features
            )

            print(":   New search boundaries defined. Running guided optimization...")
            fallback_dataset_name = clustered_dataset_name + "_fallback"

            # 3. Guided Optimization
            cf_gen_fallback = CFGenerator(
                CFProblem,
                model,
                data_wo_label,
                fallback_dataset_name,
                cluster_method + "_fallback",
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