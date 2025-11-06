# last updated: 4/12/2025
# main.py
# this script is updated version of main with knn clustering
# [MODIFIED] Now includes AHS Fallback Logic

import timeit
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist  # [NEW]
from DataProcessing import DataProcessing
from ModelTrainer import ModelTrainer
from CFGenerator import CFGenerator
from CFProblem import CFProblem
from HierarchicalClustering import HierarchicalClustering, LocalKNNSelection
from Helper import Helper # [MODIFIED] Import Helper class

# from KMeansClustering import KMeansClustering
import random

# good is 0
# bad is 1

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from sklearn.preprocessing import MinMaxScaler


# --- This radar_factory function is from the original file, leaving it as-is ---
def radar_factory(num_vars, frame="circle"):
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit radii.
            vertices = path.vertices
            if len(vertices) == 2 and vertices[0, 0] == 0.0 and vertices[1, 0] == 1.0:
                return Path(
                    vertices[1:, :], [Path.MOVETO, Path.LINETO] + path.codes[2:]
                )
            else:
                return path

        transform_path = transform_path_non_affine

        def transform_non_affine(self, points):
            points = np.asarray(points)
            north = points[:, 0]
            east = points[:, 1]
            # print(points)
            if east.max() > 1.0:
                # scale the east axis to fit in the new radius
                east = east / east.max()
            points = np.stack([north, east], axis=1)
            return points

        transform = transform_non_affine

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line segment to connect the points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            # Alter the args and kwargs as necessary for polar plots.
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            # Alter the args and kwargs as necessary for polar plots.
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0]
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0)
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta, RadarAxes
# --- End of radar_factory function ---


# [REMOVED] This function is now in Helper.py
# def find_seed_and_boundaries_from_sample(original_instance_features, seed_instance_features):
#   ...


def main():
    start = timeit.default_timer()
    file_name = "breast_cancer.csv"
    extracted_data_name = file_name.split(".")[0]
    class_label = "Diagnosis"
    mapping_values = {"M": 1, "B": 0}

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
    cluster_method = "knn"  # [NEW]

    # cf generation without clustering
    print(f"Original min: {min_values}, Original max: {max_values}")

    bad_samples = combined_data[combined_data[class_label] == original_label]
    if bad_samples.empty:
        raise ValueError("No bad samples found in the dataset.")

    # [MODIFIED] Using the same bad sample as other scripts
    # bad_sample_index = random.choice(bad_samples.index.tolist())
    bad_sample_index = 368
    if bad_sample_index not in bad_samples.index:
        print(f"Warning: Sample {bad_sample_index} is not a 'bad' sample. Choosing a random one.")
        bad_sample_index = random.choice(bad_samples.index.tolist())

    print(f"bad_sample_index: {bad_sample_index}")

    # --- This section is commented out in the original, keeping it that way ---
    # cf_generator_without = CFGenerator(
    # ...
    # )
    # hv_without_results = []
    # -------------------------------------------------------------------------

    hv_with_results = []
    num_k = 5  # number of neighbors to consider
    print(f"\n: --- Starting KNN Bound Strategy (k={num_k}) ---")

    knn_selector = LocalKNNSelection(
        combined_data, class_label, bad_sample_index
    )
    (
        knn_min_values,
        knn_max_values,
        knn_data_with_labels,  # This is our "cluster"
    ) = knn_selector.find_knn_boundaries(k=num_k)

    # --- START: AHS Stage 1 (Feasibility Check) ---
    good_samples_in_knn = knn_data_with_labels[
        knn_data_with_labels[class_label] == desired_label
        ].shape[0]

    clustered_dataset_name = extracted_data_name + f"_knn_k{num_k}"

    # We must use the same seeds for a fair comparison (if we were comparing to "without")
    random_seeds = [42]  # Using a single seed

    try:
        if good_samples_in_knn > 0:
            # --- STRATEGY 1: VIABLE NEIGHBORHOOD (Run as normal) ---
            print(f"\n: KNN neighborhood is viable ({good_samples_in_knn} good samples found).")
            print(": Running standard constrained search...")

            search_min = knn_min_values
            search_max = knn_max_values
            search_cluster_mode = cluster_method
            search_dataset_name = clustered_dataset_name

        else:
            # --- STRATEGY 2: EMPTY-CLASS NEIGHBORHOOD (Trigger Fallback) ---
            print(f"\n: WARNING: KNN neighborhood (k={num_k}) is an 'Empty-Class Cluster'.")
            print(": No good samples found. Initiating Stage 2 Fallback Strategy...")

            # 1. Find nearest viable sample (skipping the first k)
            seed_instance_features = knn_selector.find_next_viable_seed(
                k_to_skip=num_k,
                desired_label=desired_label
            )
            print(f":   Nearest viable seed instance found (original index: {seed_instance_features.index.values})")

            # 2. Target Identification & Boundary Definition
            original_instance_features = data_wo_label.loc[[bad_sample_index]]

            # [MODIFIED] Call the centralized function from the Helper class
            fallback_min, fallback_max = Helper.find_seed_and_boundaries_from_sample(
                original_instance_features,
                seed_instance_features
            )
            print(":   New search boundaries defined. Running guided optimization...")

            # 3. Set variables for the generator
            search_min = fallback_min
            search_max = fallback_max
            search_cluster_mode = cluster_method + "_fallback"
            search_dataset_name = clustered_dataset_name + "_fallback"

        # --- END AHS Check ---

        # --- Run the CF Generator using the selected strategy ---

        for seed in random_seeds:
            # [MODIFIED] This call now uses the variables set in the if/else block
            cf_gen_cluster = CFGenerator(
                CFProblem,
                model,
                data_wo_label,
                search_dataset_name,
                search_cluster_mode,
                search_min,
                search_max,
                seed,
            )

            print(f":   Generating counterfactuals for sample {bad_sample_index} (Seed: {seed})...")
            pareto_F_knn, hv_knn = cf_gen_cluster.generate_counterfactuals(
                desired_label,
                bad_sample_index
            )
            hv_with_results.append(hv_knn)

            # --- This plotting is for comparison, which we are not running ---
            # For visualization comparison
            # Helper.plot_combined_pareto_front(
            #     sample_idx=bad_sample_index,
            #     seed=seed,
            #     res_without=pareto_F_without, # This is not defined
            #     res_with=pareto_F_knn,
            #     extracted_data_name=extracted_data_name,
            #     cluster_mode=cluster_method,
            #     k=num_k,
            # )
            # --------------------------------------------------------------

        # --- The rest of this is for comparison plotting, which we can skip ---
        # hv_without_results = np.array(hv_without_results)
        # avg_hv_without = np.mean(hv_without_results, axis=0)

        # hv_with_results = np.array(hv_with_results)
        # avg_hv_with = np.mean(hv_with_results, axis=0)

        # Helper.plot_combined_avg_hv(
        #     avg_hv_without=avg_hv_without,
        #     avg_hv_with=avg_hv_with,
        #     sample_idx=bad_sample_index,
        #     cluster_mode=cluster_method,
        #     k=num_k,
        # )

        # Helper.log_avg_hv_per_row(
        #     avg_hv_without, extracted_data_name, cluster_mode="without_cluster"
        # )
        # Helper.log_avg_hv_per_row(
        #     avg_hv_with, extracted_data_name, cluster_mode=cluster_method, k=num_k
        # )
        # ---------------------------------------------------------------------

    except Exception as e:
        print(f":   KNN-BOUND STRATEGY FAILED: {e}")

    stop = timeit.default_timer()
    time_minutes = (stop - start) / 60
    print(f"\nTotal time: {time_minutes:.4f} minutes")


if __name__ == "__main__":
    main()