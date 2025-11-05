# last updated: 4/6/2025
# main.py
# this script is updated version of main with hierarchical clustering
# I need to check if the bad sample is always able to get good samples in the same cluster
# it seems like it may not be actually having any good samples in the same cluster, then the updated range would be exactly the same as the previous approach without clustering

import timeit
import pandas as pd
import numpy as np
from DataProcessing import DataProcessing
from ModelTrainer import ModelTrainer
from CFGenerator import CFGenerator
from CFProblem import CFProblem
from HierarchicalClustering import HierarchicalClustering, LocalKNNSelection
from Helper import Helper

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


def radar_factory(num_vars, frame="circle"):
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
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
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def generate_spider_plot(title, data, col_names, sample_idx, cluster_method, k=None):
    N = len(col_names)
    theta = radar_factory(N, frame="polygon")

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    colors = ["b", "r", "g"]  # for: sample, upper bound, lower bound
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(
        title,
        weight="bold",
        size="medium",
        position=(0.5, 1.1),
        horizontalalignment="center",
        verticalalignment="center",
    )

    for d, color in zip(data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")

    ax.set_varlabels(col_names)

    # Add legend
    labels = ("Sample", "Updated Upper Bound", "Updated Lower Bound")
    ax.legend(labels, loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize="small")

    # plt.show()
    if k is not None:
        spiderplot_filename = f"spiderplot_{sample_idx}_{cluster_method}_k={k}.png"
    else:
        spiderplot_filename = f"spiderplot_{sample_idx}_{cluster_method}.png"

    plt.tight_layout()
    plt.savefig(spiderplot_filename, dpi=300)
    plt.close()


def main():
    start = timeit.default_timer()
    shared_seeds = np.random.randint(0, 10000, size=10).tolist()
    file_name = "breast_cancer.csv"
    extracted_data_name = file_name.split(".")[0]
    class_label = "Diagnosis"
    mapping_values = {"M": 1, "B": 0}
    data_processing = DataProcessing(file_name, class_label, mapping_values)
    data_wo_label, target, min_values, max_values = data_processing.load_and_process()
    print("data wo label shape:", data_wo_label.shape)
    print("target shape:", target.shape)
    # combined_data = pd.concat([data_wo_label, target], axis=1)

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

    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(
        scaler.fit_transform(data_wo_label),
        columns=data_wo_label.columns,
        index=data_wo_label.index,
    )
    combined_data = pd.concat([normalized_df, target], axis=1)

    # overwrite the original min and max values with the normalized values
    col_list = data_wo_label.columns.tolist()
    min_values = normalized_df[col_list].min()
    max_values = normalized_df[col_list].max()
    # cf generation without clustering
    # print(f"Original min: {min_values}, Original max: {max_values}")
    cluster_method = "hierarchical"
    bad_samples = combined_data[combined_data[class_label] == original_label]
    if bad_samples.empty:
        raise ValueError("No bad samples found in the dataset.")
    bad_sample_indices_list = bad_samples.sample(n=5, random_state=42).index.tolist()
    print(f"bad_sample_indices_list: {bad_sample_indices_list}")
    # bad_sample_index = random.choice(bad_samples.index.tolist())
    # print(f"bad_sample_index: {bad_sample_index}")

    for bad_sample_index in bad_sample_indices_list:
        hv_without_results = []
        hv_with_results = []
        for seed in shared_seeds:
            print(f"bad_sample_index: {bad_sample_index}")
            col_names = normalized_df.columns
            print(f"col_names: {col_names}")
            bad_sample = normalized_df.loc[bad_sample_index]
            print(f"bad_sample: {bad_sample}")
            # cf generation without clustering
            # cf_generator = CFGenerator(
            #     problem_class=CFProblem,
            #     model=model,
            #     data_wo_label=data_wo_label,
            #     extracted_data_name=extracted_data_name,
            #     cluster_mode="without_cluster",
            #     min_values=min_values,
            #     max_values=max_values,
            #     seed=seed,
            # )

            # pareto_F_without, hv_without = cf_generator.generate_counterfactuals(
            #     data_wo_label,
            #     desired_label,
            #     sample_idx=bad_sample_index,
            # )

            # run hierarchical clustering to find the optimal cluster and update the min and max boundaries
            hc_search = HierarchicalClustering(
                combined_data,
                original_label,
                desired_label,
                label_col=class_label,
                bad_sample_idx=bad_sample_index,
            )
            optimal_cluster, new_min_boundaries, new_max_boundaries = (
                hc_search.run_analysis()
            )
            print(f"Optimal cluster: {optimal_cluster}")
            print(f"New min: {new_min_boundaries}, New max: {new_max_boundaries}")

            # generate the spider plot for the bad sample, upper bound, and lower bound
            bad_sample = bad_sample.tolist()
            new_max_boundaries = new_max_boundaries.tolist()
            new_min_boundaries = new_min_boundaries.tolist()
            data = [bad_sample, new_max_boundaries, new_min_boundaries]
            col_names = data_wo_label.columns.tolist()
            title = f"Sample {bad_sample_index} Spider Plot for Hierarchical Clustering"
            generate_spider_plot(
                title,
                data,
                col_names,
                bad_sample_index,
                cluster_method=cluster_method,
            )
        # ------------------------------only before this for now
        # cf generation with clustering
    #         cf_gen_cluster = CFGenerator(
    #             problem_class=CFProblem,
    #             model=model,
    #             data_wo_label=data_wo_label,
    #             extracted_data_name=extracted_data_name,
    #             cluster_mode="with_cluster",
    #             min_values=new_min_boundaries,
    #             max_values=new_max_boundaries,
    #             seed=seed,
    #         )

    #         pareto_F_with, hv_with = cf_gen_cluster.generate_counterfactuals(
    #             data_wo_label,
    #             desired_label,
    #             sample_idx=bad_sample_index,
    #         )

    #         # plot the pareto front for both without clustering and with clustering
    #         Helper.plot_combined_pareto_front(
    #             sample_idx=bad_sample_index,
    #             seed=seed,
    #             res_without=pareto_F_without,
    #             res_with=pareto_F_with,
    #             extracted_data_name=extracted_data_name,
    #         )

    #     hv_without_results.append(hv_without)
    #     hv_without_results = np.array(hv_without_results)
    #     avg_hv_without = np.mean(hv_without_results, axis=0)

    #     hv_with_results.append(hv_with)
    #     hv_with_results = np.array(hv_with_results)
    #     avg_hv_with = np.mean(hv_with_results, axis=0)

    #     # save the average hv for both without clustering and with clustering
    #     Helper.plot_combined_avg_hv(
    #         avg_hv_without=avg_hv_without,
    #         avg_hv_with=avg_hv_with,
    #         sample_idx=bad_sample_index,
    #     )

    # Helper.log_avg_hv_per_row(
    #     avg_hv_without, extracted_data_name, cluster_mode="without_cluster"
    # )
    # Helper.log_avg_hv_per_row(
    #     avg_hv_with, extracted_data_name, cluster_mode="with_cluster"
    # )

    stop = timeit.default_timer()
    time_minutes = (stop - start) / 60
    print(f"Total time: {time_minutes} minutes")


if __name__ == "__main__":
    main()
