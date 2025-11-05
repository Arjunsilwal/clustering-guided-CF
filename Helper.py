# last upated: 3/19/2025
# helper.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter


class Helper:
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
    def plot_scatter(problem, sample_idx, seed, cluster_mode, k=None):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            problem.all_f1,
            problem.all_f2,
            alpha=0.7,
            edgecolors="k",
            label="Objective Values",
        )
        plt.scatter(
            problem.max_error,
            problem.max_distance,
            color="red",
            label="Reference Point",
        )
        plt.title(
            f"Scatter plot of f1 and f2 {cluster_mode} (Sample: {sample_idx}, Seed: {seed})"
        )
        plt.xlabel("f1 (Error)")
        plt.ylabel("f2 (Distance)")
        plt.legend(loc="center", bbox_to_anchor=(0.5, 0.5))
        plt.grid(True)
        if not os.path.exists("scatter_plots_files"):
            os.makedirs("scatter_plots_files")

        if k is not None:
            plot_path = os.path.join(
                "scatter_plots_files",
                f"scatter_f1_vs_f2_{cluster_mode}_{sample_idx}_seed{seed}_k={k}.png",
            )
        else:
            plot_path = os.path.join(
                "scatter_plots_files",
                f"scatter_f1_vs_f2_{cluster_mode}_{sample_idx}_seed{seed}.png",
            )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot is saved to {plot_path}")
        plt.close()

    @staticmethod
    def plot_pareto_front(
        res, extracted_data_name, cluster_mode, pop_size, sample_idx, seed
    ):
        if not os.path.exists("pareto"):
            os.makedirs("pareto")
        # pareto_plot = Scatter()
        # pareto_plot.add(res.F, color="red")
        pareto_front_df = pd.DataFrame(res.F, columns=["Error", "Distance"])
        pareto_csv_filename = f"pareto/pareto_front_{extracted_data_name}_{cluster_mode}_{pop_size}_{sample_idx}_seed{seed}.csv"
        pareto_front_df.to_csv(pareto_csv_filename, index=False)

        # save the pareto front plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pareto_front_df["Error"],
            pareto_front_df["Distance"],
            c="red",
            edgecolors="k",
            alpha=0.7,
        )
        plt.xlabel("Error (f1)", fontsize=14)
        plt.ylabel("Distance (f2)", fontsize=14)
        plt.title(
            f"Pareto Front {cluster_mode} (Sample: {sample_idx}, Seed: {seed})",
            fontsize=16,
        )
        plt.grid(True)
        plt.tight_layout()
        pareto_plot_filename = f"pareto/pareto_front_{extracted_data_name}_{cluster_mode}_{pop_size}_{sample_idx}_seed{seed}.png"
        plt.savefig(pareto_plot_filename, dpi=300)
        plt.close()

        print("Pareto front plot is saved.")

    @staticmethod
    def log_results(
        genomes,
        col_names,
        extracted_data_name,
        cluster_mode,
        pop_size,
        data_wo_label,
        sample_idx,
        seed,
    ):
        folder_name = "nsga_results"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        txt_file_path = os.path.join(
            folder_name,
            f"{extracted_data_name}_{cluster_mode}_{pop_size}_{sample_idx}_seed{seed}_nsga_results.txt",
        )
        if not os.path.exists(txt_file_path):
            with open(txt_file_path, "w") as file:
                sample = data_wo_label.loc[sample_idx]
                val_dict = dict(zip(col_names, sample))
                file.write(f"Original Sample (Index: {sample_idx}): {val_dict}\n")
                for i, genome in enumerate(genomes):
                    genome_dict = dict(zip(col_names, genome))
                    row_str = ", ".join(
                        [f"{key}: {value}" for key, value in genome_dict.items()]
                    )
                    file.write(f"{i} [{row_str}]\n")

    @staticmethod
    def plot_avg_hv(avg_hv, sample_idx, cluster_mode):
        plt.figure(figsize=(8, 6))
        generations = range(1, len(avg_hv) + 1)
        plt.plot(generations, avg_hv, color="black")
        plt.title(f"AVG HV Convergence {cluster_mode} (Sample: {sample_idx})")
        plt.xlabel("Generation")
        plt.ylabel("Average Hypervolume")
        plt.grid(True)
        filename = f"avg_hv_sample_{sample_idx}_{cluster_mode}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def log_avg_hv_per_row(samples_hv_avg, extracted_data_name, cluster_mode):
        hv_df = pd.DataFrame(samples_hv_avg)
        hv_csv_filename = f"{extracted_data_name}_{cluster_mode}_avg_hv_by_sample.csv"
        # check if the file already exists, if it does not, write the header
        write_header = not os.path.exists(hv_csv_filename)
        # append the data if the file exists
        hv_df.to_csv(hv_csv_filename, mode="a", header=write_header, index=False)
        # hv_df.to_csv(hv_csv_filename, index=False)
        print(
            f"Average HV per generation for each sample is saved to {hv_csv_filename}"
        )

    @staticmethod
    def plot_combined_pareto_front(
        sample_idx,
        seed,
        res_without,
        res_with,
        extracted_data_name,
        cluster_mode,
        k=None,
    ):
        plt.figure(figsize=(8, 6))
        # plot without clustering (red)
        df_without = pd.DataFrame(res_without, columns=["Error", "Distance"])
        plt.scatter(
            df_without["Error"],
            df_without["Distance"],
            c="red",
            edgecolors="k",
            alpha=0.6,
            label="Without Clustering",
        )
        # plot with clustering (blue)
        df_with = pd.DataFrame(res_with, columns=["Error", "Distance"])
        plt.scatter(
            df_with["Error"],
            df_with["Distance"],
            c="blue",
            edgecolors="k",
            alpha=0.6,
            label=cluster_mode,
        )

        plt.title(f"Pareto Front Comparison (Sample: {sample_idx}, Seed: {seed})")
        plt.xlabel("Error (f1)")
        plt.ylabel("Distance (f2)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists("pareto_combined"):
            os.makedirs("pareto_combined")

        if k is not None:
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

        if not os.path.exists("avg_hv_combined"):
            os.makedirs("avg_hv_combined")

        if k is not None:
            plt.savefig(
                f"avg_hv_combined/avg_hv_combined_sample_{sample_idx}_k={k}.png",
                dpi=300,
            )
        else:
            plt.savefig(
                f"avg_hv_combined/avg_hv_combined_sample_{sample_idx}.png", dpi=300
            )
        plt.close()
