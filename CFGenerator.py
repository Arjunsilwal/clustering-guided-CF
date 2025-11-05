# last updated: 4/6/2025
# cf_generator.py
import timeit
import random
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from Helper import Helper


class CFGenerator:
    def __init__(
        self,
        problem_class,
        model,
        data_wo_label,
        extracted_data_name,
        cluster_mode,
        min_values,
        max_values,
        seed,
    ):
        self.problem_class = problem_class
        self.model = model
        self.data_wo_label = data_wo_label
        self.feature_names = data_wo_label.columns.tolist()
        self.extracted_data_name = extracted_data_name
        self.cluster_mode = cluster_mode
        self.min_values = min_values
        self.max_values = max_values
        self.seed = seed

    def run_optimization(self, problem, pop_size, num_gen, sample_idx):
        random_seed = self.seed
        all_hist_X = []
        all_hist_F = []
        algorithm = NSGA2(pop_size=pop_size)
        res = minimize(
            problem,
            algorithm,
            ("n_gen", num_gen),
            seed=random_seed,
            save_history=True,
            verbose=False,
        )

        Helper.plot_scatter(problem, sample_idx, random_seed, self.cluster_mode)
        col_names = self.data_wo_label.columns.tolist()

        Helper.log_results(
            res.opt.get("X"),
            col_names,
            self.extracted_data_name,
            self.cluster_mode,
            pop_size,
            self.data_wo_label,
            sample_idx,
            random_seed,
        )

        hist_F, hist_X = [], []
        for algo in res.history:
            opt = algo.opt
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
            hist_X.append(opt.get("X")[feas])

        approx_ideal = res.F.min(axis=0)
        approx_nadir = res.F.max(axis=0)
        buffer = 0.1
        ref_point = np.array(
            [problem.max_error + buffer, problem.max_distance + buffer]
        )
        metric = Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            ideal=approx_ideal,
            nadir=approx_nadir,
        )
        hv = [metric.do(F_gen) for F_gen in hist_F]

        # hv_results.append(hv)
        all_hist_X.append(hist_X)
        all_hist_F.append(hist_F)

        # hv_results = np.array(hv_results)
        # avg_hv_across_seeds = np.mean(hv_results, axis=0)

        # return all_hist_X, all_hist_F, hv
        return res.F, hv

    def generate_counterfactuals(
        self,
        data_wo_label,
        desired_label,
        sample_idx,
    ):

        sample = data_wo_label.loc[sample_idx].values  # convert to 2D array
        # samples_hv_avg = []

        problem = self.problem_class(
            self.model,
            feature_names=self.feature_names,
            original_sample=sample,
            sample_idx=sample_idx,
            desired_class=desired_label,
            min_values=self.min_values,
            max_values=self.max_values,
        )
        pop_size = 100
        num_gen = 50
        pareto_F, hv = self.run_optimization(problem, pop_size, num_gen, sample_idx)

        return pareto_F, hv
