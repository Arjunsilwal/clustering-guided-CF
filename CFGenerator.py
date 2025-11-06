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
        hist = res.history
        # print(hist)
        hist_X = [e.pop.get("X") for e in hist]
        hist_F = [e.pop.get("F") for e in hist]

        # plot pareto front
        # [MODIFIED] Pass 'problem' object to get all_f1/all_f2
        Helper.plot_scatter(
            res.F, problem, sample_idx, self.seed, self.cluster_mode
        )
        Helper.save_hist_FX(
            hist_F,
            hist_X,
            self.extracted_data_name,
            self.cluster_mode,
            pop_size,
            sample_idx,
            self.data_wo_label,
        )

        # get hypervolume
        # ref_point = np.array([1.0, 1.0])
        # approx_ideal = res.F.min(axis=0)
        # approx_nadir = res.F.max(axis=0)

        # [MODIFIED] Use the max error/distance tracked by the problem
        approx_ideal = np.array([0.0, 0.0])
        approx_nadir = np.array([problem.max_error, problem.max_distance])

        metric = Hypervolume(
            ref_point=np.array([1.0, 1.0]),  # Use a fixed [1, 1] ref_point for normalized data
            norm_ref_point=False,
            zero_to_one=True,
            ideal=approx_ideal,
            nadir=approx_nadir,
        )
        hv = [metric.do(F_gen) for F_gen in hist_F]

        all_hist_X.append(hist_X)
        all_hist_F.append(hist_F)

        return res.F, hv

    def generate_counterfactuals(
            self,
            desired_label,
            sample_idx,
    ):
        sample = self.data_wo_label.loc[sample_idx].values

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

        print(f":   Final Pareto front generated with {len(pareto_F)} solutions.")
        print(f":   Final Hypervolume: {hv[-1]:.4f}")

        # [FIX] Add the return statement that was missing
        return pareto_F, hv