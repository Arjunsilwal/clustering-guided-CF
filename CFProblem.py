# last updated: 3/19/2025
# problem.py

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem


class CFProblem(Problem):
    def __init__(
        self,
        model,
        feature_names,
        original_sample,
        sample_idx,
        desired_class=None,
        min_values=None,
        max_values=None,
        **kwargs,
    ):
        # to check the original sample exclude the label column
        n_features = len(original_sample)
        print(f"n_features: {n_features}")
        super().__init__(n_var=n_features, n_obj=2, n_constr=0, **kwargs)
        self.model = model
        self.feature_names = feature_names
        self.original_sample = original_sample
        self.sample_idx = sample_idx
        self.desired_class = desired_class
        self.min_values = min_values
        self.max_values = max_values

        self.max_error = -np.inf
        self.max_distance = -np.inf
        self.all_f1 = []
        self.all_f2 = []

        # set feature bounds
        self.xl = self.min_values.values
        self.xu = self.max_values.values

    def _evaluate(self, genomes, out, *args, **kwargs):
        error_values = []
        distance_values = []
        self.all_genomes = []

        for genome in genomes:
            genome = np.round(genome, 3)
            self.all_genomes.append(genome)
            genome_df = pd.DataFrame(genome.reshape(1, -1), columns=self.feature_names)
            # print(f"genome_df: {genome_df}")
            probabilities = self.model.predict_proba(genome_df)
            all_classes = list(self.model.classes_)
            prob_target_class = probabilities[:, all_classes.index(self.desired_class)]
            error = 1 - prob_target_class
            error_values.append(float(error[0]))

            distance = np.linalg.norm(genome - self.original_sample)
            distance_values.append(distance)

        # normalize objectives
        min_error, max_error = min(error_values), max(error_values)
        normalized_error = [
            (
                (error - min_error) / (max_error - min_error)
                if max_error != min_error
                else 0
            )  # prevent zero division
            for error in error_values
        ]
        min_distance, max_distance = min(distance_values), max(distance_values)
        normalized_distance = [
            (
                (dist - min_distance) / (max_distance - min_distance)
                if max_distance != min_distance
                else 0
            )
            for dist in distance_values
        ]

        self.max_error = max(normalized_error)
        self.max_distance = max(normalized_distance)
        self.all_f1.extend(normalized_error)
        self.all_f2.extend(normalized_distance)

        out["F"] = np.column_stack([normalized_error, normalized_distance])
        # print(f"out['F']: {out['F']}")
