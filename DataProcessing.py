# last updated: 3/19/2025
# data_processing.py
import pandas as pd


class DataProcessing:
    def __init__(self, file_name: str, class_label: str, mapping_values: dict = None):
        self.file_name = file_name
        self.class_label = class_label
        self.mapping_values = mapping_values

    def load_and_process(self):
        data = pd.read_csv(self.file_name)
        if self.mapping_values:
            data[self.class_label] = data[self.class_label].replace(self.mapping_values)
            data = data.replace(self.mapping_values)

        # separate target and features
        target = data[self.class_label]
        data_wo_label = data.drop(columns=[self.class_label])

        col_list = data_wo_label.columns.tolist()
        min_values = data_wo_label[col_list].min()
        max_values = data_wo_label[col_list].max()

        return data_wo_label, target, min_values, max_values
