# model_trainer.py

import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline


class ModelTrainer:
    def __init__(self, dataset, target, class_label, dataset_name, model_choice=None):
        self.dataset = dataset
        self.target = target
        self.class_label = class_label
        self.dataset_name = dataset_name
        self.model_choice = model_choice
        self.results = []

        # dictionary of models to choose
        self.available_models = {
            "KNN": lambda: KNeighborsClassifier(n_neighbors=3),
            "DTC": lambda: DecisionTreeClassifier(max_depth=25, random_state=42),
            "SVC": lambda: make_pipeline(
                MinMaxScaler(), SVC(kernel="rbf", probability=True)
            ),
            "MLP": lambda: make_pipeline(
                MinMaxScaler(), MLPClassifier(random_state=42)
            ),
            "NBC": lambda: GaussianNB(),
        }

    def train_model(self):
        X = self.dataset
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # choose the model from the model options
        if isinstance(self.model_choice, str):
            if self.model_choice in self.available_models:
                model = self.available_models[self.model_choice]()
            else:
                raise ValueError(
                    f"Model '{self.model_choice}' is not available. Choose from: {list(self.available_models.keys())}"
                )
        elif callable(self.model_choice):
            # if a callable is provided, it is expected to return a model instance
            model = self.model_choice()
        elif self.model_choice is not None:
            # if model is already instantiated
            model = self.model_choice
        else:
            # default model if nothing is specified
            model = self.available_models["MLP"]()
        # train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.results.append(
            [self.dataset_name, self.class_label, type(model).__name__, accuracy]
        )
        print(
            f"Trained {type(model).__name__} on {self.dataset_name} with accuracy: {accuracy}"
        )

        return model

    def get_results(self):
        return self.results


# class ModelTrainer:
#     def __init__(self, model_filename: str):
#         self.model_filename = model_filename

#     def train_and_save_model(self, data, target):
#         X = data
#         y = target
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )
#         model = Pipeline(
#             [("scaler", MinMaxScaler()), ("classifier", MLPClassifier(random_state=42))]
#         )

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"Accuracy: {accuracy}")

#         with open(self.model_filename, "wb") as file:
#             pickle.dump(model, file)
#         print(f"Model saved as {self.model_filename}")
#         return model

#     def load_model(self, data, target):
#         if os.path.exists(self.model_filename):
#             with open(self.model_filename, "rb") as file:
#                 model = pickle.load(file)
#             print(f"Existing model loaded from {self.model_filename}")

#         else:
#             print(
#                 f"Model file {self.model_filename} does not exist. Training a new model."
#             )
#             model = self.train_and_save_model(data, target)
#         return model
