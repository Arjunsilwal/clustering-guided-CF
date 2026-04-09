# Clustering-Guided Counterfactual Explanation Generator

A framework for generating **counterfactual explanations** for machine learning classifiers using clustering-guided search strategies. Given an instance classified as undesirable (e.g., a malignant tumor diagnosis), the system searches for the minimal feature changes needed to flip the model's prediction to a desired outcome (e.g., benign).

The core search is powered by **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), a multi-objective evolutionary algorithm that simultaneously minimizes:
- **F1 — Prediction Error**: how far the counterfactual is from the desired class
- **F2 — Feature Distance**: how different the counterfactual is from the original instance

Clustering strategies are used to constrain the search space to semantically meaningful regions of the feature space, improving the quality and plausibility of generated counterfactuals.

---

## Project Structure

```
clustering-guided-CF/
├── breast_cancer.csv              # Example dataset (Wisconsin Breast Cancer)
├── breast_cancer_main.py          # Entry point: K-Means guided CF generation
├── hierarchical_cluster_main.py   # Entry point: Hierarchical clustering guided CF generation
├── knn_bound_main.py              # Entry point: KNN-boundary guided CF generation
├── CFGenerator.py                 # Runs NSGA-II optimization and saves results
├── CFProblem.py                   # Multi-objective problem definition for pymoo
├── DataProcessing.py              # CSV loading and preprocessing
├── ModelTrainer.py                # Trains and evaluates classifiers
├── KMeansClustering.py            # K-Means clustering and boundary extraction
├── HierarchicalClustering.py      # Hierarchical clustering and KNN boundary selection
└── Helper.py                      # Shared utilities: plotting, logging, fallback strategies
```

---

## How It Works

### 1. Data Loading & Preprocessing (`DataProcessing`)
Loads a CSV dataset, optionally maps class labels to integers (e.g., `"M" → 1`, `"B" → 0`), and extracts per-feature min/max bounds for the optimizer.

### 2. Model Training (`ModelTrainer`)
Trains a binary classifier on the dataset. Supported models:

| Key   | Model                          |
|-------|--------------------------------|
| `KNN` | K-Nearest Neighbors (k=3)      |
| `DTC` | Decision Tree                  |
| `SVC` | Support Vector Machine (RBF)   |
| `MLP` | Multi-Layer Perceptron         |
| `NBC` | Gaussian Naïve Bayes           |

### 3. Counterfactual Problem Definition (`CFProblem`)
Defines the optimization problem for `pymoo`. For each candidate solution (genome), it:
- Queries the trained model's predicted probability for the desired class
- Computes Euclidean distance from the original instance
- Returns normalized `[error, distance]` objective values

### 4. CF Generation via NSGA-II (`CFGenerator`)
Runs NSGA-II with a population of 100 over 50 generations. For each run:
- Saves the full optimization history (all evaluated solutions) to `hist_FX/`
- Saves the Pareto front scatter plot to `scatter_plots_files/`
- Returns the final Pareto front and hypervolume convergence curve

### 5. Clustering Strategies

#### K-Means (`breast_cancer_main.py`)
Tries k = 2–6 clusters, scores each by:
```
score = ratio_good_in_cluster - ratio_bad_in_cluster
```
Selects the best k and cluster, then constrains the NSGA-II search to that cluster's feature bounds.

#### Hierarchical / Ward Linkage (`hierarchical_cluster_main.py`)
Builds a dendrogram using Ward's method. Iterates over n = 2–6 clusters, scores each cluster's label distribution, and selects the optimal cluster for constrained search. Saves dendrograms to `dendrograms/`.

#### KNN Boundary (`knn_bound_main.py`)
Finds the k nearest neighbors (default k=5) of the target instance and uses their bounding box as the search space.

### 6. Adaptive Heuristic Search (AHS) Fallback
All three strategies implement a two-stage fallback:

- **Stage 1 (Viable Cluster)**: If the selected cluster/neighborhood contains desired-class samples, run constrained NSGA-II normally.
- **Stage 2 (Empty-Class Fallback)**: If no desired-class samples exist in the cluster, find the nearest viable sample outside the cluster and create a bounding box between the original instance and that seed sample. Run NSGA-II in this tighter space.

---

## Output Files

| Directory             | Contents                                                            |
|-----------------------|---------------------------------------------------------------------|
| `scatter_plots_files/`| Pareto front scatter plots per sample/seed                         |
| `hist_FX/`            | CSV files with full NSGA-II population history (objectives + genes)|
| `plots/`              | K-Means cluster bar charts and distance-from-instance bar charts   |
| `dendrograms/`        | Hierarchical clustering dendrograms                                 |
| `hv_plots/`           | Hypervolume convergence plots                                       |
| `hv_logs/`            | CSV logs of average hypervolume per generation                      |
| `pareto_combined/`    | Side-by-side Pareto front comparisons (baseline vs. clustered)     |

---

## Installation

### Prerequisites
- Python 3.8+

### Install dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib pymoo
```

Or install from a requirements file if provided:
```bash
pip install -r requirements.txt
```

---

## Usage

Each entry point script follows the same structure: load data → train model → run baseline → run clustering-guided search.

### K-Means Guided (recommended starting point)
```bash
python breast_cancer_main.py
```

### Hierarchical Clustering Guided
```bash
python hierarchical_cluster_main.py
```

### KNN Boundary Guided
```bash
python knn_bound_main.py
```

### Configuration

Key parameters are set near the top of each main script:

```python
file_name = "breast_cancer.csv"   # Path to your dataset
class_label = "Diagnosis"         # Name of the target column
mapping_values = {"M": 1, "B": 0} # Optional: map string labels to integers
seed = 42                          # Random seed for reproducibility
bad_sample_index = 368             # Index of the instance to explain
model_choice = "SVC"               # Classifier to use (KNN, DTC, SVC, MLP, NBC)

original_label = 1                 # The class the instance currently belongs to
desired_label = 0                  # The class we want to flip the prediction to
```

For the KNN-boundary script, also adjust:
```python
num_k = 5  # Number of nearest neighbors to consider
```

---

## Example: Breast Cancer Dataset

The included `breast_cancer.csv` is the Wisconsin Breast Cancer dataset where:
- `Diagnosis = M (1)` → **Malignant** (the "bad" outcome)
- `Diagnosis = B (0)` → **Benign** (the desired outcome)

Running `breast_cancer_main.py` with default settings will:
1. Train an SVC classifier
2. Generate counterfactuals for sample index `368` (a malignant case)
3. Find the minimal feature changes that would lead the model to predict benign

---

## Key Classes

### `DataProcessing(file_name, class_label, mapping_values)`
| Method              | Description                                               |
|---------------------|-----------------------------------------------------------|
| `load_and_process()`| Returns `(data_wo_label, target, min_values, max_values)` |

### `ModelTrainer(dataset, target, class_label, dataset_name, model_choice)`
| Method          | Description                            |
|-----------------|----------------------------------------|
| `train_model()` | Trains and returns the fitted model    |
| `get_results()` | Returns accuracy results as a list     |

### `KMeansClustering(df_with_label, df_wo_label, col_name)`
| Method                                                 | Description                                      |
|--------------------------------------------------------|--------------------------------------------------|
| `evaluate_k(k, bad_sample_index)`                      | Scores a given k and returns cluster boundaries  |
| `compute_kmeans(k)`                                    | Fits K-Means and returns clustered DataFrame     |
| `calculate_distance_per_cluster(cluster, sample, lbl)` | Returns distance-sorted samples in a cluster     |

### `HierarchicalClustering(df, original_label, desired_label, label_col, bad_sample_idx)`
| Method                  | Description                                             |
|-------------------------|---------------------------------------------------------|
| `perform_clustering()`  | Builds the Ward linkage matrix                          |
| `find_optimal_cluster()`| Returns best cluster data and feature boundaries        |
| `plot_dendrogram()`     | Saves dendrogram to `dendrograms/`                      |

### `LocalKNNSelection(df, label_col, bad_sample_idx)`
| Method                             | Description                                        |
|------------------------------------|----------------------------------------------------|
| `find_knn_boundaries(k)`           | Returns min/max bounds from k nearest neighbors    |
| `find_next_viable_seed(k, label)`  | Finds the closest desired-class sample beyond k    |

### `CFGenerator(...)`
| Method                                           | Description                                          |
|--------------------------------------------------|------------------------------------------------------|
| `generate_counterfactuals(desired_label, idx)`   | Runs NSGA-II and returns `(pareto_F, hv)`            |

---

## Algorithm Details

### Objective Functions (in `CFProblem._evaluate`)
For each candidate solution `x`:
```
F1 = 1 - P(desired_class | x)      # prediction error
F2 = ||x - x_original||_2          # Euclidean distance
```
Both objectives are normalized within each generation to `[0, 1]`.

### Hypervolume Metric
Convergence is tracked via hypervolume indicator with reference point `[1, 1]` using `pymoo.indicators.hv.Hypervolume`. A higher hypervolume indicates a better-spread Pareto front closer to the ideal point `[0, 0]`.

### Cluster Scoring
Both K-Means and Hierarchical strategies score clusters using:
```
score = ratio_good_in_cluster - ratio_bad_in_cluster * 0.5
```
where ratios are computed relative to the total counts across the dataset.

---

## Dependencies

| Package      | Purpose                                         |
|--------------|-------------------------------------------------|
| `numpy`      | Numerical computation                           |
| `pandas`     | Data loading and manipulation                   |
| `scikit-learn`| ML classifiers, preprocessing, metrics         |
| `scipy`      | Distance metrics, hierarchical clustering       |
| `matplotlib` | Visualization (bar charts, scatter, dendrogram) |
| `pymoo`      | Multi-objective optimization (NSGA-II, HV)      |
