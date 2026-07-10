#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 
 
from Tools.ClusteringVisualizer import ClusteringVisualizer

# MIT License
#
# Copyright (c) 2025 2025 MalDataGen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    import sys
    import argparse
    import logging
    import numpy
    from dataclasses import dataclass

    from Engine.DataIO.CSVLoader import CSVDataProcessor
    from Tools.PlotHeatMap import HeatmapComparator

    from Tools.PlotTrainingCurve import PlotTrainingCurve

    from Tools.PlotDistanceMetrics import PlotDistanceMetrics
    from Tools.PlotConfusionMatrix import PlotConfusionMatrix
    import Tools.config as config
    from Tools.PlotClasssificationMetrics import PlotClassificationMetrics
    from plots_svm import plot_heatmap_svm
except ImportError as error:
    print(error)
    sys.exit(-1)

"""
SYNDATAGEN VISUALIZATION SUITE - Demonstration
============================================================

This script provides a complete demonstration of SynDataGen's integrated visualization capabilities
for machine learning workflows. It serves as both a production-ready tool and educational reference
implementation.

Key Features:
------------
1. Multi-modal Visualization:
   - Comparative analysis: Side-by-side heatmap comparisons
   - Model diagnostics: Training curves, confusion matrices
   - Statistical analysis: Distance metrics, classification reports

2. Advanced Configuration:
   - Dynamic parameterization via command line interface
   - Flexible data handling for various input formats
   - Customizable visualization aesthetics

3. Production-Grade Features:
   - Comprehensive error handling
   - Batch processing for multiple folds
   - Automated output organization

Implementation Notes:
---------------------
- Uses matplotlib/seaborn for backend rendering
- Implements proper figure sizing for publication-quality outputs
- Supports both interactive display and file export modes

"""
def list_of_strs(arg):
    return list(map(str, arg.split(',')))
@dataclass
class Arguments:
    data_load_label_column: object = -1
    data_load_max_samples: int = -1
    data_load_max_columns: int = -1
    data_load_start_column: int = 0
    data_load_end_column: int = 50
    data_load_path_file_input: str = ''
    data_load_path_file_output: str = ''
    data_load_exclude_columns: list = None
    number_samples_per_class: dict = None
    data_type: str = 'continuous'


def load_dataset_processor(path: str):
    processor = CSVDataProcessor(Arguments(data_load_path_file_input=path,
                                           data_load_path_file_output=''))
    processor.load_csv()
    return processor


def get_available_labels(processor):
    labels = numpy.ravel(processor._data_loaded_labels)
    if labels.size == 0:
        return []
    return sorted(numpy.unique(labels).tolist())


def format_label(label):
    try:
        numeric_label = float(label)
        if numeric_label.is_integer():
            return str(int(numeric_label))
    except (TypeError, ValueError):
        pass
    return str(label)


def select_plot_labels(processor, max_labels=2):
    labels = get_available_labels(processor)
    preferred_labels = [1, 0]
    selected = [label for label in preferred_labels if label in labels]
    selected.extend(label for label in labels if label not in selected)
    return selected[:max_labels]


def get_feature_block(processor, label, row_limit=None, column_limit=None):
    features = processor.get_features_by_label(label)
    if features.size == 0:
        return None

    columns = min(column_limit or processor.get_number_columns(), features.shape[1])
    rows = min(row_limit or columns, features.shape[0])
    if rows == 0 or columns == 0:
        return None

    return features[:rows, :columns]


def plot_optional(description, callback):
    try:
        callback()
    except Exception as error:
        logging.warning("Skipping %s: %s", description, error)

def plot_heatmaps_from_dataset_comparison(dataset_path: str, synthetic_path: str, output_file: str):
    real_processor = load_dataset_processor(dataset_path)
    synthetic_processor = load_dataset_processor(synthetic_path)

    for label in select_plot_labels(real_processor):
        real_features = real_processor.get_features_by_label(label)
        synthetic_features = synthetic_processor.get_features_by_label(label)
        if real_features.size == 0 or synthetic_features.size == 0:
            logging.warning("Skipping comparison heatmap for missing label %s.", format_label(label))
            continue

        columns = min(real_features.shape[1], synthetic_features.shape[1],
                      real_processor.get_number_columns(), synthetic_processor.get_number_columns())
        rows = min(columns, real_features.shape[0], synthetic_features.shape[0])
        if rows == 0 or columns == 0:
            logging.warning("Skipping comparison heatmap for label %s without enough samples.", format_label(label))
            continue

        real = real_features[:rows, :columns]
        synth = synthetic_features[:rows, :columns]
        label_name = format_label(label)
        HeatmapComparator(
            figure_size=(16, 8),
            color_map='coolwarm',
            titles=(f'Real Dataset (label={label_name})', f'Synthetic Dataset (label={label_name})'),
            linewidths=0,
            linecolor='black',
            x_tick_labels=False,
            y_tick_labels=False,
            annotations=False,
            font_scale=1.2,
            color_bar=True,
            color_bar_label="Value Intensity",
            normalize=True,
            grid_shape=(1, 3),
            export_path=f'{output_file}_label_{label_name}.pdf',
            tight_layout=True,
            style="whitegrid",
            show_difference_matrix=True
        ).plot(real, synth)


def plot_clusters_from_dataset(dataset_path: str, output_file: str,
                               cluster_algo: str = 'agglo',
                               reduction_algo: str = 'pca',
                               n_clusters: int = 8,
                               n_components: int = 2):
    """
    Plots clusters for samples with label 1 from a dataset.

    Args:
        dataset_path: Path to the input CSV file
        output_file: Path to save the output visualization
        cluster_algo: Clustering algorithm ('kmeans', 'dbscan', 'agglo', etc.)
        reduction_algo: Dimensionality reduction method ('pca', 'tsne', 'umap')
        n_clusters: Number of clusters to identify
        n_components: 2 for 2D visualization or 3 for 3D
    """
    processor = load_dataset_processor(dataset_path)
    labels = select_plot_labels(processor, max_labels=1)
    if not labels:
        logging.warning("Skipping cluster plot because no labels are available.")
        return

    # Prefer label 1 when present, otherwise use the first available label.
    X = processor.get_features_by_label(labels[0])
    number_cols = processor.get_number_columns()
    X = X[:number_cols, :number_cols]  # Use the same size limitation as in heatmap function
    if X.shape[0] < 2:
        logging.warning("Skipping cluster plot because label %s has fewer than 2 samples.", format_label(labels[0]))
        return
    n_clusters = min(n_clusters, X.shape[0])

    # Initialize the visualizer
    visualizer = ClusteringVisualizer(
        cluster_algo=cluster_algo,
        reduction_algo=reduction_algo,
        number_clusters=n_clusters,
        number_components=n_components,
        random_state=42,
        point_size=50,
        color_map='viridis',
        title_font_size=14,
        show_legend=True,
        export_path=output_file  # Added export path parameter
    )

    # Preprocess, cluster and visualize
    X_scaled = visualizer.preprocess(X)
    labels = visualizer.cluster(X_scaled)
    visualizer.plot_clusters()



def plot_heatmaps_from_dataset(dataset_path: str, output_file: str):
    processor = load_dataset_processor(dataset_path)
    blocks = []
    for label in select_plot_labels(processor):
        block = get_feature_block(processor, label)
        if block is not None:
            blocks.append((label, block))

    if not blocks:
        logging.warning("Skipping heatmap for %s because no labeled samples are available.", dataset_path)
        return

    columns = min(block.shape[1] for _, block in blocks)
    rows = min(columns, *(block.shape[0] for _, block in blocks))
    if rows == 0 or columns == 0:
        logging.warning("Skipping heatmap for %s because there are not enough samples.", dataset_path)
        return

    matrices = [block[:rows, :columns] for _, block in blocks]
    titles = tuple(f'Sample label={format_label(label)}' for label, _ in blocks)

    HeatmapComparator(
        figure_size=(16, 8),
        color_map='coolwarm',
        titles=titles,
        linewidths=0,
        linecolor='black',
        x_tick_labels=False,
        y_tick_labels=False,
        font_scale=1.2,
        color_bar=True,
        color_bar_label="Value Intensity",
        normalize=True,
        grid_shape=(1, len(matrices)),
        export_path=output_file,
        tight_layout=True,
        style="whitegrid",
        show_difference_matrix=False).plot(*matrices)


 
def main():

    parser = argparse.ArgumentParser(description='Plots SynDataGen')

    parser.add_argument("--results", "-r", nargs="+", type=list_of_strs, required=True)
    parser.add_argument("--training", "-t", nargs='+', type=str, required=False)
    parser.add_argument("--title", "-i", nargs='+', type=str, default=[""])
    parser.add_argument("--folds", "-k", nargs='+', type=int, default=5)
    parser.add_argument("--dataset", "-d", nargs='+', type=str, required=True)
    parser.add_argument("--output_dir", "-o", nargs='+', type=str, required=True)
    parser.add_argument("--model", "-m", nargs='+', type=str, default="none")
    parser.add_argument("--f_plot", "-f_pl",action='store_true')
    args = parser.parse_args()
    args.results = args.results[0]
 
    if (args.f_plot):
        plot_optional("predictive metrics plot",
                      lambda: PlotClassificationMetrics(input_files=[args.results[-1]], title=" - ".join(args.title)))
        plot_optional("distance metrics plot",
                      lambda: PlotDistanceMetrics(input_files=[args.results[-1]], title=" - ".join(args.title)))
        plot_optional("confusion matrix plot",
                      lambda: PlotConfusionMatrix(input_file=args.results[-1], title=" - ".join(args.title)))
        output=args.output_dir
        plot_optional("SVM heatmap plot",
                      lambda: plot_heatmap_svm(input_files=args.results,title= "/".join(output[0].split("/")[:2])))
    else: 
            plot_optional("predictive metrics plot",
                          lambda: PlotClassificationMetrics(input_files=args.results, title=" - ".join(args.title)))
            plot_optional("distance metrics plot",
                          lambda: PlotDistanceMetrics(input_files=args.results, title=" - ".join(args.title)))
            plot_optional("confusion matrix plot",
                          lambda: PlotConfusionMatrix(input_file=args.results[0], title=" - ".join(args.title)))
    if args.training:
        plot_optional("training curve plot",
                      lambda: PlotTrainingCurve(input_file=args.training, title=" - ".join(args.title)))

    plot_optional("original dataset heatmap",
                  lambda: plot_heatmaps_from_dataset(
                      args.dataset[0],
                      f'{args.output_dir[0]}/EvaluationResults/heat_map_original_data.pdf'))

    for k in range(args.folds[0]):
        path = f'{args.output_dir[0]}/DataGenerated/DataOutput_K_fold_{k}_{args.model[0]}.txt'
        plot_optional(f'synthetic dataset heatmap fold {k}',
                      lambda path=path, k=k: plot_heatmaps_from_dataset(
                          path, f'{args.output_dir[0]}/EvaluationResults/heat_map_k_{k}.pdf'))
        plot_optional(f'comparison heatmap fold {k}',
                      lambda path=path, k=k: plot_heatmaps_from_dataset_comparison(
                          args.dataset[0], path,
                          f'{args.output_dir[0]}/EvaluationResults/Comparison_heat_map_k_{k}'))

    plot_optional("cluster plot",
                  lambda: plot_clusters_from_dataset(
                      dataset_path=args.dataset[0],
                      output_file=f'{args.output_dir[0]}/EvaluationResults/ClusterMapOriginalDataPositiveMalware.pdf',
                      cluster_algo='agglo',
                      reduction_algo='pca',
                      n_clusters=5,
                      n_components=2
                  ))


if __name__ == '__main__':
    sys.exit(main())
