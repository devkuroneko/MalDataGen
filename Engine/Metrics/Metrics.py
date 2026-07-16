#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
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
    import json
    import numpy
    import logging
    import pandas as pd 
     
    import time 
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    from Engine.Metrics.Binary.Recall import Recall

    from Engine.Metrics.Binary.F1_Score import F1Score
    from Engine.Metrics.Binary.Accuracy import Accuracy

    from Engine.Metrics.Binary.Precision import Precision

    from Engine.Metrics.Binary.Specificity import Specificity

    from Engine.Metrics.Binary.TruePositive import TruePositive
    from Engine.Metrics.Binary.TrueNegative import TrueNegative

    from Engine.Metrics.Binary.FalsePositive import FalsePositive
    from Engine.Metrics.Binary.FalseNegative import FalseNegative

    from Engine.Metrics.Binary.AreaUnderCurve import AreaUnderCurve

    from Engine.Metrics.Binary.MeanSquaredError import MeanSquareError
    from Engine.Metrics.Binary.TrueNegativeRate import TrueNegativeRate

    from Engine.Metrics.Binary.FalsePositiveRate import FalsePositiveRate
    from Engine.Metrics.Binary.MeanAbsoluteError import MeanAbsoluteError

    from Engine.Metrics.Distance.EuclideanDistance import EuclideanDistance
    from Engine.Metrics.Distance.HellingerDistance import HellingerDistance
    from Engine.Metrics.Distance.ManhattanDistance import ManhattanDistance

    from Engine.Metrics.Distance.HammingDistance import HammingDistance
    from Engine.Metrics.Distance.JaccardDistance import JaccardDistance
    from Engine.Metrics.Distance.PermutationTest import PermutationTest
    from Engine.Utils.ResourceMonitor import Timer
    from Engine.Utils.ResourceMonitor import log_resource_usage
    from Engine.Utils.ResourceMonitor import get_current_memory_mb
    from Engine.Utils.ResourceMonitor import psutil
    from Engine.Utils.ResourceMonitor import NOT_AVAILABLE


except ImportError as error:
    print(error)
    sys.exit(-1)

NOT_APPLICABLE = "not_applicable"

def import_metrics(function):
    """
    Decorator to create an instance of the Metrics class
    before executing the wrapped function.

    Parameters:
        function (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that initializes Metrics.
    """
    def wrapper(self, *args, **kwargs):
        # Create an instance of Metrics, passing the arguments from the instance
        Metrics.__init__(self, self.arguments)
        # Call the wrapped function with the metrics instance and other arguments
        return function(self, *args, **kwargs)

    return wrapper

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.float32, numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.int32, numpy.int64)):
            return int(obj)
        return super().default(obj)
    
class Metrics:
    """
    Class to calculate and manage various evaluation metrics for machine learning models.
    Initializes dictionaries to store binary metrics, distance metrics, and area under curve metrics.

    Attributes:
        _dictionary_binary_metrics (dict): Dictionary of binary evaluation metric instances.
        _dictionary_distance_metrics (dict): Dictionary of distance metric instances.
        _dictionary_area_under_curve (dict): Dictionary for area under curve metric instances.
        _classifier_list (list): List of classifiers used in the evaluation.
        _dictionary_metrics (dict): Dictionary to store metrics for different classifiers and evaluation types.
    """

    def __init__(self, arguments):
        """
        Initializes the Metrics class by creating instances of various metrics
        and setting up the metrics dictionary based on provided arguments.

        Parameters:
            arguments (Namespace): Command-line arguments containing settings for the metrics.
        """
        self._data_type = getattr(arguments, "data_type", "binary")
        self._target_type = getattr(arguments, "target_type", "auto")
        self.arguments = arguments
        self._resource_total_start_time = time.perf_counter()

        # Initialize dictionaries for binary metrics with their corresponding instances
        self._dictionary_binary_metrics = {
            Accuracy.__name__: Accuracy(),
            Precision.__name__: Precision(),
            Recall.__name__: Recall(),
            F1Score.__name__: F1Score(),
            Specificity.__name__: Specificity(),
            FalsePositiveRate.__name__: FalsePositiveRate(),
            TrueNegativeRate.__name__: TrueNegativeRate(),
            MeanSquareError.__name__: MeanSquareError(),
            MeanAbsoluteError.__name__: MeanAbsoluteError(),
            TruePositive.__name__: TruePositive(),
            FalsePositive.__name__: FalsePositive(),
            TrueNegative.__name__: TrueNegative(),
            FalseNegative.__name__: FalseNegative(),
        }

        # Initialize dictionaries for distance metrics
        self._dictionary_distance_metrics = {
            EuclideanDistance.__name__ : EuclideanDistance(),
            HellingerDistance.__name__ : HellingerDistance(),
            ManhattanDistance.__name__ : ManhattanDistance(),
        }

        if self._data_type != "continuous":
            self._dictionary_distance_metrics.update({
                HammingDistance.__name__ : HammingDistance(),
                JaccardDistance.__name__ : JaccardDistance(),
            })

        # Initialize dictionary for area under curve metrics
        self._dictionary_area_under_curve = {"AreaUnderCurve": AreaUnderCurve()}

        self._classifier_list = list()
        for c in arguments.classifier:
            if c in self.dictionary_classifiers_name:
                self._classifier_list.append(c)

        # Initialize the metrics dictionary based on the provided arguments
        self.__initialize_dictionary(arguments)
        self._process = psutil.Process() if psutil is not None else None

    def __initialize_dictionary(self, arguments):
        """
        Initializes the metrics dictionary to store evaluation metrics for different classifiers
        and evaluation types, including cross-validation folds.

        Parameters:
            arguments (Namespace): Command-line arguments containing the number of folds.
        """

        self.list_classifier_metrics = self._get_classifier_metric_names()

        # # List of distribution metrics
        self.list_distance_metrics = self._dictionary_distance_metrics.keys()
        self.list_efficiency_metrics =  ["Process_CPU_%", "Process_Memory_MB", "System_CPU_%", "System_Memory_MB", "System_Memory_%", "Time_training_ms", "Time_generating_ms"]
        self.list_sdv_metrics = ["diagnostic", "quality"]

        # Initialize the main metrics dictionary for different evaluation types and classifiers
        self._dictionary_metrics = self._dictionary_metrics  | {
            "Schema": {
                "version": "2.0",
                "compatibility": "legacy_metrics_with_evaluation_metadata",
            },
            "TS-TR": {
                classifier: {
                    **{
                        f'{fold}-Fold': {metric: NOT_APPLICABLE for metric in self.list_classifier_metrics}
                        for fold in range(1, arguments.number_k_folds + 1)
                    },
                    'Summary': {
                        metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_classifier_metrics
                    }
                } for classifier in self._classifier_list
            },
            "TR-TS": {
                classifier: {
                    **{
                        f"{fold}-Fold": {metric: NOT_APPLICABLE for metric in self.list_classifier_metrics}
                        for fold in range(1, arguments.number_k_folds + 1)
                    },
                    "Summary": {
                        metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_classifier_metrics
                    }
                } for classifier in self._classifier_list
            },
           "TR-TR": {
               classifier: {
                   **{
                       f'{fold}-Fold': {metric: NOT_APPLICABLE for metric in self.list_classifier_metrics}
                       for fold in range(1, arguments.number_k_folds + 1)
                   },
                   'Summary': {
                       metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_classifier_metrics
                   }
               } for classifier in self._classifier_list
           },
           "TR+TS-TR": {
               classifier: {
                   **{
                       f'{fold}-Fold': {metric: NOT_APPLICABLE for metric in self.list_classifier_metrics}
                       for fold in range(1, arguments.number_k_folds + 1)
                   },
                   'Summary': {
                       metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_classifier_metrics
                   }
               } for classifier in self._classifier_list
           },

            "DistanceMetrics": {
                methodology: {
                **{
                    f'{fold}-Fold': {metric: NOT_APPLICABLE for metric in self.list_distance_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_distance_metrics
                }
                } for methodology in [ "R-S", "R-R"]
            },

            "EfficiencyMetrics": {
                **{
                    f'{fold}-Fold': {metric: 0 for metric in self.list_efficiency_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': 0, 'std': 0} for metric in self.list_efficiency_metrics
                }
            },

            "SDVMetrics": {
                **{
                    f'{fold}-Fold': {metric: NOT_APPLICABLE for metric in self.list_sdv_metrics}
                    for fold in range(1, arguments.number_k_folds + 1)
                },
                'Summary': {
                    metric: {'mean': NOT_APPLICABLE, 'std': NOT_APPLICABLE} for metric in self.list_sdv_metrics
                }
            },
            "BatchClassifier": {
                **{
                    f'{fold}-Fold': {} for fold in range(1, arguments.number_k_folds + 1)
                },
                "Summary": {
                    "note": (
                        "normal and batches metrics can differ when different classifiers are configured; "
                        "use matching subset classifiers when comparing modes."
                    )
                }
            },
            "EvaluationMetadata": {
                **{
                    f'{fold}-Fold': {} for fold in range(1, arguments.number_k_folds + 1)
                },
                "Summary": {},
            },
            "Diagnostics": {
                **{
                    f'{fold}-Fold': {} for fold in range(1, arguments.number_k_folds + 1)
                },
                "Summary": {},
            },
            "ResourceUsage": {
                "current_memory_mb_by_stage": {},
                "peak_memory_mb_by_stage": {},
                "elapsed_seconds_by_stage": {},
                "total_elapsed_seconds": 0,
                "batch_processing": {
                    "largest_batch_processed": 0,
                    "number_of_batches": 0,
                    "effective_batch_size": getattr(arguments, "batch_size", NOT_AVAILABLE),
                },
            }

        }

    def _get_classifier_metric_names(self):
        if self._is_binary_task():
            return list(self._dictionary_binary_metrics.keys())

        return [
            "Accuracy",
            "MacroPrecision",
            "MacroRecall",
            "MacroF1",
            "WeightedPrecision",
            "WeightedRecall",
            "WeightedF1",
            "BalancedAccuracy",
        ]

    @staticmethod
    def _labels_to_vector(labels):
        labels = numpy.ravel(numpy.asarray(labels))
        try:
            return labels.astype(numpy.int64)
        except (TypeError, ValueError):
            return labels

    def _is_binary_task(self):
        if self._target_type == "binary":
            return True
        if self._target_type == "multiclass":
            return False
        if self._target_type in ("regression", "none"):
            return False
        return self._data_type == "binary"

    def _is_classification_applicable(self):
        return self._target_type not in ("regression", "none")

    @staticmethod
    def _numeric_metric_value(value):
        try:
            value = float(value)
            if numpy.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
        return NOT_APPLICABLE

    @staticmethod
    def _numeric_summary(values):
        numeric_values = []
        for value in values:
            try:
                numeric_value = float(value)
                if numpy.isfinite(numeric_value):
                    numeric_values.append(numeric_value)
            except (TypeError, ValueError):
                continue

        if not numeric_values:
            return NOT_APPLICABLE, NOT_APPLICABLE

        return numpy.mean(numeric_values), numpy.std(numeric_values)

    def _get_multiclass_metric_values(self, real_labels, predict_labels):
        real_labels = self._labels_to_vector(real_labels)
        predict_labels = self._labels_to_vector(predict_labels)

        return {
            "Accuracy": accuracy_score(real_labels, predict_labels),
            "MacroPrecision": precision_score(real_labels, predict_labels, average="macro", zero_division=0),
            "MacroRecall": recall_score(real_labels, predict_labels, average="macro", zero_division=0),
            "MacroF1": f1_score(real_labels, predict_labels, average="macro", zero_division=0),
            "WeightedPrecision": precision_score(real_labels, predict_labels, average="weighted", zero_division=0),
            "WeightedRecall": recall_score(real_labels, predict_labels, average="weighted", zero_division=0),
            "WeightedF1": f1_score(real_labels, predict_labels, average="weighted", zero_division=0),
            "BalancedAccuracy": balanced_accuracy_score(real_labels, predict_labels),
        }

    def _configured_num_classes_for_metrics(self):
        arguments = getattr(self, "arguments", None)
        value = getattr(arguments, "num_classes", None)
        if value:
            return int(value)

        metadata = getattr(arguments, "number_samples_per_class", None)
        if isinstance(metadata, dict) and metadata.get("number_classes"):
            return int(metadata["number_classes"])

        return None

    def _record_predictive_diagnostics(self, real_labels, predict_labels, evaluation_type, classifier, fold, metric_values):
        real_labels = self._labels_to_vector(real_labels)
        predict_labels = self._labels_to_vector(predict_labels)
        fold_key = f"{fold}-Fold"
        num_classes = self._configured_num_classes_for_metrics()

        observed_true = {int(label) for label in numpy.unique(real_labels)}
        predicted_unique, predicted_counts = numpy.unique(predict_labels, return_counts=True)
        observed_pred = {int(label) for label in predicted_unique}
        if predicted_counts.size:
            majority_index = int(numpy.argmax(predicted_counts))
            most_predicted_class = int(predicted_unique[majority_index])
            most_predicted_count = int(predicted_counts[majority_index])
            most_predicted_rate = float(most_predicted_count / max(1, predict_labels.shape[0]))
        else:
            most_predicted_class = NOT_APPLICABLE
            most_predicted_count = 0
            most_predicted_rate = NOT_APPLICABLE
        if num_classes is not None:
            expected = set(range(num_classes))
            missing_true = sorted(expected - observed_true)
            missing_pred = sorted(expected - observed_pred)
        else:
            expected = observed_true | observed_pred
            missing_true = []
            missing_pred = []

        accuracy = metric_values.get("Accuracy", NOT_APPLICABLE)
        chance_level_suspected = False
        if num_classes == 200:
            try:
                chance_level_suspected = 0.004 <= float(accuracy) <= 0.006
            except (TypeError, ValueError):
                chance_level_suspected = False

        self._dictionary_metrics.setdefault("Diagnostics", {}).setdefault(fold_key, {}).setdefault(
            evaluation_type, {}
        )[classifier] = {
            "num_classes": num_classes,
            "observed_true_class_count": int(len(observed_true)),
            "observed_pred_class_count": int(len(observed_pred)),
            "predicted_class_count": int(len(observed_pred)),
            "most_predicted_class": most_predicted_class,
            "most_predicted_count": most_predicted_count,
            "most_predicted_rate": most_predicted_rate,
            "missing_true_classes": missing_true,
            "missing_pred_classes": missing_pred,
            "chance_level_suspected": bool(chance_level_suspected),
        }


    def monitoring_start_training(self):
        self._time_start_training = time.perf_counter_ns()
        self._process_mem_start = get_current_memory_mb()
        if psutil is not None:
            self._process_cpu_start = self._process.cpu_percent(interval=None)
            self._system_cpu_start = self._process.cpu_percent(interval=None)
            self._system_mem_start = psutil.virtual_memory().used / (1000**2) #MB
            self._system_mem_start_perc = psutil.virtual_memory().percent



    def monitoring_stop_training(self, fold):
        self._time_end_training = time.perf_counter_ns()
        duration_ns  = self._time_end_training - self._time_start_training
        self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold']['Time_training_ms'] = duration_ns / 1_000_000
    

    def monitoring_start_generating(self):
        self._time_start_generating = time.perf_counter_ns()
        self._process_mem_start = get_current_memory_mb()

    def monitoring_stop_generating(self, fold):
        self._time_end_generating = time.perf_counter_ns()
        duration_ns = self._time_end_generating - self._time_start_generating
        #self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold']['Time_generating_secs'] = duration.total_seconds()

        if psutil is None:
            self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold'].update({
            'Time_generating_ms':  duration_ns / 1_000_000,
            'Process_CPU_%': NOT_AVAILABLE,
            'Process_Memory_MB': NOT_AVAILABLE,
            'System_CPU_%': NOT_AVAILABLE,
            'System_Memory_MB': NOT_AVAILABLE,
            'System_Memory_%': NOT_AVAILABLE,
            })
            return

        # Uso de CPU (percentual médio durante a execução)
        cpu_usage = self._process.cpu_percent(interval=None) / psutil.cpu_count()
        
        # Uso de memória (diferença entre início e fim)
        process_mem_end = self._process.memory_info().rss / (1000**2)  # Em MB
        if self._process_mem_start == NOT_AVAILABLE:
            process_mem_usage = NOT_AVAILABLE
        else:
            process_mem_usage = process_mem_end - self._process_mem_start

        self._dictionary_metrics["EfficiencyMetrics"][f'{fold+1}-Fold'].update({
        'Time_generating_ms':  duration_ns / 1_000_000,
        'Process_CPU_%': cpu_usage,
        'Process_Memory_MB': process_mem_usage,
        'System_CPU_%': psutil.cpu_percent(interval=None),
        'System_Memory_MB': psutil.virtual_memory().used / (1000**2),
        'System_Memory_%': psutil.virtual_memory().percent,
        })

    def save_dictionary_to_json(self, output_file_results):
        """
        Saves the metrics dictionary to a JSON file.

        Parameters:
            output_file_results (str): The file path where the JSON will be saved.
        """

        try:
            with Timer("saving") as timer:
                with open(output_file_results, 'w') as json_file:
                    json.dump(self._dictionary_metrics, json_file, indent=4, cls=NumpyEncoder)
                    print(f"Dictionary successfully saved to {output_file_results}")
            self.record_resource_usage("saving", timer.elapsed_seconds)
            metrics_json_path = output_file_results.rsplit('/', 1)[0] + "/metrics.json"
            if metrics_json_path != output_file_results:
                with open(metrics_json_path, 'w') as json_file:
                    json.dump(self._dictionary_metrics, json_file, indent=4, cls=NumpyEncoder)

        except Exception as e:
            # Print an error message if saving fails
            print(f"Error saving the dictionary: {e}")

    def record_resource_usage(self, stage_name, elapsed_seconds=None):
        stage_key = self._resource_stage_key(stage_name)
        usage = log_resource_usage(stage_key)
        resource_metrics = self._dictionary_metrics.setdefault("ResourceUsage", {})
        resource_metrics.setdefault("current_memory_mb_by_stage", {})[stage_key] = usage["current_memory_mb"]
        resource_metrics.setdefault("peak_memory_mb_by_stage", {})[stage_key] = usage["peak_memory_mb"]
        if elapsed_seconds is not None:
            resource_metrics.setdefault("elapsed_seconds_by_stage", {})[stage_key] = round(float(elapsed_seconds), 6)
        resource_metrics["total_elapsed_seconds"] = round(
            time.perf_counter() - getattr(self, "_resource_total_start_time", time.perf_counter()),
            6,
        )

    def resource_timer(self, stage_name):
        owner = self

        class _ResourceTimer:
            def __enter__(self):
                self._timer = Timer(stage_name)
                self._timer.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self._timer.__exit__(exc_type, exc_value, traceback)
                owner.record_resource_usage(stage_name, self._timer.elapsed_seconds)
                return False

        return _ResourceTimer()

    def record_batch_processed(self, batch_size):
        resource_metrics = self._dictionary_metrics.setdefault("ResourceUsage", {})
        batch_metrics = resource_metrics.setdefault("batch_processing", {
            "largest_batch_processed": 0,
            "number_of_batches": 0,
            "effective_batch_size": getattr(self.arguments, "batch_size", NOT_AVAILABLE),
        })
        batch_size = int(batch_size)
        batch_metrics["largest_batch_processed"] = max(int(batch_metrics.get("largest_batch_processed", 0)), batch_size)
        batch_metrics["number_of_batches"] = int(batch_metrics.get("number_of_batches", 0)) + 1
        batch_metrics["effective_batch_size"] = getattr(self.arguments, "batch_size", NOT_AVAILABLE)

    def _resource_stage_key(self, stage_name):
        fold_number = getattr(self, "fold_number", None)
        if fold_number is None:
            return stage_name
        if stage_name in {"training", "generation", "evaluation"}:
            return f"{stage_name}_fold_{fold_number + 1}"
        return stage_name

    def record_batch_classifier_metadata(self, evaluation_type, classifier, fold, metadata):
        fold_key = f"{fold}-Fold"
        block = self._dictionary_metrics.setdefault("BatchClassifier", {}).setdefault(fold_key, {})
        block[evaluation_type] = {
            "classifier": classifier,
            **metadata,
        }

    def update_mean_std_fold(self):
        """Updates the mean and standard deviation of evaluation metrics across all folds.

        This method calculates and stores the mean and standard deviation for each metric
        across all cross-validation folds, for each methodology and classifier combination.
        The results are stored back in the metrics dictionary under the 'Mean-Fold' entry.

        The method processes two methodologies ("TS-TR" and "TR-TS") and all classifiers
        stored in the dictionary. For each combination, it:

            1. Identifies all fold entries (keys ending with "-Fold")

            2. Computes mean and std for each metric across folds

            3. Stores the results in the 'Mean-Fold' dictionary structure

        Note:
            - Expects self._dictionary_metrics to be properly initialized
            - The 'Mean-Fold' entry must exist for each classifier-methodology pair
            - Modifies the dictionary in-place by adding mean/std values
        """

        for methodology in ["TS-TR", "TR-TS", "TR-TR", "TR+TS-TR"]:
            for classifier in self._dictionary_classifiers_name:
                if methodology not in self._dictionary_metrics or classifier not in self._dictionary_metrics[methodology]:
                    continue

                # Get the metrics data for current methodology and classifier
                data = self._dictionary_metrics[methodology][classifier]

                # Identify all fold keys (excluding "Mean-Fold")
                folds = [key for key in data if key.endswith("-Fold")]

                # Calculate mean and std for each metric across folds
                for metric in data["Summary"]:

                    # Collect all values for this metric across folds
                    values = [data[fold][metric] for fold in folds]

                    # Compute statistics
                    mean_value, std_value = self._numeric_summary(values)

                    # Store results in Mean-Fold structure
                    self._dictionary_metrics[methodology][classifier]["Summary"][metric]["mean"] = mean_value
                    self._dictionary_metrics[methodology][classifier]["Summary"][metric]["std"] = std_value
        
         
        data = self._dictionary_metrics["DistanceMetrics"]
        folds = [key for key in data["R-S"] if key.endswith("-Fold")]
        for methodology in ["R-S", "R-R"]:
            for metric in data[methodology]["Summary"].keys():
                values = [data[methodology][fold][metric] for fold in folds]
                mean_value, std_value = self._numeric_summary(values)
                data[methodology]["Summary"][metric]["mean"] = mean_value
                data[methodology]["Summary"][metric]["std"] = std_value

        data = self._dictionary_metrics["EfficiencyMetrics"]
        folds = [key for key in data if key.endswith("-Fold")]
        for metric in data["Summary"].keys():
            values = [data[fold][metric] for fold in folds]
            mean_value, std_value = self._numeric_summary(values)
            data["Summary"][metric]["mean"] = mean_value
            data["Summary"][metric]["std"] = std_value

    def mark_fold_not_applicable(self, fold, reason):
        """Mark unsupported evaluations explicitly instead of leaving fake zero metrics."""
        for methodology in ["TS-TR", "TR-TS", "TR-TR", "TR+TS-TR"]:
            for classifier in self._dictionary_classifiers_name:
                fold_key = f"{fold}-Fold"
                if fold_key in self._dictionary_metrics[methodology][classifier]:
                    for metric in self._dictionary_metrics[methodology][classifier][fold_key]:
                        self._dictionary_metrics[methodology][classifier][fold_key][metric] = NOT_APPLICABLE

        for methodology in ["R-S", "R-R"]:
            fold_key = f"{fold}-Fold"
            if fold_key in self._dictionary_metrics["DistanceMetrics"][methodology]:
                for metric in self._dictionary_metrics["DistanceMetrics"][methodology][fold_key]:
                    self._dictionary_metrics["DistanceMetrics"][methodology][fold_key][metric] = NOT_APPLICABLE

        self._dictionary_metrics.setdefault("NotApplicable", {})[f"{fold}-Fold"] = reason

    def mark_classifier_metrics_not_applicable(self, evaluation_type, classifier, fold, reason):
        fold_key = f"{fold}-Fold"
        metric_block = self._dictionary_metrics.get(evaluation_type, {}).get(classifier, {})
        if fold_key in metric_block:
            for metric in metric_block[fold_key]:
                metric_block[fold_key][metric] = NOT_APPLICABLE
        self._dictionary_metrics.setdefault("NotApplicable", {}).setdefault(fold_key, {})[
            f"{evaluation_type}:{classifier}"
        ] = reason

    def mark_evaluation_classifiers_not_applicable(self, evaluation_type, fold, reason):
        logging.warning("%s Marking %s classifier metrics for fold %s as %s.", reason, evaluation_type, fold, NOT_APPLICABLE)
        for classifier in self._dictionary_classifiers_name:
            self.mark_classifier_metrics_not_applicable(evaluation_type, classifier, fold, reason)

    def mark_distance_metrics_not_applicable(self, evaluation_type, fold, reason):
        fold_key = f"{fold}-Fold"
        distance_block = self._dictionary_metrics.get("DistanceMetrics", {}).get(evaluation_type, {})
        if fold_key in distance_block:
            for metric in distance_block[fold_key]:
                distance_block[fold_key][metric] = NOT_APPLICABLE
        self._dictionary_metrics.setdefault("NotApplicable", {}).setdefault(fold_key, {})[
            f"DistanceMetrics:{evaluation_type}"
        ] = reason

    def get_binary_metrics(self, real_labels, predict_labels, evaluation_type, classifier, fold):
        """
        Calculates binary metrics using real and predicted labels and updates the metrics dictionary.

        Parameters:
            real_labels (array-like): The true labels.
            predict_labels (array-like): The predicted labels from the model.
            evaluation_type (str): The evaluation type (e.g., "TS-TR").
            classifier (str): The name of the classifier being evaluated.
            fold (str): The current fold number in cross-validation.
        """
         
        logging.info(f"\t\t\t Binary metrics")
        for metric_name, instance in self._dictionary_binary_metrics.items():
            # Calculate the metric and update the dictionary with the result
            try:
                metric_value = instance.get_metric(real_labels, predict_labels)
            except Exception as error:
                logging.warning("Binary metric %s is %s: %s", metric_name, NOT_APPLICABLE, error)
                metric_value = NOT_APPLICABLE
            self._dictionary_metrics[evaluation_type][classifier][f"{fold}-Fold"][metric_name] = (
                self._numeric_metric_value(metric_value)
            )

    def get_task_metrics(self, real_labels, predict_labels, evaluation_type, classifier, fold):
        """
        Calculates predictive metrics according to the configured dataset mode.

        binary:
            Keeps the original binary metric set for backward compatibility.
        multiclass:
            Uses macro/weighted classification metrics that work for two or more classes.
        continuous:
            Preserves continuous features and uses multiclass-safe predictive metrics for
            class-conditioned generation. Distribution fidelity remains handled by distance metrics.
        """
        if not self._is_classification_applicable():
            reason = f"target_type={self._target_type} has no classification metrics in this evaluator."
            logging.warning("%s Marking %s/%s fold %s as %s.", reason, evaluation_type, classifier, fold, NOT_APPLICABLE)
            self.mark_classifier_metrics_not_applicable(evaluation_type, classifier, fold, reason)
            return

        if self._is_binary_task():
            self.get_binary_metrics(real_labels, predict_labels, evaluation_type, classifier, fold)
            return

        logging.info(f"\t\t\t {self._target_type} predictive metrics")
        try:
            metric_values = self._get_multiclass_metric_values(real_labels, predict_labels)
        except Exception as error:
            logging.warning(
                "Error calculating %s predictive metrics. Marking as %s: %s",
                self._target_type,
                NOT_APPLICABLE,
                error,
            )
            metric_values = {metric: NOT_APPLICABLE for metric in self.list_classifier_metrics}

        for metric_name in self.list_classifier_metrics:
            self._dictionary_metrics[evaluation_type][classifier][f"{fold}-Fold"][metric_name] = (
                self._numeric_metric_value(metric_values.get(metric_name, NOT_APPLICABLE))
            )

        self._record_predictive_diagnostics(
            real_labels,
            predict_labels,
            evaluation_type,
            classifier,
            fold,
            metric_values,
        )

    
    def get_distance_metrics(self, x_evaluation_real, x_evaluation_synthetic, evaluation_type, fold):
        """
        Calculates distance metrics between two distributions and updates the metrics dictionary.

        Parameters:
            real_distribution (array-like): The true distribution.
            synthetic_distribution (array-like): The synthetic distribution generated by the model.
            fold (str): The current fold number in cross-validation.
        """

        logging.info(f"\t\t\t Distance metrics")
        for metric_name, instance in self._dictionary_distance_metrics.items():
            # Calculate the distance metric and update the dictionary with the result
            try:
                metric_value = instance.get_metric(x_evaluation_real, x_evaluation_synthetic)
            except Exception as error:
                logging.warning("Distance metric %s is %s: %s", metric_name, NOT_APPLICABLE, error)
                metric_value = NOT_APPLICABLE
            self._dictionary_metrics["DistanceMetrics"][evaluation_type][f"{fold}-Fold"][metric_name] = (  
                self._numeric_metric_value(metric_value)
            )

    def get_AUC_metric(self, real_label, synthetic_label_probability, evaluation_type, classifier, fold):
        """
        Calculates the Area Under Curve (AUC) metric and updates the metrics dictionary.

        Parameters:
            real_label (array-like): The true labels.
            synthetic_label_probability (array-like): The predicted probabilities from the model.
            evaluation_type (str): The evaluation type (e.g., "TS-TR").
            classifier (str): The name of the classifier being evaluated.
            fold (str): The current fold number in cross-validation.
        """
        for metric_name, instance in self._dictionary_area_under_curve.items():
            # Calculate the AUC and update the dictionary with the result
            self._dictionary_metrics[evaluation_type][classifier][fold][metric_name] = (
                instance.get_metric(real_label, synthetic_label_probability)
            )
