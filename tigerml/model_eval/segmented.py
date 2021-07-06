from tigerml.core.utils.segmented import calculate_all_segments, get_segment_filter
from .base import ClassificationEvaluation, RegressionEvaluation
import tigerml.core.dataframe as td
from tigerml.core.reports import create_report
from tigerml.core.utils import time_now_readable


class SegmentedReport:

    def __init__(self, is_classification, segment_by, model=None, model_dict=None):
        self.segment_by = segment_by
        self.model = model
        self.model_dict = model_dict
        self.is_classification = is_classification

    def _calculate_all_segments(self):
        self.all_segments = calculate_all_segments(self.x_train, self.segment_by)

    def get_report_for_segment(self, segment):
        if self.model:
            model = self.model
        else:
            model = self.model_dict[segment]
        data = self.x_train[get_segment_filter(self.x_train, self.segment_by, segment)]
        y_data = self.y_train[get_segment_filter(self.y_train, self.segment_by, segment)]
        report = ClassificationEvaluation(model)
        report.fit(data, y_data)
        data = self.x_test[get_segment_filter(self.x_test, self.segment_by, segment)]
        y_data = self.y_test[get_segment_filter(self.y_test, self.segment_by, segment)]
        report.score(data, y_data)
        return report

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self._calculate_all_segments()

    def score(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test


class ClassificationSegmentedPerformanceMixin:

    def get_report_data_per_segment(self, segment):
        report = self.get_report_for_segment(segment)
        report_data = {}
        report_data['metrics'] = report.evaluator.get_metrics()
        report_data['roc_curve'] = report.evaluator.compute_roc_curve()
        report_data['pr_curve'] = report.evaluator.compute_pr_curve()
        report_data['confusion_matrices'] = report.evaluator.compute_cm()
        report_data['threshold_analysis'] = report.evaluator.compute_dt_data()
        return report_data

    def create_report_data(self):
        self.data = {}
        for segment in self.all_segments:
            data = self.get_report_data_per_segment(segment)
            for key in data:
                if key not in self.data:
                    self.data[key] = td.DataFrame()
                td.concat([self.data[key], data[key]], axis=0)

    def create_metrics_table(self):
        pass

    def create_roc_curve(self):
        pass

    def create_pr_curve(self):
        pass

    def create_confusion_matrices(self):
        pass

    def create_threshold_analysis(self):
        pass

    def get_performance_report(self):
        perf_dict = {}
        perf_dict['metrics'] = self.create_metrics_table()
        return perf_dict


class RegressionSegmentedPerformanceMixin:

    def get_report_data_per_segment(self, segment):
        report = self.get_report_for_segment(segment)
        report_data = {}
        report_data['metrics'] = report.evaluator.get_metrics()
        report_data['best_fit_data'] = report.evaluator.compute_best_fit()
        report_data['avp_scatter_plots'] = report.evaluator.create_actual_v_predicted()
        report_data['residual_distribution_data'] = report.evaluator.create_residuals_histogram()
        report_data['residual_scatter_plots'] = report.evaluator.create_residuals_scatter()
        return report_data

    def create_report_data(self):
        self.data = {}
        for segment in self.all_segments:
            data = self.get_report_data_per_segment(segment)
            for key in data:
                if key not in self.data:
                    self.data[key] = td.DataFrame()
                td.concat([self.data[key], data[key]], axis=0)

    def create_metrics_table(self):
        pass

    def create_best_fit_plot(self):
        pass

    def create_actual_v_predicted(self):
        pass

    def create_residual_distribution(self):
        pass

    def create_residuals_scatter(self):
        pass

    def get_performance_report(self):
        perf_dict = {}
        perf_dict['metrics'] = self.create_metrics_table()
        avp_dict = {}
        avp_dict['best_fit'] = self.create_best_fit_plot()
        avp_dict['actual_vs_predicted_scatter'] = self.create_actual_v_predicted()
        perf_dict['actual_vs_predicted'] = avp_dict
        residuals_dict = {}
        residuals_dict['residual_distribution'] = self.create_residual_distribution()
        residuals_dict['residual_scatter'] = self.create_residuals_scatter()
        perf_dict['residual_analysis'] = residuals_dict
        return perf_dict


def return_segmented_report(models, segment_by, is_classification):
    if is_classification:
        mixin_class = ClassificationSegmentedPerformanceMixin
    else:
        mixin_class = RegressionSegmentedPerformanceMixin

    class SegmentedModelReport(SegmentedReport, mixin_class):

        def feature_importances(self):
            pass

        def get_report(self, file_path=""):
            perf_dict = self.get_performance_report()
            interpret_dict = {}
            interpret_dict["feature_importances"] = [self.feature_importances()]
            report_dict = {}
            report_dict["performance"] = perf_dict
            report_dict["interpretation"] = interpret_dict
            if not file_path:
                file_path = (
                    f'{"classification" if self.is_classification else "regression"}'
                    f'_segmented_report_at_{time_now_readable()}'
                )
            create_report(report_dict, name=file_path, format=".html")

    if isinstance(models, dict):
        model = None
        model_dict = models
    else:
        model = models
        model_dict = None

    return SegmentedModelReport(is_classification, segment_by, model, model_dict)


class ClassificationSegmentedReport:

    def __new__(cls, segment_by, models):
        return return_segmented_report(models, segment_by, True)


class RegressionSegmentedReport:

    def __new__(cls, segment_by, models):
        return return_segmented_report(models, segment_by, False)

