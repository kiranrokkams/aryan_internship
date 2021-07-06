# Library imports
import yaml
import json
import pandas as pd
import great_expectations as ge

from sklearn import linear_model
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from ta_lib.regression.api import RegressionReport 
from ta_lib.classification.api import ClassificationReport


class CSVal_DataPrep:
    
    def __init__(self, reg_conf_path, cls_conf_path):
        self.reg_conf_path = reg_conf_path
        self.cls_conf_path = cls_conf_path

    def reg_data_prepare(self):
        # Config Read
        raw_config = yaml.load(open(self.reg_conf_path), Loader=yaml.FullLoader)
        config = {key: val.format(**raw_config) for key, val in raw_config.items()}
#         print(config)

        # data processing
        full_data = pd.read_csv(config['raw_data_path'])
        full_data.rename(columns = {config['target_var']:"target"})
        config['target_var'] = "target"
        y = full_data[config['target_var']]
        X = full_data.drop(config['target_var'], axis=1)

        # Train-Test data split
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.1, random_state=int(config['random_state']))

        # Data Save 
        train_x.to_csv(config["train_x_path"], index=False)
        train_y.to_csv(config["train_y_path"], index=False)
        test_x.to_csv(config["test_x_path"], index=False)
        test_y.to_csv(config["test_y_path"], index=False)


    def cls_data_prepare(self):
        # Config Read
        raw_config = yaml.load(open(self.cls_conf_path), Loader=yaml.FullLoader)
        config = {key: val.format(**raw_config) for key, val in raw_config.items()}
#         print(config)

        # data processing
        full_data = pd.read_csv(config['raw_data_path'])
        y = full_data[config['target_var']]
        X = full_data.drop(config['target_var'], axis=1)

        # Train-Test data split
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=0.1, random_state=int(config['random_state']))

        # Data Save 
        train_x.to_csv(config["train_x_path"], index=False)
        train_y.to_csv(config["train_y_path"], index=False)
        test_x.to_csv(config["test_x_path"], index=False)
        test_y.to_csv(config["test_y_path"], index=False)


class RegressionChecks():

    def __init__(self, config_path: str, n_features: int, errorbuckets_spec = None, cutoff_value: float = 0.5):
        self.errorbuckets_spec = errorbuckets_spec
        self.n_features = n_features
        self.cutoff_value = cutoff_value
        
        # Config Read
        raw_config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        self.config = {key: val.format(**raw_config) for key, val in raw_config.items()}
#         print(self.config)
        
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()
        
    def load_data(self):
        # Data Read
        self.train_x = pd.read_csv(self.config["train_x_path"])
        self.train_y = pd.read_csv(self.config["train_y_path"])
        self.test_x = pd.read_csv(self.config["test_x_path"])
        self.test_y = pd.read_csv(self.config["test_y_path"])
        
    def python_com_data(self, model):
        """
        Objective: Pull all model_eval data for R comparison in pandas dataframe format
        """
        # Model Report object
        reg_linear_report = RegressionReport(model=model, 
                                             x_train=self.train_x.copy(), 
                                             y_train=self.train_y.copy(), 
                                             x_test=self.test_x.copy(), 
                                             y_test=self.test_y.copy(), 
                                             refit=True)

        # Data dictionary
        data_to_compare = dict()

        # Metrics
        data_to_compare['metrics'] = reg_linear_report.get_performance_report(cutoff_value=self.cutoff_value)['metrics'].data.reset_index()

        # Model Coefficients
        if len(reg_linear_report.explainer.interpretations) > 0 and hasattr(reg_linear_report, "explainer"):
            interpretation_obj = reg_linear_report.explainer.get_plots(include_shap=False, 
                                                      errorbuckets_spec=self.errorbuckets_spec, 
                                                      n_features=self.n_features, 
                                                      include_shap_test_error_analysis=False)
            if "coeff_table" in interpretation_obj:
                data_to_compare['coeff_table'] = interpretation_obj['coeff_table']

        # Error report
        yhat_train = reg_linear_report.evaluator.yhat_train
        residual_train = reg_linear_report.evaluator.residual_train
        data_to_compare["train_report"] = pd.DataFrame({"actual":self.train_y['target'], "pred":yhat_train, "error":residual_train})

        yhat_test = reg_linear_report.evaluator.yhat_test
        residual_test = reg_linear_report.evaluator.residual_test
        data_to_compare["test_report"] = pd.DataFrame({"actual":self.test_y['target'], "pred":yhat_test, "error":residual_test})

        return data_to_compare

    def read_data(self, algo: str):
        obj = dict()
        obj["metrics"] = pd.read_csv(self.config["r_processed_data_path"] + algo + self.config["metrics"])
        obj["train_report"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_train" + self.config["error"])
        obj["test_report"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_test" + self.config["error"])

        if algo == "linear":
            obj["coeff_table"] = pd.read_csv(self.config["r_processed_data_path"] + algo + self.config["coeffs"])

        return obj

    def read_r_processed_data(self):
        r_model_eval_algos = dict()

        # Linear
        if "linear" in self.config:
            r_model_eval_algos["linear"] = self.read_data(self.config["linear"])

        # Random Forest
        if "rf" in self.config:
            r_model_eval_algos["rf"] = self.read_data(self.config["rf"])

        # Linear
        if "xgb" in self.config:
            r_model_eval_algos["xgb"] = self.read_data(self.config["xgb"])

        return r_model_eval_algos

    def compare_model_report(self, py_model_report: dict, r_model_report: dict, algo: str):
        # Dictionary to hold the differences
        report_diff_dict = dict()

        # Metrics
    #     py_model_report["metrics"]["train"] = py_model_report["metrics"]["train"].reset_index()
        train_diff = py_model_report["metrics"]["train"].astype(float) - r_model_report["metrics"]["train"].astype(float)
        test_diff = py_model_report["metrics"]["test"].astype(float) - r_model_report["metrics"]["test"].astype(float)
        report_diff_dict["metric_report"] = pd.DataFrame({"train_diff": train_diff, "test_diff": test_diff})

        # Train error report
        train_actual_diff = py_model_report["train_report"]["actual"].astype(float) - r_model_report["train_report"]["actual"].astype(float)
        train_pred_diff = py_model_report["train_report"]["pred"].astype(float) - r_model_report["train_report"]["pred"].astype(float)
        train_error_diff = py_model_report["train_report"]["error"].astype(float) - r_model_report["train_report"]["error"].astype(float)
        report_diff_dict["train_report"] = pd.DataFrame({"actual_diff": train_actual_diff, "pred_diff": train_pred_diff, "error_diff": train_error_diff})

        # Test error report
        test_actual_diff = py_model_report["test_report"]["actual"].astype(float) - r_model_report["test_report"]["actual"].astype(float)
        test_pred_diff = py_model_report["test_report"]["pred"].astype(float) - r_model_report["test_report"]["pred"].astype(float)
        test_error_diff = py_model_report["test_report"]["error"].astype(float) - r_model_report["test_report"]["error"].astype(float)
        report_diff_dict["test_report"] = pd.DataFrame({"actual_diff": test_actual_diff, "pred_diff": test_pred_diff, "error_diff": test_error_diff})

        # Coeff
        if algo == "linear":
            py_coeff = py_model_report["coeff_table"].copy()
            py_coeff = py_coeff.reset_index()
            merged_summary = pd.merge(py_coeff, r_model_report["coeff_table"], left_on="index", right_on="term", how='inner')
            merged_summary['coeff_diff'] = merged_summary["coefficients"] - merged_summary["estimate"]
            report_diff_dict["coeff_diff"] = pd.DataFrame({"coeff_diff": merged_summary['coeff_diff']})

        return report_diff_dict
    
    def multi_model_compare(self):
        # Model Dict
        mode_dict = {
            "linear": linear_model.LinearRegression(),
            "ridge": linear_model.Ridge(alpha=1, fit_intercept=True, normalize=True),
            "lasso": linear_model.Lasso(alpha=0.1),
            "rf": RandomForestRegressor(n_estimators=20),
            "xgb": XGBRegressor()
        }

        # Python data 
        self.load_data()
        model_eval_df = {key: self.python_com_data(val) for key, val in mode_dict.items()}
        model_eval_algos = {key: val for key, val in model_eval_df.items() if key in self.config}

        # R data
        r_model_eval_algos = self.read_r_processed_data()

        # Comparison Report
        final_keys = list(set(model_eval_algos.keys()).intersection(set(r_model_eval_algos.keys())))
        model_eval_algos = {k: v for k, v in model_eval_algos.items() if k in final_keys}
        self.diff_report = {key: self.compare_model_report(model_eval_algos[key], r_model_eval_algos[key], key) for key in model_eval_algos}

    def ge_report(self):
        self.ge_diff_report = dict()
        for key, report_dict in self.diff_report.items():
            """
            For more details on result_format use https://docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/result_format.html#result-format
            For more details on expect_column_values_to_be_between use https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/dataset/index.html#great_expectations.dataset.dataset.Dataset.expect_column_values_to_be_between
            For more details on list of expectations use https://docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html
            """
            results = dict()

            # Train report
            train_dict = dict()
            train_ge = ge.from_pandas(report_dict["train_report"])
            for column in train_ge.columns:
                train_dict[column] = train_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["train"] = train_dict

            # Test report
            test_dict = dict()
            test_ge = ge.from_pandas(report_dict["test_report"])
            for column in test_ge.columns:
                test_dict[column] = test_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["test"] = test_dict

            # Metric report
            metric_dict = dict()
            metric_ge = ge.from_pandas(report_dict["metric_report"])
            for column in metric_ge.columns:
                metric_dict[column] = metric_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["metrics"] = metric_dict

            if key == "linear":
                coeff_ge = ge.from_pandas(report_dict["coeff_diff"])
        #         results["coeff_diff"] = coeff_ge.expect_column_mean_to_be_between("coeff_diff", -1, 1).to_json_dict()
                results["coeff_diff"] = coeff_ge.expect_column_values_to_be_between("coeff_diff", min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()

            self.ge_diff_report[key] = results
            
    def get_report(self, format: str = "json", save_path: str = "regression"):
        # Create Data differences
        self.multi_model_compare()
        
        # Create greate_expectations report
        self.ge_report()
        
        # Save Report
        if(format == "json"):
            # Serializing json  
            json_object = json.dumps(self.ge_diff_report, indent = 4) 

            # Writing to sample.json 
            with open("{}/cs_validation_report.json".format(save_path), "w") as outfile: 
                outfile.write(json_object)
        else:
            for key in self.ge_diff_report:
                df = pd.json_normalize(self.ge_diff_report[key])
                df.T.to_csv("{}/".format(save_path) + key + "_cs_validation.csv")
                
                
class ClassificationChecks():

    def __init__(self, config_path: str, n_features: int, errorbuckets_spec = None, cutoff_value: float = 0.5):
        self.errorbuckets_spec = errorbuckets_spec
        self.n_features = n_features
        self.cutoff_value = cutoff_value
        
        # Config Read
        raw_config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        self.config = {key: val.format(**raw_config) for key, val in raw_config.items()}
#         print(config)
        
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()
        
    def load_data(self):
        # Data Read
        self.train_x = pd.read_csv(self.config["train_x_path"])
        self.train_y = pd.read_csv(self.config["train_y_path"])
        self.test_x = pd.read_csv(self.config["test_x_path"])
        self.test_y = pd.read_csv(self.config["test_y_path"])
        
    def python_com_data(self, model):
        """
        Objective: Pull all model_eval data for R comparison in pandas dataframe format
        """
        # Model Report object
        cls_linear_report = ClassificationReport(model=model,
                                                 x_train=self.train_x,
                                                 y_train=self.train_y,
                                                 x_test=self.test_x,
                                                 y_test=self.test_y,
                                                 refit=True)

        # Data dictionary
        data_to_compare = dict()

        # Metrics
        data_to_compare['metrics'] = cls_linear_report.evaluator._get_metrics_for_decision_threshold().transpose().reset_index(level=[1]).pivot(columns="dataset")[0].reset_index()

        # Gains Table
    #     data_to_compare["gains"] = cls_linear_report.evaluator.gains_table()
        cls_linear_report.evaluator._init_gains()
        data_to_compare["gains"] = {"train": cls_linear_report.evaluator.gains_table_result_train.reset_index(), 
                                    "test": cls_linear_report.evaluator.gains_table_result_test.reset_index()}
        data_to_compare["confusion_matrix"] = {}

        # Train
        train_y_pred = cls_linear_report.evaluator.yhat_train.copy()
        train_y_pred[train_y_pred >= self.cutoff_value] = 1
        train_y_pred[train_y_pred < 1] = 0
        data_to_compare["confusion_matrix"]["train"] = cls_linear_report.evaluator._compute_cm(cls_linear_report.evaluator.y_train, train_y_pred)

        # Test
        test_y_pred = cls_linear_report.evaluator.yhat_test.copy()
        test_y_pred[test_y_pred >= self.cutoff_value] = 1
        test_y_pred[test_y_pred < 1] = 0
        data_to_compare["confusion_matrix"]["test"] = cls_linear_report.evaluator._compute_cm(cls_linear_report.evaluator.y_test, test_y_pred)

        # Model Coefficients
        if len(cls_linear_report.explainer.interpretations) > 0 and hasattr(cls_linear_report, "explainer"):
            interpretation_obj = cls_linear_report.explainer.get_plots(include_shap=False, 
                                                      errorbuckets_spec=self.errorbuckets_spec, 
                                                      n_features=self.n_features, 
                                                      include_shap_test_error_analysis=False)
            if "coeff_table" in interpretation_obj:
                data_to_compare['coeff_table'] = interpretation_obj['coeff_table']

        # Error report
        data_to_compare["train_report"] = pd.DataFrame({"actual":cls_linear_report.evaluator.y_train, 
                                                        "pred":train_y_pred,
                                                       "pred_prob": cls_linear_report.evaluator.yhat_train.copy()})

        data_to_compare["test_report"] = pd.DataFrame({"actual":cls_linear_report.evaluator.y_test, 
                                                       "pred":test_y_pred,
                                                      "pred_prob": cls_linear_report.evaluator.yhat_test.copy()})

        return data_to_compare


    def read_data(self, algo: str):
        obj = dict()
        obj["metrics"] = pd.read_csv(self.config["r_processed_data_path"] + algo + self.config["metrics"])
        obj["train_report"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_train" + self.config["error"])
        obj["test_report"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_test" + self.config["error"])

        # Gains
        obj["gains"] = {}
        obj["gains"]["train"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_train" + self.config["gains"])
        obj["gains"]["test"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_test" + self.config["gains"])

        # Confusion Matrix
        obj["confusion_matrix"] = {}
        obj["confusion_matrix"]["train"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_train_cm.csv")
        obj["confusion_matrix"]["test"] = pd.read_csv(self.config["r_processed_data_path"] + algo + "_test_cm.csv")

        if algo == "linear":
            obj["coeff_table"] = pd.read_csv(self.config["r_processed_data_path"] + algo + self.config["coeffs"])

        return obj

    def read_r_processed_data(self):
        r_model_eval_algos = dict()

        # Linear
        if "linear" in self.config:
            r_model_eval_algos["linear"] = self.read_data(self.config["linear"])

        # Random Forest
        if "rf" in self.config:
            r_model_eval_algos["rf"] = self.read_data(self.config["rf"])

        # Linear
        if "xgb" in self.config:
            r_model_eval_algos["xgb"] = self.read_data(self.config["xgb"])

        return r_model_eval_algos

    def compare_model_report(self, py_model_report: dict, r_model_report: dict, algo: str):
        # Dictionary to hold the differences
        report_diff_dict = dict()

        # Metrics
        train_diff = py_model_report["metrics"]["train"].astype(float) - r_model_report["metrics"]["train"].astype(float)
        test_diff = py_model_report["metrics"]["test"].astype(float) - r_model_report["metrics"]["test"].astype(float)
        report_diff_dict["metric_report"] = pd.DataFrame({"train_diff": train_diff, "test_diff": test_diff})

        # Train error report
        train_actual_diff = py_model_report["train_report"]["actual"].astype(float) - r_model_report["train_report"]["actual"].astype(float)
        train_pred_diff = py_model_report["train_report"]["pred"].astype(float) - r_model_report["train_report"]["pred"].astype(float)
        train_prob_diff = py_model_report["train_report"]["pred_prob"].astype(float) - r_model_report["train_report"]["pred_prob"].astype(float)
        report_diff_dict["train_report"] = pd.DataFrame({"actual_diff": train_actual_diff, "pred_diff": train_pred_diff, "prob_diff": train_prob_diff})

        # Test error report
        test_actual_diff = py_model_report["test_report"]["actual"].astype(float) - r_model_report["test_report"]["actual"].astype(float)
        test_pred_diff = py_model_report["test_report"]["pred"].astype(float) - r_model_report["test_report"]["pred"].astype(float)
        test_prob_diff = py_model_report["test_report"]["pred_prob"].astype(float) - r_model_report["test_report"]["pred_prob"].astype(float)
        report_diff_dict["test_report"] = pd.DataFrame({"actual_diff": test_actual_diff, "pred_diff": test_pred_diff, "prob_diff": test_prob_diff})

        # Gains error report
        train_lift_diff = py_model_report["gains"]["train"]["lift"].astype(float) - r_model_report["gains"]["train"]["lift"].astype(float)
        test_lift_diff = py_model_report["gains"]["test"]["lift"].astype(float) - r_model_report["gains"]["test"]["lift"].astype(float)
        report_diff_dict["gains_report"] = pd.DataFrame({"train_lift_diff": train_lift_diff, "test_lift_diff": test_lift_diff})

        # Coeff
        if algo == "linear":
            py_coeff = py_model_report["coeff_table"].copy()
            py_coeff = py_coeff.reset_index()
            py_coeff["index"] = py_coeff["index"].map(lambda x: x.replace(" ", "."))
            merged_summary = pd.merge(py_coeff, r_model_report["coeff_table"], left_on="index", right_on="term", how='inner')
            merged_summary['coeff_diff'] = merged_summary["coefficients"] - merged_summary["estimate"]
            report_diff_dict["coeff_diff"] = pd.DataFrame({"coeff_diff": merged_summary['coeff_diff']})

        return report_diff_dict
    
    def multi_model_compare(self):
        # Model Dict
        mode_dict = {
            "linear": linear_model.LogisticRegression(),
            "rf": RandomForestClassifier(),
            "xgb": XGBClassifier()
        }

        # Python data 
        self.load_data()
        # Python data 
        model_eval_df = {key: self.python_com_data(model=val) for key, val in mode_dict.items()}
        model_eval_algos = {key: val for key, val in model_eval_df.items() if key in self.config}

        # R data
        r_model_eval_algos = self.read_r_processed_data()

        # Comparison Report
        final_keys = list(set(model_eval_algos.keys()).intersection(set(r_model_eval_algos.keys())))
        model_eval_algos = {k: v for k, v in model_eval_algos.items() if k in final_keys}
        self.diff_report = {key: self.compare_model_report(model_eval_algos[key], r_model_eval_algos[key], key) for key in model_eval_algos}
        
    def ge_report(self):
        self.ge_diff_report = dict()
        for key, report_dict in self.diff_report.items():
            """
            For more details on result_format use https://docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/result_format.html#result-format
            For more details on expect_column_values_to_be_between use https://docs.greatexpectations.io/en/latest/autoapi/great_expectations/dataset/dataset/index.html#great_expectations.dataset.dataset.Dataset.expect_column_values_to_be_between
            For more details on list of expectations use https://docs.greatexpectations.io/en/latest/reference/glossary_of_expectations.html
            """
            results = dict()

            # Train report
            train_dict = dict()
            train_ge = ge.from_pandas(report_dict["train_report"])
            for column in train_ge.columns:
                train_dict[column] = train_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["train"] = train_dict

            # Test report
            test_dict = dict()
            test_ge = ge.from_pandas(report_dict["test_report"])
            for column in test_ge.columns:
                test_dict[column] = test_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["test"] = test_dict

            # Metric report
            metric_dict = dict()
            metric_ge = ge.from_pandas(report_dict["metric_report"])
            for column in metric_ge.columns:
                metric_dict[column] = metric_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["metrics"] = metric_dict

            # Gains report
            gains_dict = dict()
            gains_ge = ge.from_pandas(report_dict["gains_report"])
            for column in gains_ge.columns:
                gains_dict[column] = gains_ge.expect_column_values_to_be_between(column, min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()
            results["metrics"] = gains_dict

            if key == "linear":
                coeff_ge = ge.from_pandas(report_dict["coeff_diff"])
        #         results["coeff_diff"] = coeff_ge.expect_column_mean_to_be_between("coeff_diff", -1, 1).to_json_dict()
                results["coeff_diff"] = coeff_ge.expect_column_values_to_be_between("coeff_diff", min_value=-1, max_value=1, mostly=None, 
                                                            result_format="BASIC", include_config=True).to_json_dict()

            self.ge_diff_report[key] = results
            
    def get_report(self, format: str = "json", save_path: str = "classification"):
        # Create Data differences
        self.multi_model_compare()
        
        # Create greate_expectations report
        self.ge_report()
        
        # Save Report
        if format == "json":
            # Serializing json  
            json_object = json.dumps(self.ge_diff_report, indent = 4) 

            # Writing to sample.json 
            with open("{}/cs_validation_report.json".format(save_path), "w") as outfile: 
                outfile.write(json_object) 
        elif format == "csv":
            for key in self.ge_diff_report:
                df = pd.json_normalize(self.ge_diff_report[key])
                df.T.to_csv("{}/".format(save_path) + key + "_cs_validation.csv")
        else:
            raise Exception("output_type is not correct")