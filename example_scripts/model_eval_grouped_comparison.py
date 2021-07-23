import pandas as pd
import numpy as np
from tigerml.model_eval import MultiModelComparisonRegression

# Load the data
# Download the csv file from following Google Drive location
# https://drive.google.com/file/d/1ZQhtKQYmvOiRm2y33edOpjzipPFx4JYb
results_df = pd.read_csv(
    "external_sources/script_data/store_item_daily_predictions.csv",
    parse_dates=["date"],
)
results_df.info()
# Make sure all the grouping columns are of str type
# as it will increase visibility of all values on heatmap axis
results_df["item"] = results_df["item"].astype(str)
results_df.head()

# Initialize the model comparison object and get report (w/o baseline)
mmcr = MultiModelComparisonRegression(
    data=results_df,
    group_cols=["store", "item"],
    y_true_col="actuals",
    y_pred_col="predicted",
)
mmcr.get_report(file_path="Reports/MultiModelComparisonReport--Regression", format=".xlsx")

# Create a dummy baseline predictions column
np.random.seed(42)
noise = np.random.choice(range(10), size=results_df.shape[0])
baseline = results_df[["actuals", "predicted"]].mean(axis=1) + noise
results_df["baseline"] = baseline
results_df.head()

mmcr2 = MultiModelComparisonRegression(
    data=results_df,
    group_cols=["store", "item"],
    y_true_col="actuals",
    y_pred_col="predicted",
    y_base_col="baseline",
)
mmcr2.get_report(
    file_path="Reports/MultiModelComparisonReport--Regression--with_Baseline",
    format=".xlsx"
)
