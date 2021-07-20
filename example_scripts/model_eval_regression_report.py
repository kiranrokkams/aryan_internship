import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tigerml.model_eval import RegressionReport

boston = load_boston()
X = pd.DataFrame(boston["data"], columns=boston["feature_names"])
y = boston["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)
yhat_train = reg.predict(X_train)
yhat_test = reg.predict(X_test)

# Option 1 - with model
regOpt1 = RegressionReport(
    y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test, model=reg
)
regOpt1.get_report(
    file_path="Reports/RegressionReport--report_option-1", include_shap=True
)

# Option 2 - without model
regOpt2 = RegressionReport(
    y_train=y_train,
    x_train=X_train,
    x_test=X_test,
    y_test=y_test,
    yhat_train=yhat_train,
    yhat_test=yhat_test,
)
regOpt2.get_report(file_path="Reports/RegressionReport--report_option-2")
