import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tigerml.model_eval import RegressionComparison

boston = load_boston()
X = pd.DataFrame(boston["data"], columns=boston["feature_names"])
y = boston["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Model 1 - Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
yhat_test_lr = lr.predict(X_test)

# Model 2 - Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
yhat_test_rf = rf.predict(X_test)

# Option 1 - with model
regOpt1 = RegressionComparison(y=y_test, models=[lr, rf], x=X_test)
regOpt1.get_report(file_path="Reports/RegressionComparisonReport--report_option-1", format=".xlsx")

# Option 2 - without model
yhats = {"Linear Regression": yhat_test_lr, "Random Forest": yhat_test_rf}
regOpt2 = RegressionComparison(y=y_test, yhats=yhats)
regOpt2.get_report(file_path="Reports/RegressionComparisonReport--report_option-2", format=".xlsx")
