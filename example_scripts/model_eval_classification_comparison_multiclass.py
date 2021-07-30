import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tigerml.model_eval import ClassificationComparison

iris = load_iris()
X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Model 1 - Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
yhat_test_lr = lr.predict_proba(X_test)

# Model 2 - Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
yhat_test_rf = rf.predict_proba(X_test)

# Option 1 - with model
clsOpt1 = ClassificationComparison(y=y_test, models=[lr, rf], x=X_test)
clsOpt1.get_report(
    file_path="Reports/ClassificationComparisonReport--Multi-class--report_option-1",
    format=".xlsx",
)

# Option 2 - without model
yhats = {"Logistic Regression": yhat_test_lr, "Random Forest": yhat_test_rf}
clsOpt2 = ClassificationComparison(y=y_test, yhats=yhats)
clsOpt2.get_report(
    file_path="Reports/ClassificationComparisonReport--Multi-class--report_option-2",
    format=".xlsx",
)
