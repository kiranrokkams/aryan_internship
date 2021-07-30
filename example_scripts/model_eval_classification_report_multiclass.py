import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tigerml.model_eval import ClassificationReport

iris = load_iris()
X = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
yhat_train = rf.predict(X_train)
yhat_test = rf.predict(X_test)

# Option 1 - with model
clsOpt1 = ClassificationReport(
    y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test, model=rf
)
clsOpt1.get_report(
    file_path="Reports/ClassificationReport--Multi-class--report_option-1",
    include_shap=True,
    format=".xlsx",
)

# Option 2 - without model
clsOpt2 = ClassificationReport(
    y_train=y_train,
    x_train=X_train,
    x_test=X_test,
    y_test=y_test,
    yhat_train=yhat_train,
    yhat_test=yhat_test,
)
clsOpt2.get_report(
    file_path="Reports/ClassificationReport--Multi-class--report_option-2",
    format=".xlsx",
)
