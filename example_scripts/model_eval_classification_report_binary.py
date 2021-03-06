import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tigerml.model_eval import ClassificationReport

cancer = load_breast_cancer()
X = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cls = LogisticRegression(max_iter=10000)
cls.fit(X_train, y_train)
yhat_train = cls.predict(X_train)
yhat_test = cls.predict(X_test)

# Option 1 - with model
clsOpt1 = ClassificationReport(y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test,
                               model=cls)
clsOpt1.get_report(file_path="Reports/ClassificationReport--Binary-class--report_option-1", include_shap=True)

# Option 2 - without model
clsOpt2 = ClassificationReport(y_train=y_train, x_train=X_train, x_test=X_test, y_test=y_test,
                               yhat_train=yhat_train, yhat_test=yhat_test)
clsOpt2.get_report(file_path="Reports/ClassificationReport--Binary-class--report_option-2")