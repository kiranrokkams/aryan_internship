import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

regRep = RegressionReport(
    y_train=y_train,
    x_train=X_train,
    x_test=X_test,
    y_test=y_test,
    yhat_train=yhat_train,
    yhat_test=yhat_test,
)

# regRep.get_report(format=".html")
regRep.get_report(format=".xlsx")
# regRep.get_report(format=".pptx")
