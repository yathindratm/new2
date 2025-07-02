from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X,y = load_iris(return_X_y = True)
clf = LogisticRegression(max_iter=200).fit(X,y)

print("Model trained successfully")
