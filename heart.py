import pandas as pd

data = pd.read_csv("heart.csv")


data.info()

data.describe()

%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=8, figsize=(20,15))
plt.show()

data = pd.read_csv("heart.csv")
data = data.drop("chol", 1)
data = data.drop("fbs", 1)
data = data.drop("restecg", 1)


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)


hearts = train_set.copy()
hearts
X = hearts.drop("target", 1)
y = hearts["target"]

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
logreg.fit(X,y)

X_test = test_set.drop("target", 1)
y_test = test_set["target"]
pred = logreg.predict(X_test)
pred

probs = logreg.predict_proba(X_test)
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
probs[:10]

logreg.score(X_test, y_test)
