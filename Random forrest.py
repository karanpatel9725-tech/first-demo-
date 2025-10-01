from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# split the data set into traon test concept
X_train, X_test, Y_train, Y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
# 42 is also called as a seed which randomly train and test data
model = RandomForestClassifier(n_estimators=100, random_state=42)

# testing the model
model.fit(X_train, Y_train)

# training the model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"accuracy of model: {accuracy:2f}")
