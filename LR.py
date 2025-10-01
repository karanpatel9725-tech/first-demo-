import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5]])  # Hours studied
y = np.array([40, 50, 60, 65, 80])  # Marks obtained


model = LinearRegression()

model.fit(x, y)

predicted = model.predict([[6]])
print("Predicted marks for 6 hours:", predicted[0])
# here [0] is the index

print("Slope (m):", model.coef_[0])
print("Intersept (c):", model.intercept_)
