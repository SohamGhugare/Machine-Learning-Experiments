import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/homeprices.csv')

X = np.array(df.drop("price", axis=1))
y = np.array(df["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

## Plotting data
# pyplot.xlabel("Area (Sq. Ft.)")
# pyplot.ylabel("Price (USD)")
# pyplot.scatter(df.area, df.price, color="red", marker="+")
# pyplot.show()

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

for x in range(len(predictions)):
    print(f"Predict: {round(predictions[x])}  Actual: {y_test[x]}")

print(f"Accuracy: {round(model.score(X_train, y_train)*100)}%")
print(f"Coeff: {model.coef_}")
print(f"Intercept: {model.intercept_}")