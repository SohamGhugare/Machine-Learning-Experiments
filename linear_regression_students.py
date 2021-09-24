import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

""" 
    Reading the dataset
    Note: The csv file has values seperated by ; and not ,
        therefore we need to add a sep argument while reading the csv file 
"""

data = pd.read_csv("datasets/student-mat.csv", sep=";")

"""
    Trimming the dataset:
        Select data that is specifically suitable for prediction
        Try playing around different attributes for more precision
        Values should be preferably integer, if not then convert into int
"""

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


""" 
    Attributes -> Data used for training the model for predicting stuff (Here, every column except G3 will be attribute)
    Labels -> Actual data to be predicted (Here, G3 will be the label)
"""

predict = "G3"
x = np.array(data.drop([predict], axis=1)) # Converting the dataset to array, x -> attributes array (training data); axis=1 -> columns
y = np.array(data[predict]) # y -> labels array (values to be predicted)


""" 
    Splitting the data into training data and test data
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train%20test%20split#sklearn.model_selection.train_test_split
"""
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # test_size=0.1 means 10% -> test data

def linear_predict(x, y):
    """ Function to carry out spliting and training, returns the linear model and accuracy """
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    """ 
        Initiating an object of LinearRegression class
        Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression
    """
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) # Fitting all the data using fit() method
    accuracy = linear.score(x_test, y_test) # Predicting the accuracy of prediction using score() method
    return linear, accuracy

""" 
    Looping 30 times to get the best linear accuracy model 
    Current best acc = 94 %
    Model name = student-model.pickle
    (Un-comment the below code block to re-run the loop & find the new best accuracy)
"""

# best_score = 0
# for _ in range(30):
#     linear, accuracy = linear_predict(x, y)
#     if accuracy > best_score:
#         best_score = accuracy
#         # writing the model into the file
#         with open("student-model.pickle", "wb") as f:
#             pickle.dump(linear, f)
#         print(f"New best acc: {best_score:.2f} | Model saved successfully!")


""" 
    Loading the saved model
    Model name: student-model.pickle
"""
with open("student-model.pickle", "rb") as pickle_in:
    linear = pickle.load(pickle_in)

"""
    Predicting using the predict() method
    Takes -> Array, Shape
    Returns -> Array, Shape
"""
# accuracy = linear.score(x_test, y_test) # Accuracy of the model
predictions = linear.predict(x_test)


""" Printing all of the data gathered """

# print(f"Accuracy: {accuracy}")
print(f"Coefficient: \n{linear.coef_}")
print(f"Intercept: \n{linear.intercept_}")

for x in range(len(predictions)):
    """
        Prediction -> Value attained by the predict() method
        Actual -> Actual value expected (taken from y_test)
        Error -> Farness of the predicted value from the actual value
        Input -> Input array given while prediction (from x_test)
    """
    print(f"Prediction: {predictions[x]:.2f} | Actual: {y_test[x]} | Error: {float(predictions[x])-y_test[x]:.2f} | Input: {x_test[x]}")


""" Plotting and styling """
attr = "G1" # Any 1 attribute
style.use("ggplot")
plt.scatter(data[attr], data["G3"])
plt.xlabel(attr)
plt.ylabel("Final Grade")
plt.show()