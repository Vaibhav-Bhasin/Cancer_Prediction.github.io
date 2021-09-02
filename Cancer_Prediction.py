#  Project -- AI Cancer Prediction
# malignant and benign
#1 Collect data 

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
#print(data["feature_names"])
features = data["data"]
outcomes = data["target"]
print(features)
#2 proprocess data #spliting training data and testing data
from sklearn.model_selection import train_test_split
train_features, test_features, train_outcomes, test_outcomes = train_test_split(features,outcomes)

#3 choose the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
