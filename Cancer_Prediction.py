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

#4 train the model 
clf = clf.fit(train_features,train_outcomes)
#5 prediction
p1 = clf.predict(test_features)
from sklearn.metrics import accuracy_score
print("Accuracy::",accuracy_score(test_outcomes,p1)*100, "%")
import matplotlib.pyplot as plt
fig = plt.figure()
ax= fig.add_axes([0,0,1,1])
modles =["DecisionTree","XYZ","ABC"]
accuracy =[89,88,90]
plt.bar(modles,accuracy)
plt.show()
