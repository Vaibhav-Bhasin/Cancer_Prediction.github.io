#  Project -- AI Cancer Prediction
# malignant and benign
#1 Collect data 

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
#print(data["feature_names"])
features = data["data"]
outcomes = data["target"]
print(features)
