
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight

# from sklearn import svm


# Number of random trials
NUM_TRIALS = 30

# Load the dataset
file = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\scripts\\centroids.pkl'
with (open(file, "rb")) as f:
    df = pickle.load(f)
    ce = np.array(df["centroids"])
    ce = ce/np.linalg.norm(ce, axis=0)
    fl = np.array(df["flux"])
    fl = fl/np.linalg.norm(ce, axis=0)
    X = np.concatenate([ce,fl],axis=1)
    # X = ce 
    y = df["labels"]
    print(X.shape)

# import ipdb; ipdb.set_trace()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Set up possible values of parameters to optimize over
# p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}

# # We will use a Support Vector Classifier with "rbf" kernel
# svm = SVC(kernel="rbf")

# # Fit the SVM model on the entire training set
# clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=4)
# clf.fit(X_train, y_train)

# # Predict labels for the test set
# y_pred = clf.predict(X_test)
# svm = SVC(kernel='linear')
# p_grid = {"C": [1, 10, 100,50], "gamma": [0.01, 0.1,0.001]}
# clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=4)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = [200,300,100,-10]
print(class_weights)
svc = SVC(decision_function_shape='ovo', class_weight=dict(zip(np.unique(y_train), class_weights)))

clf = OneVsRestClassifier(svc)
y_pred = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix = 
# Create confusion matrix plot
def plot_confusion_matrix(conf_matrix, classes):
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()


plot_confusion_matrix(conf_matrix, classes=["Present/Original", "Not Present", "Rejected", "Accepted"])
