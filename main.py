from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


def printf(format, *values):
    print(format % values)


# Models to be used: DecisionTree(0), KNN(1), SVC(2)

# Dataset Given (Height, Weight, Shoe Size):
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Decision Tree Implementation
clf_d = tree.DecisionTreeClassifier().fit(X, Y)

# K Nearest Neighbours Implementation

clf_k = KNeighborsClassifier(n_neighbors=3,
                             weights='uniform',
                             algorithm='ball_tree').fit(X, Y)

# Support Vector Classification Implementation

clf_s = SVC(gamma='auto').fit(X, Y)

# Predict samples

# Check correctness
print("Checking Correctness...")
printf("DTree - Predicted: %s, Expected: male",
       clf_d.predict([[183, 70, 44.5]]))
printf("KNN - Predicted: %s, Expected: male",
       clf_k.predict([[183, 70, 44.5]]))
printf("SVC - Predicted: %s, Expected: male",
       clf_s.predict([[183, 70, 44.5]]))

# Test Accuracy

X_test = np.array([[183, 70, 44.5], [158, 48, 37], [171, 75, 42],
                   [173, 65, 42], [173, 57, 39], [189, 104, 45],
                   [175, 64, 39], [177, 70, 40], [159, 55, 37],
                   [181, 85, 43], [169, 103, 36]])

Y_test = np.array(['male', 'female', 'male', 'male', 'male', 'male', 'male',
                   'male', 'female', 'male', 'female'])

# Apply results to test data
predict_d = clf_d.predict(X_test)
predict_k = clf_k.predict(X_test)
predict_s = clf_s.predict(X_test)

# Validation and Finding Accuracy
accuracy_d = 0
accuracy_k = 0
accuracy_s = 0

for i in range(0, len(Y_test)):
    if(Y_test[i] == predict_d[i]):
        accuracy_d = accuracy_d + 1
    if(Y_test[i] == predict_k[i]):
        accuracy_k = accuracy_k + 1
    if(Y_test[i] == predict_s[i]):
        accuracy_s = accuracy_s + 1

accuracy_d = accuracy_d / len(Y_test)
accuracy_k = accuracy_k / len(Y_test)
accuracy_s = accuracy_s / len(Y_test)

print("Input::")
print(X_test)
print("Expected Output:")
print(Y_test)
print("Actual Output:")

if accuracy_d >= accuracy_k and accuracy_d >= accuracy_s:
    print(predict_d)
    printf("Accuracy (DTree): %f", accuracy_d)
elif accuracy_k >= accuracy_s:
    print(predict_k)
    printf("Accuracy (KNN): %f", accuracy_k)
else:
    print(predict_s)
    printf("Accuracy (SVM): %f", accuracy_s)
