# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the Iris Dataset

Step 3. Separate features (x) and labels (y), then split the data into training and test sets using train_test_split().

Step 4. Initialize an SGDClassifier and fit it on the training data (x_train, y_train).

Step 5. Use the trained model to predict the labels for the test set and calculate the accuracy score.

Step 6. Generate and print a confusion matrix to evaluate the model's performance.

Step 7. End

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: nagalakshmi s
RegisterNumber:  25003017
*/
```
```
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data=load_iris()
x,y=data.data,data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(multi_class='multinomial',solver='lbfgs')
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:2f}")

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="201" height="36" alt="Screenshot 2026-02-06 112214" src="https://github.com/user-attachments/assets/872e571c-d44c-4cd4-b5cb-6c09ec2e0127" />
<img width="193" height="102" alt="Screenshot 2026-02-06 112223" src="https://github.com/user-attachments/assets/6e3d13aa-6a67-4d2a-908c-cbd09b3c60ac" />





## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
