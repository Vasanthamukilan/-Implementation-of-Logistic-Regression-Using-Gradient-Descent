# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all necessary packages and dataset that you need to implement Logistic Regression using Gradient Descent.
2. Copy the actual dataset and remove fields which are unnecessary.
3. Then select dependent variable and independent variable from the dataset.
4. And perform Logistic Regression using Gradient Descent.
5. Print accuracy value, predicted value and actual values. 

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Vasanthamukilan M
RegisterNumber:212222230167
*/
import pandas as pd
import numpy as np
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
X=data1.iloc[:,:-1]
Y=data1['status']
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,x,y):
  h=sigmoid(x.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=X.T.dot(h-y)
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

## Output:

i.) Dataset:

![image](https://github.com/shoaib3136/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117919362/2be11e7b-742d-4b1e-ae84-fcf2a4e5fa15)

ii.) Accuracy and actual,predicted values:

![image](https://github.com/shoaib3136/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117919362/f672e017-619f-4b17-b66b-b31b20b90909)

![image](https://github.com/shoaib3136/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/117919362/1cf7f40a-f62b-4a1f-9afc-6ac24f10a5ae)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
