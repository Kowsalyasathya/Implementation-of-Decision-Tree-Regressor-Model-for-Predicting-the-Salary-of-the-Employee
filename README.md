# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets.

2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters.

3.Train your model -Fit model to training data -Calculate mean salary value for each subset.

4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance.

5.Tune hyperparameters -Experiment with different hyperparameters to improve performance.

6.Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KOWSALYA M
RegisterNumber:  212222230069
```
```
import pandas as pd
df=pd.read_csv("Salary.csv")

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()

x=df[['Position','Level']]
y=df['Salary']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
### Initial dataset:
![ML7 1](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/b15a2b8f-a961-4340-8079-56b845fffd80)
### Data info:
![ML7 2](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/1410d609-a19b-4a1f-9dcc-dd0a800c9ed4)
### Optimization of null values:
![ML7 3](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/5bc247f7-d77a-4492-b450-cb1201433817)
### Converting string literals to numericl values using label encoder:
![ML7 5](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/23514a1e-61b7-4dd0-9c08-69b34fbd7dc5)
### Mean Squared Error:
![ML7 4](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/cf70d7eb-76b6-4a28-b3ce-c334e9272b85)
### R2 variance:
![ML7 6](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/c2b36d0d-ad2d-4b41-ba20-af03feb59fe1)
### Prediction:
![ML7 7](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118671457/30f38613-ebce-4d46-8071-ce0b76f29605)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
