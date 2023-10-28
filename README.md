# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model. 2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.
3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function. 
4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters. 
5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.
6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm. 
7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.
8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Bala murugan P
RegisterNumber: 212222230017 
*/
```
```

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/b4a87e90-d9da-4ccc-8092-c345a3ae5304)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/d28cd559-8abd-44c3-bf84-a2fe3eba8522)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/3c1e7f6b-5bd9-46ed-9996-17890d2ddb8d)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/44d837f1-ae80-45e3-838c-6603d9262d00)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/bea67387-c4ca-445c-9014-ce08e10ba009)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/c20bf12c-95eb-41a8-8849-5bddfd9ef081)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/d96919ae-86e6-4887-a110-fb7eb330f96c)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/73725c4e-3b8d-4c00-87dc-bafc37539543)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/c10aa1ff-ab16-4494-97e8-8449ad2df0d4)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/63df3204-07be-4340-b193-162609dda965)

![image](https://github.com/Bala1511/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680410/27d1a313-a4de-470c-b85c-e06d69651c06)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
