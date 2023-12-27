# CLASSIFICATION-OF-CANCER-CELL-USING-KNEAREST-NEIGHBOR-MACHINE-LEARNING-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv('cancer_data.csv')
data.head()

plt.scatter(data['radius_mean'],data['diagnosis'])
plt.xlabel('radius_mean')
plt.ylabel('diagnosis')
plt.title('radius_mean VS diagnosis')

plt.scatter(data['perimeter_mean'],data['diagnosis'])
plt.xlabel('perimeter_mean')
plt.ylabel('diagnosis')
plt.title('perimeter_mean VS diagnosis')

plt.scatter(data['texture_mean'],data['diagnosis'])
plt.xlabel('texture_mean')
plt.ylabel('diagnosis')
plt.title('texture_mean VS diagnosis')

plt.scatter(data['compactness_mean'],data['diagnosis'])
plt.xlabel('compactness_mean')
plt.ylabel('diagnosis')
plt.title('compactness_mean VS diagnosis')

plt.scatter(data['concave points_mean'],data['diagnosis'])
plt.xlabel('concave points_mean')
plt.ylabel('diagnosis')
plt.title('concave points_mean VS diagnosis')

plt.scatter(data['concavity_mean'],data['diagnosis'])
plt.xlabel('concavity_mean')
plt.ylabel('diagnosis')
plt.title('concavity_mean VS diagnosis')

plt.scatter(data['symmetry_worst'],data['diagnosis'])
plt.xlabel('symmetry_worst')
plt.ylabel('diagnosis')
plt.title('symmetry_worst VS diagnosis')

# Data cleaning
print(len(data.index))
data.info()
# Data Splitting and Model Training
predct=dict(zip(data.diagnosis.unique(),data.radius_mean.unique()))
# predct
predct=dict(zip(data.diagnosis.unique(),data.smoothness_mean.unique()))
predct
predct=dict(zip(data.diagnosis.unique(),data.area_mean.unique()))
predct
predct=dict(zip(data.diagnosis.unique(),data.compactness_mean.unique()))
predct
# output
y=data['diagnosis'].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.03,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
md=KNeighborsClassifier()
md.fit(X_train,y_train)
X_test
# Model Evalutation
y_test
y_pred=md.predict(X_test)
y_pred
# Newdata set to insert and check where to lies
y1_pred=md.predict([[144,21.82,87.5,519.8,0.123,0.19,0.18,0.09,0.23,0.07,0.30]])
y1_pred
# How many give Accurate Answer
md.score(X_test,y_test)*100
