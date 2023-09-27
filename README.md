# END TO END  RANDOM FOREST ALGORITHM
### In the following project we will be implementing a Random Forest Algorithm  and end toe nd model where we will be implementing the random forest regression algorithm and we will be deploying it using flask and fastapi
  We will fisrt be describing all the files that are in this project briefly explaining what each file perfoms and what it contains

  1. Templates - This file contains the frotn end code for the model we will be deploying using using flask we are using html,css  for the development  of the front end interface
  2. Gitignore -This is a file that is provided in github and one can also create it using vscode it is used to store files that you ant to 'ignore'   when committing the file to github there are files that dont need to be commited such as the environment that one will create ,there is  no need to commit all these files as theyy  actually wont be used thus this is where the gitignore comes in handy.
  3. Dockerfile -A  dockerfile is very important when deploying the app as a container for it to be used in teams using collaboration it somes in handy  when  simplyfing set up in diffferent  machines when used b different team members ,thus comes in handy while collaborating   with different team members.
  4. app.py-basically this is the fastapi framework that requires no front end modification to run ( formerly kmown as the swaggerapi
  5. flask.py -it is directly connected to the templates files where when the flask app is run it connects to the templates file and displays the frontend in the templates file
  6. Heart-disease.ipynb -it is a jupyter notebook that contains the ingestion of the data,the cleaning of the data,the building of the model and the fine tuning of our model which brings us to the other bit saving the model
  7. trained-model.sav- this file contains the file and  the model that we have saved and we connect it to the flask.py and to the app.py  where it will make predictions based on the file we have saved
  8. Requirements.txt -The requirements.txt file is used to load all the libraries that we will need for the model to run in the environment that we create in our vscode


## Project Overview
In the following notebook we will be predicting  the likelihood of a person of getting a heart disease based on the following factors
age -age in years sex -(1=male,0 = female) 

cp =chest pain type 0.Typical angina :Chest pain related decrease of blood supply to the heart 1.Typicalangina:chest pain not related to heart 2.Non -anginal pain:typically esophageal spasms (non heart related) Asymptomatic -Chest pains not showing signs of heart diease 

trestbps -resting blood presssure (in mm Hg on admission to the hospital) anything above 130-140 is typically a concern 

chol -serum cholestoral in mg/dl antything above 200 is a concern fbs -(fasting blood sugar>120mg/dl)(1=true;0=false) 7

restecg -resting electrocardiograph results Nothing to note ST-T Wave abnormality can range from mild to severe symptoms signals non normal heartbeat possible or definite left ventricular hypertrophy Enlarged hearts main pumping chamber

thalac =maximum heart rate achieved 

exang = exercise induced angina(1=yes , 2=no) 

old peak =ST depression induced by exercise relative to rest looks at stress of heart during exercise unhealthy heart will stress more 

slope =the slpoe of peak exercise ST segment 0.Upsloping:better heart rate with exercise( uncommon) 1.flatsloping;minmal change (typical healthy heart ) 2.downsloping -signs of an unhealthy heart

ca -number of major vessels (0-3)colored by flourosopy colored vessel means blood s passing through 

thal -thalium stress result 1,3.normal 6.fixed defect used to be defect but ow fixed 77.reversable effect no proper blood mvt while exercising 

target -have the disease or are healthy(1=yes,0=no)
## Data ingestion  and cleaning
We will have to import the following libraries to begin our project
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
%matplotlib inline
```
we will look for data that is misssing and our data is not missing any values (if we have columns that we have a significant number of null values we then fill the null missing values with median -this is done to prevent introductuion of skeweness to our data,howewver if only a small number of columns is missing then we can drop the columns)
We will aldo look for the info using the `df.info()` command this is done when examining our columns and look for thedifferent types of data in our columns 

## Explolatory Data Analysis
We will have to examine our data  but first since this is a classification problem we will need to look for the distribution of our target column to prevent bias in our data,If the data is not balanced between the two target outputs (yes and no) then we have to introduce smote to randomly sample a fixed number of target outputs.
```
df.describe()
df.target.value_counts()
```
since our target outputs are balance there is  no need to introduce smote

We will then b looking at individual columns and their effect towards the target variable we will start by dividing the columns to two  groups the categorical and non -numeric values  we will access the categorical columns using the following line of code
```
categorical_val =[]
continous_val =[]
for column in df.columns:
    if len(df[column].unique())<=10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
```
We will then create multiple subplots using the categorical  values and we will look at the effect of each categorical values on our target column:
```
import matplotlib.pyplot as plt

num_cols = 3
num_rows = (len(categorical_val) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for i, column in enumerate(categorical_val):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    df[df['target'] == 0][column].value_counts().plot(kind='bar', color='blue', label='Have a heart disease = NO', alpha=0.6, ax=ax)
    df[df['target'] == 1][column].value_counts().plot(kind='bar', color='red', label='Have a heart disease = Yes', alpha=0.6, ax=ax)
    ax.legend()
    ax.set_xlabel(column)

plt.show()
```
![aa](https://github.com/stilinsk/end-to-end-heart-prediction/assets/113185012/91303df3-d87f-414a-9914-db78babe21b2)


We also enumerate and plot the continous values  and look  atheir effects on their target columns:

```
plt.figure(figsize=(15,15))
for i,column in enumerate(continous_val ,1):
    
    plt.subplot(3,3,i)
    df[df['target']==0][column].hist(bins =35,color ='blue',label ='Have a heart disease = NO', alpha =0.6)
    df[df['target']==1][column].hist(bins =35,color ='red',label ='Have a heart disease = Yes', alpha =0.6)
    plt.legend()
    plt.xlabel(column)
```
![aaa](https://github.com/stilinsk/end-to-end-heart-prediction/assets/113185012/657f9785-8519-4f18-b286-4918821015c3)

## conclusions
1.cp =patients with a cp of 1,2 and 3 are more likely to have a heart disease than poeple with 0 

2.restecg(resting electrocardiograph results(patients with a value of 1 signals wave abnormality are more likely to have a heart disease. 

3.exang =(exercise induced angina)-patients with aa vlue 0(No =exercise induced angina)have a heart disease more than patients with 1(Yes =exercise induced angina)

4.slope = {the slope of the peak exercise ST segment} : People with slope value equal to 2 (Downslopins: signs of unhealthy heart) are more likely to have heart disease than people with slope value equal to 0 (Upsloping: better heart rate with excercise) or 1 (Flatsloping: minimal change (typical healthy heart)). 

5.ca {number of major vessels (0-3) colored by flourosopy} : the more blood movement the better so people with ca equal to 0 are more likely to have heart disease.

6.thal {thalium stress result} : People with thal value equal to 2 (fixed defect: used to be defect but ok now) are more likely to have heart disease.

1.trestbps =: resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern 

2.{serum cholestoral in mg/dl} : above 200 is cause for concern.

3.thalach {maximum heart rate achieved} : People how acheived a maximum more than 140 are more likely to have heart disease.

4.oldpeak ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more


We will  also need to plot a heatmap to look at how much the columns are correlated ,columns that are lessly correlated will have to be dropped  fro dimensionality reduction since there is no need for those columns to be included in our model ,this also comes in handy when looking at columns that have an inverse correlation
```
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
```
![aaaa](https://github.com/stilinsk/end-to-end-heart-prediction/assets/113185012/4f56f7c0-d279-4057-8f4e-acbbe1607cf9)


From the above heat map we will not be dropping any columns / we wont be having feature selection since most of our data is actually affecting our target variable.

## Model building
we are going to be  standardizing the data then create the  model and evaluate its perfomance ,we will look at the accuracy of the test and train sets and observe whether our data is overfitting and whether we need to tune it futher
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Separate the X and y data
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the training set X variables
X_train_standardized = scaler.fit_transform(X_train)

# Transform the test set X variables using the fitted scaler
X_test_standardized = scaler.transform(X_test)

# Create a Random Forest Classifier with specified parameters
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

# Fit the classifier to the standardized training data
rf_clf.fit(X_train_standardized, y_train)

# Predict the target variable for the standardized test set
y_pred = rf_clf.predict(X_test_standardized)

# Calculate and print the accuracy scores
test_score = accuracy_score(y_test, y_pred) * 100
train_score = accuracy_score(y_train, rf_clf.predict(X_train_standardized)) * 100

print("Training Accuracy: {:.2f}%".format(train_score))
print("Test Accuracy: {:.2f}%".format(test_score))
```
##### Training Accuracy: 100.00%
##### Test Accuracy: 86.89%
After training the test and train percentages we find that they have a very big diffference. means that our data is overfitting(learns more on our training data such that it cannot predict new data)

## Fine tuning our model

 #### we have introduced the following parameters to train our model
 n_estimators: This parameter represents the number of decision trees in the random forest. Increasing the number of estimators generally improves the performance of the model but also increases the computational complexity.

max_features: It determines the maximum number of features to consider when looking for the best split at each tree node. It can take different values, such as 'auto' (which considers all features), 'sqrt' (square root of the total number of features), or a specific integer value (e.g., 10 or 20).

max_depth: This parameter specifies the maximum depth of each decision tree in the random forest. A deeper tree can capture more complex relationships in the data, but it can also overfit. Setting it to None allows the trees to grow until all leaves are pure or contain a minimum number of samples.

min_samples_split: It represents the minimum number of samples required to split an internal node. If the number of samples at a node is below this threshold, the node will not be split further, resulting in a leaf node. Setting a higher value can help control overfitting.

min_samples_leaf: This parameter specifies the minimum number of samples required to be at a leaf node. If the number of samples at a leaf is below this threshold, the tree builder will attempt to split the parent node. Similar to min_samples_split, setting a higher value can help prevent overfitting.
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Define the parameter grid
n_estimators = [500, 900, 1100, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5, 10, 15, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

params_grid = {
    'n_estimators': n_estimators, 
    'max_features': max_features,
    'max_depth': max_depth, 
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

# Create the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform grid search to find the best parameters
rf_cv = GridSearchCV(rf_clf, params_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
rf_cv.fit(X_train_standardized, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

# Create the Random Forest Classifier with the best parameters
rf_clf = RandomForestClassifier(**best_params)

# Fit the classifier to the standardized training data
rf_clf.fit(X_train_standardized, y_train)

# Predict the target variable for the standardized test set
y_pred_test = rf_clf.predict(X_test_standardized)

# Calculate and print the accuracy scores
test_score = accuracy_score(y_test, y_pred_test) * 100
train_score = accuracy_score(y_train, rf_clf.predict(X_train_standardized)) * 100

print("Training Accuracy: {:.2f}%".format(train_score))
print("Test Accuracy: {:.2f}%".format(test_score))
```
##### Best parameters: {'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 1100}
##### Training Accuracy: 88.02%
##### Test Accuracy: 86.89%

After training we have to implement the grid searchcv to search for the best parameters and after the model searches the best parameters it will fit the parameters in our model and finally we have have the best output and results out of our model. as we can see the train accuracy is 88 % and test accuracy is almost 87%.
#### confusion matrix
```
# Generate the confusion matrix plot
def confusion_matrix_plotter(predictions, actuals):
    cf_matrix = confusion_matrix(predictions, actuals)
    group_names = ['True Neg', 'False Neg', 'False Pos', 'True Pos']  # Updated order
    group_counts = [f'{value:0.0f}' for value in cf_matrix.flatten()]
    group_percentages = [f'({value:.2%})' for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.show()

confusion_matrix_plotter(y_pred_test, y_test)
```
![aaaa](https://github.com/stilinsk/end-to-end-heart-prediction/assets/113185012/49e1c1da-d7a3-4cdd-b62c-01019fbc512d)

#### Other parameters include precison,recall and f1 score
```
# Calculate and print precision, recall, F1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1))
```
Precision: 85% of the samples predicted as positive are actually true positives.

Recall: 91% of the actual positive samples are correctly identified by the model.

F1-score: The F1-score is 88%, which represents a balanced measure of precision and recall.

We will then save our model using pickle

```
# saving the trained model
import pickle
filename ='model.sav'
pickle.dump(rf_clf ,open(filename,'wb'))
#loading the saved model
load=pickle.load(open('model.sav','rb'))
```
