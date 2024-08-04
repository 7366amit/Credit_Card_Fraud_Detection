# Credit_Card_Fraud_Detection
 the models we build found that the XGBOOST model with Random Oversampling with StratifiedKFold CV gave us the best accuracy and ROC on oversampled data
<br>
<h1>Problem Statement:</h1>
<br>
<p> For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

In the banking industry, credit card fraud detection using machine learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.

In this project we will detect fraudulent credit card transactions with the help of Machine learning models. We will analyse customer-level data that has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group.</p>
<br>

<h1>Data Understanding </h1>
<br>
<p>The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with principal component analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value of 1 in cases of fraud and 0 in others.</p>

<h3>Table of Contents</h3>
<p>
1. Importing dependencies <br>
2. Exploratory data analysis <br>
3. Splitting the data into train & test data <br>
4. Model Building<br>
    -- Perform cross validation with RepeatedKFold <br>
    -- Perform cross validation with StratifiedKFold <br>
    -- RandomOverSampler with StratifiedKFold Cross Validation <br>
    -- Oversampling with SMOTE Oversampling <br>
    -- Oversampling with ADASYN Oversampling <br>
5. Hyperparameter Tuning <br>
6. Conclusion <br>
</p>
Skip to main content
Credit_Card_Fraud_Detection.ipynb
Credit_Card_Fraud_Detection.ipynb_Notebook unstarred
All changes saved
Credit Card Fraud Detection


# Importing Drive (Dataset.csv) 
[ ]
from google.colab import drive
drive.mount('/content/drive')
Double-click (or enter) to edit

Importing Dependencies

[ ]
# Importing the libraries
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import AdaBoostClassifier

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
Exploratory data analysis

[ ]
# Mounting the google drive
from google.colab import drive
drive.mount('/content/gdrive')
Mounted at /content/gdrive

[ ]
# Loading the data
df = pd.read_csv('gdrive/MyDrive/Colab Notebooks/creditcard.csv')
# df = pd.read_csv('./data/creditcard.csv')
df.head()


[ ]
# Checking the shape
df.shape
(284807, 31)

[ ]
# Checking the datatypes and null/non-null distribution
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB

[ ]
# Checking distribution of numerical values in the dataset
df.describe()


[ ]
# Checking the class distribution of the target variable
df['Class'].value_counts()
0    284315
1       492
Name: Class, dtype: int64

[ ]
# Checking the class distribution of the target variable in percentage
print((df.groupby('Class')['Class'].count()/df['Class'].count()) *100)
((df.groupby('Class')['Class'].count()/df['Class'].count()) *100).plot.pie()


[ ]
# Checking the correlation
corr = df.corr()
corr


[ ]
# Checking the correlation in heatmap
plt.figure(figsize=(24,18))

sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()

Here we will observe the distribution of our classes


[ ]
# Checking the % distribution of normal vs fraud
classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100

print(normal_share)
print(fraud_share)
99.82725143693798
0.1727485630620034

[ ]
# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
plt.figure(figsize=(7,5))
sns.countplot(df['Class'])
plt.title("Class Count", fontsize=18)
plt.xlabel("Record counts by class", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


[ ]
# As time is given in relative fashion, we are using pandas.Timedelta which Represents a duration, the difference between two times or dates.
Delta_Time = pd.to_timedelta(df['Time'], unit='s')

#Create derived columns Mins and hours
df['Time_Day'] = (Delta_Time.dt.components.days).astype(int)
df['Time_Hour'] = (Delta_Time.dt.components.hours).astype(int)
df['Time_Min'] = (Delta_Time.dt.components.minutes).astype(int)

[ ]
# Drop unnecessary columns
# We will drop Time,as we have derived the Day/Hour/Minutes from the time column 
df.drop('Time', axis = 1, inplace= True)
# We will keep only derived column hour, as day/minutes might not be very useful
df.drop(['Time_Day', 'Time_Min'], axis = 1, inplace= True)

Splitting the data into train & test data

[ ]
# Splitting the dataset into X and y
y= df['Class']
X = df.drop(['Class'], axis=1)

[ ]
# Checking some rows of X
X.head()


[ ]
# Checking some rows of y
y.head()
0    0
1    0
2    0
3    0
4    0
Name: Class, dtype: int64

[ ]
# Splitting the dataset using train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.20)
Preserve X_test & y_test to evaluate on the test data once you build the model


[ ]
# Checking the spread of data post split
print(np.sum(y))
print(np.sum(y_train))
print(np.sum(y_test))
492
396
96
Plotting the distribution of a variable


[ ]
# Accumulating all the column names under one variable
cols = list(X.columns.values)

[ ]
# plot the histogram of a variable from the dataset to see the skewness
normal_records = df.Class == 0
fraud_records = df.Class == 1

plt.figure(figsize=(20, 60))
for n, col in enumerate(cols):
  plt.subplot(10,3,n+1)
  sns.distplot(X[col][normal_records], color='green')
  sns.distplot(X[col][fraud_records], color='red')
  plt.title(col, fontsize=17)
plt.show()

Model Building

[ ]
#Create a dataframe to store results
df_Results = pd.DataFrame(columns=['Methodology','Model','Accuracy','roc_value','threshold'])

[ ]
# Created a common function to plot confusion matrix
def Plot_confusion_matrix(y_test, pred_test):
  cm = confusion_matrix(y_test, pred_test)
  plt.clf()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
  categoryNames = ['Non-Fraudalent','Fraudalent']
  plt.title('Confusion Matrix - Test Data')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  ticks = np.arange(len(categoryNames))


[ ]
# # Created a common function to fit and predict on a Logistic Regression model for both L1 and L2
def buildAndRunLogisticModels(df_Results, Methodology, X_train,y_train, X_test, y_test ):

  # Logistic Regression
  from sklearn import linear_model
  from sklearn.model_selection import KFold

  num_C = list(np.power(10.0, np.arange(-10, 10)))
  cv_num = KFold(n_splits=10, shuffle=True, random_state=42)



[ ]
# Created a common function to fit and predict on a KNN model
def buildAndRunKNNModels(df_Results,Methodology, X_train,y_train, X_test, y_test ):

  #create KNN model and fit the model with train dataset
  knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)
  knn.fit(X_train,y_train)
  score = knn.score(X_test,y_test)
  print("model score")
  print(score)
  


[ ]
# Created a common function to fit and predict on a Tree models for both gini and entropy criteria
def buildAndRunTreeModels(df_Results, Methodology, X_train,y_train, X_test, y_test ):
  #Evaluate Decision Tree model with 'gini' & 'entropy'
  criteria = ['gini', 'entropy'] 
  scores = {} 
    
  for c in criteria: 
      dt = DecisionTreeClassifier(criterion = c, random_state=42) 
      dt.fit(X_train, y_train) 
      y_pred = dt.predict(X_test)
      test_score = dt.score(X_test, y_test) 
      tree_preds = dt.predict_proba(X_test)[:, 1]
      tree_roc_value = roc_auc_score(y_test, tree_preds)
      scores = test_score 
      print(c + " score: {0}" .format(test_score))
      print("Confusion Matrix")
      Plot_confusion_matrix(y_test, y_pred)
      print("classification Report")
      print(classification_report(y_test, y_pred))
      print(c + " tree_roc_value: {0}" .format(tree_roc_value))
      fpr, tpr, thresholds = metrics.roc_curve(y_test, tree_preds)
      threshold = thresholds[np.argmax(tpr-fpr)]
      print("Tree threshold: {0}".format(threshold))
      roc_auc = metrics.auc(fpr, tpr)
      print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
      plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
      plt.legend(loc=4)
      plt.show()
  
      df_Results = df_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'Tree Model with {0} criteria'.format(c),'Accuracy': test_score,'roc_value': tree_roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  return df_Results

[ ]
# Created a common function to fit and predict on a Random Forest model
def buildAndRunRandomForestModels(df_Results, Methodology, X_train,y_train, X_test, y_test ):
  #Evaluate Random Forest model

  # Create the model with 100 trees
  RF_model = RandomForestClassifier(n_estimators=100, 
                                bootstrap = True,
                                max_features = 'sqrt', random_state=42)
  # Fit on training data
  RF_model.fit(X_train, y_train)
  RF_test_score = RF_model.score(X_test, y_test)
  RF_model.predict(X_test)

  print('Model Accuracy: {0}'.format(RF_test_score))


  # Actual class predictions
  rf_predictions = RF_model.predict(X_test)

  print("Confusion Matrix")
  Plot_confusion_matrix(y_test, rf_predictions)
  print("classification Report")
  print(classification_report(y_test, rf_predictions))

  # Probabilities for each class
  rf_probs = RF_model.predict_proba(X_test)[:, 1]

  # Calculate roc auc
  roc_value = roc_auc_score(y_test, rf_probs)

  print("Random Forest roc_value: {0}" .format(roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(y_test, rf_probs)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("Random Forest threshold: {0}".format(threshold))
  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  df_Results = df_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'Random Forest','Accuracy': RF_test_score,'roc_value': roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  return df_Results

[ ]
# Created a common function to fit and predict on a XGBoost model
def buildAndRunXGBoostModels(df_Results, Methodology,X_train,y_train, X_test, y_test ):
  #Evaluate XGboost model
  XGBmodel = XGBClassifier(random_state=42)
  XGBmodel.fit(X_train, y_train)
  y_pred = XGBmodel.predict(X_test)

  XGB_test_score = XGBmodel.score(X_test, y_test)
  print('Model Accuracy: {0}'.format(XGB_test_score))

  print("Confusion Matrix")
  Plot_confusion_matrix(y_test, y_pred)
  print("classification Report")
  print(classification_report(y_test, y_pred))
  # Probabilities for each class
  XGB_probs = XGBmodel.predict_proba(X_test)[:, 1]

  # Calculate roc auc
  XGB_roc_value = roc_auc_score(y_test, XGB_probs)

  print("XGboost roc_value: {0}" .format(XGB_roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(y_test, XGB_probs)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("XGBoost threshold: {0}".format(threshold))
  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  df_Results = df_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'XGBoost','Accuracy': XGB_test_score,'roc_value': XGB_roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  return df_Results


[ ]
# Created a common function to fit and predict on a SVM model
def buildAndRunSVMModels(df_Results, Methodology, X_train,y_train, X_test, y_test ):
  #Evaluate SVM model with sigmoid kernel  model
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import roc_auc_score

  clf = SVC(kernel='sigmoid', random_state=42)
  clf.fit(X_train,y_train)
  y_pred_SVM = clf.predict(X_test)
  SVM_Score = accuracy_score(y_test,y_pred_SVM)
  print("accuracy_score : {0}".format(SVM_Score))
  print("Confusion Matrix")
  Plot_confusion_matrix(y_test, y_pred_SVM)
  print("classification Report")
  print(classification_report(y_test, y_pred_SVM))

  # Run classifier
  classifier = SVC(kernel='sigmoid' , probability=True)
  svm_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]

  # Calculate roc auc
  roc_value = roc_auc_score(y_test, svm_probs)
  
  print("SVM roc_value: {0}" .format(roc_value))
  fpr, tpr, thresholds = metrics.roc_curve(y_test, svm_probs)
  threshold = thresholds[np.argmax(tpr-fpr)]
  print("SVM threshold: {0}".format(threshold))
  roc_auc = metrics.auc(fpr, tpr)
  print("ROC for the test dataset",'{:.1%}'.format(roc_auc))
  plt.plot(fpr,tpr,label="Test, auc="+str(roc_auc))
  plt.legend(loc=4)
  plt.show()
  
  df_Results = df_Results.append(pd.DataFrame({'Methodology': Methodology,'Model': 'SVM','Accuracy': SVM_Score,'roc_value': roc_value,'threshold': threshold}, index=[0]),ignore_index= True)

  return df_Results
Build different models on the imbalanced dataset and see the result
Perform cross validation with RepeatedKFold

[ ]
#Lets perfrom RepeatedKFold and check the results
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in rkf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     5      6     12 ... 284791 284796 284803]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [    10     11     15 ... 284785 284793 284795]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     4     18     24 ... 284788 284794 284800]
TRAIN: [     0      1      2 ... 284803 284805 284806] TEST: [     3      7     13 ... 284799 284801 284804]
TRAIN: [     3      4      5 ... 284801 284803 284804] TEST: [     0      1      2 ... 284802 284805 284806]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     7      8      9 ... 284795 284797 284802]
TRAIN: [     0      1      2 ... 284802 284803 284804] TEST: [     4      6     15 ... 284796 284805 284806]
TRAIN: [     0      1      3 ... 284804 284805 284806] TEST: [     2     10     11 ... 284793 284801 284803]
TRAIN: [     2      3      4 ... 284804 284805 284806] TEST: [     0      1     17 ... 284782 284790 284794]
TRAIN: [     0      1      2 ... 284803 284805 284806] TEST: [     3      5     12 ... 284799 284800 284804]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     3      7      8 ... 284791 284792 284797]
TRAIN: [     0      2      3 ... 284803 284804 284805] TEST: [     1     12     19 ... 284785 284800 284806]
TRAIN: [     0      1      3 ... 284803 284805 284806] TEST: [     2      4      9 ... 284801 284802 284804]
TRAIN: [     0      1      2 ... 284802 284804 284806] TEST: [     5      6     11 ... 284793 284803 284805]
TRAIN: [     1      2      3 ... 284804 284805 284806] TEST: [     0     17     26 ... 284794 284796 284799]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     3      9     28 ... 284798 284800 284801]
TRAIN: [     1      3      4 ... 284804 284805 284806] TEST: [     0      2      7 ... 284787 284795 284796]
TRAIN: [     0      2      3 ... 284803 284804 284806] TEST: [     1      6      8 ... 284799 284802 284805]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     4      5     16 ... 284779 284781 284784]
TRAIN: [     0      1      2 ... 284801 284802 284805] TEST: [    11     13     14 ... 284803 284804 284806]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [    12     13     14 ... 284788 284790 284800]
TRAIN: [     0      2      3 ... 284804 284805 284806] TEST: [     1      5      6 ... 284793 284795 284796]
TRAIN: [     0      1      2 ... 284803 284804 284806] TEST: [     4      7      8 ... 284794 284798 284805]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [    10     11     15 ... 284787 284801 284802]
TRAIN: [     1      4      5 ... 284801 284802 284805] TEST: [     0      2      3 ... 284803 284804 284806]
TRAIN: [     0      1      3 ... 284804 284805 284806] TEST: [     2      4      5 ... 284796 284801 284802]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     8     13     15 ... 284794 284797 284800]
TRAIN: [     0      1      2 ... 284802 284804 284806] TEST: [     9     14     21 ... 284790 284803 284805]
TRAIN: [     0      2      4 ... 284803 284805 284806] TEST: [     1      3      6 ... 284792 284799 284804]
TRAIN: [     1      2      3 ... 284803 284804 284805] TEST: [     0     11     12 ... 284787 284798 284806]
TRAIN: [     0      1      2 ... 284803 284805 284806] TEST: [     3     12     15 ... 284800 284802 284804]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     5     22     25 ... 284782 284789 284792]
TRAIN: [     1      2      3 ... 284804 284805 284806] TEST: [     0     13     18 ... 284794 284796 284799]
TRAIN: [     0      3      5 ... 284802 284804 284806] TEST: [     1      2      4 ... 284801 284803 284805]
TRAIN: [     0      1      2 ... 284803 284804 284805] TEST: [     7      8      9 ... 284797 284798 284806]
TRAIN: [     1      2      3 ... 284803 284804 284806] TEST: [     0      8     24 ... 284793 284794 284805]
TRAIN: [     0      1      2 ... 284803 284804 284805] TEST: [     9     10     13 ... 284782 284796 284806]
TRAIN: [     0      3      4 ... 284804 284805 284806] TEST: [     1      2      6 ... 284799 284800 284803]
TRAIN: [     0      1      2 ... 284803 284805 284806] TEST: [     3      5      7 ... 284786 284802 284804]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     4     11     12 ... 284790 284792 284801]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     6      8     20 ... 284792 284801 284802]
TRAIN: [     0      2      3 ... 284804 284805 284806] TEST: [     1      4      9 ... 284793 284800 284803]
TRAIN: [     1      4      6 ... 284803 284805 284806] TEST: [     0      2      3 ... 284789 284798 284804]
TRAIN: [     0      1      2 ... 284803 284804 284806] TEST: [     7     13     14 ... 284796 284797 284805]
TRAIN: [     0      1      2 ... 284803 284804 284805] TEST: [    17     30     35 ... 284788 284799 284806]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     6     17     20 ... 284790 284791 284800]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [     5     13     15 ... 284799 284801 284803]
TRAIN: [     0      4      5 ... 284803 284805 284806] TEST: [     1      2      3 ... 284795 284802 284804]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [    16     26     27 ... 284786 284787 284796]
TRAIN: [     1      2      3 ... 284802 284803 284804] TEST: [     0      4     11 ... 284797 284805 284806]

[ ]
#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results,"RepeatedKFold Cross Validation", X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run KNN Model
print("KNN Model")
start_time = time.time()
df_Results = buildAndRunKNNModels(df_Results,"RepeatedKFold Cross Validation",X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run Decision Tree Models with  'gini' & 'entropy' criteria
print("Decision Tree Models with  'gini' & 'entropy' criteria")
start_time = time.time()
df_Results = buildAndRunTreeModels(df_Results,"RepeatedKFold Cross Validation",X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run Random Forest Model
print("Random Forest Model")
start_time = time.time()
df_Results = buildAndRunRandomForestModels(df_Results,"RepeatedKFold Cross Validation",X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run XGBoost Modela
print("XGBoost Model")
start_time = time.time()
df_Results = buildAndRunXGBoostModels(df_Results,"RepeatedKFold Cross Validation",X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run SVM Model with Sigmoid Kernel
print("SVM Model with Sigmoid Kernel")
start_time = time.time()
df_Results = buildAndRunSVMModels(df_Results,"RepeatedKFold Cross Validation",X_train_cv,y_train_cv, X_test_cv, y_test_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))


[ ]
# Checking the df_result dataframe which contains consolidated results of all the runs
df_Results

Results for cross validation with RepeatedKFold:
Looking at Accuracy and ROC value we have "Logistic Regression with L2 Regularisation" which has provided best results for cross validation with RepeatedKFold technique

Perform cross validation with StratifiedKFold

[ ]
#Lets perfrom StratifiedKFold and check the results
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train_SKF_cv, X_test_SKF_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_SKF_cv, y_test_SKF_cv = y.iloc[train_index], y.iloc[test_index]
TRAIN: [ 30473  30496  31002 ... 284804 284805 284806] TEST: [    0     1     2 ... 57017 57018 57019]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [ 30473  30496  31002 ... 113964 113965 113966]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [ 81609  82400  83053 ... 170946 170947 170948]
TRAIN: [     0      1      2 ... 284804 284805 284806] TEST: [150654 150660 150661 ... 227866 227867 227868]
TRAIN: [     0      1      2 ... 227866 227867 227868] TEST: [212516 212644 213092 ... 284804 284805 284806]

[ ]
#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results,"StratifiedKFold Cross Validation", X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run KNN Model
print("KNN Model")
start_time = time.time()
df_Results = buildAndRunKNNModels(df_Results,"StratifiedKFold Cross Validation",X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run Decision Tree Models with  'gini' & 'entropy' criteria
print("Decision Tree Models with  'gini' & 'entropy' criteria")
start_time = time.time()
df_Results = buildAndRunTreeModels(df_Results,"StratifiedKFold Cross Validation",X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run Random Forest Model
print("Random Forest Model")
start_time = time.time()
df_Results = buildAndRunRandomForestModels(df_Results,"StratifiedKFold Cross Validation",X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run XGBoost Modela
print("XGBoost Model")
start_time = time.time()
df_Results = buildAndRunXGBoostModels(df_Results,"StratifiedKFold Cross Validation",X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run SVM Model with Sigmoid Kernel
print("SVM Model with Sigmoid Kernel")
start_time = time.time()
df_Results = buildAndRunSVMModels(df_Results,"StratifiedKFold Cross Validation",X_train_SKF_cv,y_train_SKF_cv, X_test_SKF_cv, y_test_SKF_cv)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))


[ ]
# Checking the df_result dataframe which contains consolidated results of all the runs
df_Results

Results for cross validation with StratifiedKFold:
Looking at the ROC value we have Logistic Regression with L2 Regularisation has provided best results for cross validation with StratifiedKFold technique

Conclusion :
As the results show Logistic Regression with L2 Regularisation for StratifiedKFold cross validation provided best results
Proceed with the model which shows the best result
Apply the best hyperparameter on the model
Predict on the test dataset

[ ]
# Logistic Regression
from sklearn import linear_model #import the package
from sklearn.model_selection import KFold

num_C = list(np.power(10.0, np.arange(-10, 10)))
cv_num = KFold(n_splits=10, shuffle=True, random_state=42)

clf = linear_model.LogisticRegressionCV(
          Cs= num_C
          ,penalty='l2'

Max auc_roc for l2: 0.9865741449266722
Parameters for l2 regularisations
[[ 8.81955672e-03  4.12852966e-02 -8.73597273e-02  2.31593930e-01
   8.23465655e-02 -5.16810590e-02 -4.00236018e-02 -1.21892419e-01
  -8.37934902e-02 -1.88393098e-01  1.47808325e-01 -2.13768201e-01
  -3.73003674e-02 -3.80938168e-01 -4.79315972e-03 -1.05538339e-01
  -9.33039347e-02 -4.43804397e-03  1.14866801e-02 -7.35149625e-03
   4.44308093e-02  3.04505742e-02 -8.51460452e-03 -1.50983112e-02
  -6.56644541e-03  5.37855175e-03 -8.33077400e-03 -6.17909130e-05
   3.22049047e-04  9.96568363e-03]]
[-7.6233989]
{1: array([[0.53980902, 0.54182146, 0.56634264, 0.74371077, 0.91438765,
        0.95390987, 0.96803901, 0.97816813, 0.96897688, 0.95944719,
        0.95944719, 0.95485347, 0.95485347, 0.95485347, 0.95485347,
        0.95485347, 0.95485347, 0.95485347, 0.95485347, 0.95485347],
       [0.58448142, 0.58599444, 0.60685041, 0.76199495, 0.92335672,
        0.94993447, 0.96479963, 0.96984747, 0.97499254, 0.9766534 ,
        0.97621921, 0.97621921, 0.97621921, 0.97621921, 0.97621921,
        0.97621921, 0.97621921, 0.97621921, 0.97621921, 0.97621921],
       [0.61346624, 0.61476302, 0.63158702, 0.75831428, 0.8961028 ,
        0.93985439, 0.97944884, 0.98794782, 0.98007622, 0.97403205,
        0.97371054, 0.97371054, 0.97371054, 0.97371054, 0.97371054,
        0.97371054, 0.97371054, 0.97371054, 0.97371054, 0.97371054],
       [0.61268627, 0.61429074, 0.63028167, 0.76311901, 0.9009578 ,
        0.92148495, 0.96233168, 0.98011512, 0.97796851, 0.98070367,
        0.98070367, 0.98070367, 0.98070367, 0.98070367, 0.98070367,
        0.98070367, 0.98070367, 0.98070367, 0.98070367, 0.98070367],
       [0.58405655, 0.58578797, 0.6052037 , 0.7573307 , 0.92823764,
        0.97402797, 0.99266764, 0.99176292, 0.9868733 , 0.98227916,
        0.98227916, 0.98227916, 0.98227916, 0.98227916, 0.98227916,
        0.98227916, 0.98227916, 0.98227916, 0.98227916, 0.98227916],
       [0.59340887, 0.59567696, 0.621504  , 0.78234736, 0.92556162,
        0.95741592, 0.98566669, 0.98682866, 0.98212257, 0.97861291,
        0.97802242, 0.97802242, 0.97802242, 0.97802242, 0.97802242,
        0.97802242, 0.97802242, 0.97802242, 0.97802242, 0.97802242],
       [0.73007433, 0.7318635 , 0.75021658, 0.8662425 , 0.96029462,
        0.9708664 , 0.98480934, 0.98428829, 0.98343451, 0.98508243,
        0.98508243, 0.98508243, 0.98508243, 0.98508243, 0.98508243,
        0.98508243, 0.98508243, 0.98508243, 0.98508243, 0.98508243],
       [0.53912527, 0.54116075, 0.56759131, 0.76047758, 0.94354308,
        0.98492946, 0.99690329, 0.99822171, 0.99787561, 0.99667096,
        0.99667096, 0.99667096, 0.99667096, 0.99667096, 0.99667096,
        0.99667096, 0.99667096, 0.99667096, 0.99667096, 0.99667096],
       [0.60207428, 0.6013821 , 0.62200089, 0.76699866, 0.92923325,
        0.97203443, 0.99246834, 0.99463957, 0.99135228, 0.98242386,
        0.98242386, 0.98242386, 0.98242386, 0.98242386, 0.98242386,
        0.98242386, 0.98242386, 0.98242386, 0.98242386, 0.98242386],
       [0.58410202, 0.58567802, 0.60373999, 0.74236053, 0.89363762,
        0.94870851, 0.98802385, 0.99392176, 0.99312315, 0.99088946,
        0.99088946, 0.99088946, 0.99088946, 0.99088946, 0.99088946,
        0.99088946, 0.99088946, 0.99088946, 0.99088946, 0.99088946]])}
Accuarcy of Logistic model with l2 regularisation : 0.9988764439450862
l2 roc_value: 0.9765679037855075
l2 threshold: 0.0010284453371876342

[ ]
# Checking for the coefficient values
clf.coef_
array([[ 8.81955672e-03,  4.12852966e-02, -8.73597273e-02,
         2.31593930e-01,  8.23465655e-02, -5.16810590e-02,
        -4.00236018e-02, -1.21892419e-01, -8.37934902e-02,
        -1.88393098e-01,  1.47808325e-01, -2.13768201e-01,
        -3.73003674e-02, -3.80938168e-01, -4.79315972e-03,
        -1.05538339e-01, -9.33039347e-02, -4.43804397e-03,
         1.14866801e-02, -7.35149625e-03,  4.44308093e-02,
         3.04505742e-02, -8.51460452e-03, -1.50983112e-02,
        -6.56644541e-03,  5.37855175e-03, -8.33077400e-03,
        -6.17909130e-05,  3.22049047e-04,  9.96568363e-03]])

[ ]
# Creating a dataframe with the coefficient values
coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(clf.coef_))], axis = 1)
coefficients.columns = ['Feature','Importance Coefficient']

[ ]
coefficients

Print the important features of the best model to understand the dataset
This will not give much explanation on the already transformed dataset
But it will help us in understanding if the dataset is not PCA transformed

[ ]
# Plotting the coefficient values
plt.figure(figsize=(20,5))
sns.barplot(x='Feature', y='Importance Coefficient', data=coefficients)
plt.title("Logistic Regression with L2 Regularisation Feature Importance", fontsize=18)

plt.show()

Hence it implies that V4, v5,V11 has + ve importance whereas V10, V12, V14 seems to have -ve impact on the predictaions

Model building with balancing Classes
Perform class balancing with :
Random Oversampling
SMOTE
ADASYN
Oversampling with RandomOverSampler with StratifiedKFold Cross Validation
We will use Random Oversampling method to handle the class imbalance

[ ]
# Creating the dataset with RandomOverSampler and StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

skf = StratifiedKFold(n_splits=5, random_state=None)

for fold, (train_index, test_index) in enumerate(skf.split(X,y), 1):
    X_train = X.loc[train_index]
    y_train = y.loc[train_index] 
    X_test = X.loc[test_index]


[ ]
Data_Imbalance_Handiling	 = "Random Oversampling with StratifiedKFold CV "
#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results , Data_Imbalance_Handiling , X_over, y_over, X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*60 )

#Run KNN Model
print("KNN Model")



[ ]
# Checking the df_result dataframe which contains consolidated results of all the runs
df_Results

Results for Random Oversampling with StratifiedKFold technique:
Looking at the Accuracy and ROC value we have XGBoost which has provided best results for Random Oversampling and StratifiedKFold technique

Oversampling with SMOTE Oversampling
We will use SMOTE Oversampling method to handle the class imbalance

[ ]
# Creating dataframe with Smote and StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from imblearn import over_sampling

skf = StratifiedKFold(n_splits=5, random_state=None)

for fold, (train_index, test_index) in enumerate(skf.split(X,y), 1):
    X_train = X.loc[train_index]
    y_train = y.loc[train_index] 
    X_test = X.loc[test_index]
    y_test = y.loc[test_index]  
    SMOTE = over_sampling.SMOTE(random_state=0)
    X_train_Smote, y_train_Smote= SMOTE.fit_resample(X_train, y_train)
  
X_train_Smote = pd.DataFrame(data=X_train_Smote,   columns=cols)

[ ]
Data_Imbalance_Handiling	 = "SMOTE Oversampling with StratifiedKFold CV "
#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results, Data_Imbalance_Handiling, X_train_Smote, y_train_Smote , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )

#Run KNN Model
print("KNN Model")



[ ]
# Checking the df_result dataframe which contains consolidated results of all the runs
df_Results

Results for SMOTE Oversampling with StratifiedKFold:
Looking at Accuracy and ROC value we have XGBoost which has provided best results for SMOTE Oversampling with StratifiedKFold technique

Oversampling with ADASYN Oversampling
We will use ADASYN Oversampling method to handle the class imbalance

[ ]
# Creating dataframe with ADASYN and StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from imblearn import over_sampling

skf = StratifiedKFold(n_splits=5, random_state=None)

for fold, (train_index, test_index) in enumerate(skf.split(X,y), 1):
    X_train = X.loc[train_index]
    y_train = y.loc[train_index] 
    X_test = X.loc[test_index]


[ ]
Data_Imbalance_Handiling     = "ADASYN Oversampling with StratifiedKFold CV "
#Run Logistic Regression with L1 And L2 Regularisation
print("Logistic Regression with L1 And L2 Regularisation")
start_time = time.time()
df_Results = buildAndRunLogisticModels(df_Results, Data_Imbalance_Handiling, X_train_ADASYN, y_train_ADASYN , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )

#Run KNN Model
print("KNN Model")
start_time = time.time()
df_Results = buildAndRunKNNModels(df_Results, Data_Imbalance_Handiling,X_train_ADASYN, y_train_ADASYN , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )

#Run Decision Tree Models with  'gini' & 'entropy' criteria
print("Decision Tree Models with  'gini' & 'entropy' criteria")
start_time = time.time()
df_Results = buildAndRunTreeModels(df_Results, Data_Imbalance_Handiling,X_train_ADASYN, y_train_ADASYN , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )

#Run Random Forest Model
print("Random Forest Model")
start_time = time.time()
df_Results = buildAndRunRandomForestModels(df_Results, Data_Imbalance_Handiling,X_train_ADASYN, y_train_ADASYN , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )

#Run XGBoost Model
print("XGBoost Model")
start_time = time.time()
df_Results = buildAndRunXGBoostModels(df_Results, Data_Imbalance_Handiling,X_train_ADASYN, y_train_ADASYN , X_test, y_test)
print("Time Taken by Model: --- %s seconds ---" % (time.time() - start_time))
print('-'*80 )


[ ]
# Checking the df_result dataframe which contains consolidated results of all the runs
df_Results

Results for ADASYN Oversampling with StratifiedKFold:
Looking at Accuracy and ROC value we have XGBoost which has provided best results for ADASYN Oversampling with StratifiedKFold technique

Overall conclusion after running the models on Oversampled data :
Looking at above results it seems XGBOOST model with Random Oversampling with StratifiedKFold CV has provided the best results under the category of all oversampling techniques. So we will try to tune the hyperparameters of this model to get best results.

Hyperparameter Tuning
HPT - Xgboost Regression

[ ]
# Performing Hyperparameter tuning
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_test = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2),
 'n_estimators':range(60,130,150),
 'learning_rate':[0.05,0.1,0.125,0.15,0.2],
 'gamma':[i/10.0 for i in range(0,5)],
 'subsample':[i/10.0 for i in range(7,10)],

({'mean_fit_time': array([ 78.0790154 ,  89.69091663, 113.30138698,  41.1210391 ,
         124.47327914]),
  'mean_score_time': array([0.33167305, 0.39689088, 0.43635864, 0.25277925, 0.39153495]),
  'mean_test_score': array([0.99737639, 0.9998794 , 0.99985979, 0.99902151, 0.99986951]),
  'param_colsample_bytree': masked_array(data=[0.9, 0.7, 0.7, 0.7, 0.9],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_gamma': masked_array(data=[0.2, 0.2, 0.0, 0.4, 0.0],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_learning_rate': masked_array(data=[0.15, 0.125, 0.1, 0.2, 0.125],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_max_depth': masked_array(data=[5, 7, 9, 3, 9],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_min_child_weight': masked_array(data=[3, 5, 3, 3, 1],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_n_estimators': masked_array(data=[60, 60, 60, 60, 60],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'param_subsample': masked_array(data=[0.7, 0.8, 0.9, 0.8, 0.7],
               mask=[False, False, False, False, False],
         fill_value='?',
              dtype=object),
  'params': [{'colsample_bytree': 0.9,
    'gamma': 0.2,
    'learning_rate': 0.15,
    'max_depth': 5,
    'min_child_weight': 3,
    'n_estimators': 60,
    'subsample': 0.7},
   {'colsample_bytree': 0.7,
    'gamma': 0.2,
    'learning_rate': 0.125,
    'max_depth': 7,
    'min_child_weight': 5,
    'n_estimators': 60,
    'subsample': 0.8},
   {'colsample_bytree': 0.7,
    'gamma': 0.0,
    'learning_rate': 0.1,
    'max_depth': 9,
    'min_child_weight': 3,
    'n_estimators': 60,
    'subsample': 0.9},
   {'colsample_bytree': 0.7,
    'gamma': 0.4,
    'learning_rate': 0.2,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 60,
    'subsample': 0.8},
   {'colsample_bytree': 0.9,
    'gamma': 0.0,
    'learning_rate': 0.125,
    'max_depth': 9,
    'min_child_weight': 1,
    'n_estimators': 60,
    'subsample': 0.7}],
  'rank_test_score': array([5, 1, 3, 4, 2], dtype=int32),
  'split0_test_score': array([0.99958227, 0.99949354, 0.99939512, 0.9993089 , 0.99942503]),
  'split1_test_score': array([0.99999274, 1.        , 1.        , 0.99996136, 1.        ]),
  'split2_test_score': array([0.98740142, 0.99997809, 0.99997734, 0.99643454, 0.99998174]),
  'split3_test_score': array([0.99991968, 0.99993518, 0.99993145, 0.99966687, 0.9999445 ]),
  'split4_test_score': array([0.99998584, 0.99999019, 0.99999503, 0.99973587, 0.99999629]),
  'std_fit_time': array([ 0.20135244,  0.89121741,  0.72970204,  0.2287905 , 14.90251455]),
  'std_score_time': array([0.0130524 , 0.01287   , 0.02972883, 0.0140801 , 0.04357486]),
  'std_test_score': array([0.00498977, 0.0001942 , 0.00023359, 0.00131035, 0.0002231 ])},
 {'colsample_bytree': 0.7,
  'gamma': 0.2,
  'learning_rate': 0.125,
  'max_depth': 7,
  'min_child_weight': 5,
  'n_estimators': 60,
  'subsample': 0.8},
 0.9998793992377142)
Please note that the hyperparameters found above using RandomizedSearchCV and the hyperparameters used below in creating the final model might be different, the reason being, I have executed the RandomizedSearchCV multiple times to find which set of hyperparameters gives the optimum result and finally used the one below which gave me the best performance.


[ ]
# Creating XGBoost model with selected hyperparameters
from xgboost import XGBClassifier

clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.2,
              learning_rate=0.125, max_delta_step=0, max_depth=7,
              min_child_weight=5, missing=None, n_estimators=60, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=42,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.8, verbosity=1)

# fit on the dataset
clf.fit(X_over, y_over ) 
XGB_test_score = clf.score(X_test, y_test)
print('Model Accuracy: {0}'.format(XGB_test_score))

# Probabilities for each class
XGB_probs = clf.predict_proba(X_test)[:, 1]

# Calculate roc auc
XGB_roc_value = roc_auc_score(y_test, XGB_probs)

print("XGboost roc_value: {0}" .format(XGB_roc_value))
fpr, tpr, thresholds = metrics.roc_curve(y_test, XGB_probs)
threshold = thresholds[np.argmax(tpr-fpr)]
print("XGBoost threshold: {0}".format(threshold))
Model Accuracy: 0.9993328768806727
XGboost roc_value: 0.9815403079438694
XGBoost threshold: 0.01721232570707798
Print the important features of the best model to understand the dataset

[ ]
imp_var = []
for i in clf.feature_importances_:
    imp_var.append(i)
print('Top var =', imp_var.index(np.sort(clf.feature_importances_)[-1])+1)
print('2nd Top var =', imp_var.index(np.sort(clf.feature_importances_)[-2])+1)
print('3rd Top var =', imp_var.index(np.sort(clf.feature_importances_)[-3])+1)
Top var = 14
2nd Top var = 17
3rd Top var = 10

[ ]
# Calculate roc auc
XGB_roc_value = roc_auc_score(y_test, XGB_probs)

print("XGboost roc_value: {0}" .format(XGB_roc_value))
fpr, tpr, thresholds = metrics.roc_curve(y_test, XGB_probs)
threshold = thresholds[np.argmax(tpr-fpr)]
print("XGBoost threshold: {0}".format(threshold))
XGboost roc_value: 0.9815403079438694
XGBoost threshold: 0.01721232570707798

# Conclusion
In the oversample cases, of all the models we build found that the XGBOOST model with Random Oversampling with StratifiedKFold CV gave us the best accuracy and ROC on oversampled data. Post that we performed hyperparameter tuning and got the below metrices :

XGboost roc_value: 0.9815403079438694 XGBoost threshold: 0.01721232570707798

However, of all the models we created we found Logistic Regression with L2 Regularisation for StratifiedKFold cross validation (without any oversampling or undersampling) gave us the best result.

Colab paid products - Cancel contracts here
Connected to Python 3 Google Compute Engine backend


