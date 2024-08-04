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
1. Importing dependencies
2. Exploratory data analysis
3. Splitting the data into train & test data
4. Model Building
    --Perform cross validation with RepeatedKFold
    --Perform cross validation with StratifiedKFold
    --RandomOverSampler with StratifiedKFold Cross Validation
    --Oversampling with SMOTE Oversampling
    --Oversampling with ADASYN Oversampling
5. Hyperparameter Tuning
6. Conclusion
</p>
