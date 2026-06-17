Fraud Detection Classification Project:----
About Project:

In this project,I built a classification model to predict whether an order is returned or not.The dataset was imbalanced,so I used SMOTE to balance the classes before training the models.

Dataset:Product-Sales-Region.xlsx

Targeted Column:Returned

Work Done:
i. Loaded the dataset
ii. Removed missing values
iii. Converted date columns into numeric format
iv. Converted categorical data
v. Split data into training and testing sets
vi. Applied SMOTE for class balancing
vii. Trained Logistic Regression model
viii. Trained Random Forest model
ix. Performed Hyperparameter Tuning using GridSearchCV
x. Evaluated models using Precision, Recall and ROC-AUC

Tools Used:
i. Python
ii.Pandas
iii.Scikit-Learn
iv.Imbalanced-Learn

Conclusion:
Random Forest and Logistic Regression models were trained on the dataset.SMOTE helped handle the class imbalance.The models were evaluated using Precision,Recall and ROC-AUC metrics.