import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,roc_auc_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
df = pd.read_excel(r"C:\Users\manas\OneDrive\Desktop\Decodelabs project\Dataset\Product-Sales-Region.xlsx")
print("Dataset Shape:", df.shape)
print(df.head())
#missing values
df=df.dropna()
#converting date to time
date_cols=["Date","OrderDate","DeliveryDate"]
for col in date_cols:
    df[col]=pd.to_datetime(df[col])
    df[col]=df[col].astype("int64")
#targetted col
y=df["Returned"]
X=df.drop("Returned", axis=1)
#categorical data to numeric
X=pd.get_dummies(X, drop_first=True)
print("Features Shape:", X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

log_pipeline=Pipeline([("smote", SMOTE(random_state=42)),("scaler", StandardScaler()),("model", LogisticRegression(max_iter=1000))])
log_pipeline.fit(X_train, y_train)
log_pred=log_pipeline.predict(X_test)
log_prob=log_pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic regression")
print("Precision:",precision_score(y_test, log_pred))
print("Recall:",recall_score(y_test, log_pred))
print("ROC-AUC:",roc_auc_score(y_test, log_prob))

rf_pipeline=Pipeline([("smote", SMOTE(random_state=42)),("model", RandomForestClassifier(random_state=42))])
rf_pipeline.fit(X_train, y_train)

rf_pred=rf_pipeline.predict(X_test)
rf_prob=rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom forest")
print("Precision:",precision_score(y_test, rf_pred))
print("Recall:",recall_score(y_test, rf_pred))
print("ROC-AUC:",roc_auc_score(y_test, rf_prob))
#hyperparameter tuning
param_grid={"model__n_estimators": [100, 200],"model__max_depth": [5, 10, None],"model__min_samples_split":[2, 5]}
grid=GridSearchCV(rf_pipeline,param_grid,cv=3,scoring="roc_auc",n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

best_model=grid.best_estimator_
final_pred=best_model.predict(X_test)
final_prob=best_model.predict_proba(X_test)[:,1]

print("\nFinal model")
print("Precision:",precision_score(y_test,final_pred))
print("Recall:",recall_score(y_test,final_pred))
print("ROC-AUC:",roc_auc_score(y_test,final_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,final_pred))

print("\nClassification Report")
print(classification_report(y_test,final_pred))
