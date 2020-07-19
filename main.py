#%%
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import FeatureHasher
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#%% Gerekli sınıflar
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#%% Preprocess vs. için gerekli fonksiyonlar
def save_df_to_file(data,file):
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'value'])
        for (n, m), val in np.ndenumerate(data):
            writer.writerow([n, m, val])

def label_encoding(x_train):
    train = pd.DataFrame()
    label = LabelEncoder()
    for c in x_train.columns:
        if (x_train[c].dtype == 'object'):
            train[c] = label.fit_transform(x_train[c])
        else:
            train[c] = x_train[c]
    return train

def remove_correlated_variables(train,treshold):
    threshold = 0.9
    corr_matrix = train.corr().abs()
    #corr_matrix.head()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #upper.head()
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    train = train.drop(columns=to_drop)
    return  train

def one_hot_encoding(x_train):
    one = OneHotEncoder()
    one.fit(x_train)
    train = one.transform(x_train)
    print('----train data set has got {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    return train

def feature_hasher(x_train):
    X_train_hash = x_train.copy()
    for c in x_train.columns:
        X_train_hash[c] = x_train[c].astype('str')
    hashing = FeatureHasher(input_type='string')
    train = hashing.transform(X_train_hash.values)
    return train


#%% model fonksiyonları
def modelAcc(clf,X,y,fold):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=fold)
    clf.fit(X_train,y_train)
    y_pre=clf.predict(X_test)
    acc = accuracy_score(y_test,y_pre)
    print(f'Accuracy : {acc}',)
    return acc

def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pre=lr.predict(X_test)
    print('Accuracy : ',accuracy_score(y_test,y_pre))




#%%

x_train = pd.read_csv("piu_train.csv")
y_train = pd.read_csv("piu_train_label.csv")
x_test = pd.read_csv("piu_test.csv")

print('x_train data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1]))
print('y_train data set has got {} rows and {} columns'.format(y_train.shape[0],y_train.shape[1]))




#Handling Missing Values
x_train = DataFrameImputer().fit_transform(x_train)
y_train = DataFrameImputer().fit_transform(y_train)
x_test = DataFrameImputer().fit_transform(x_test)

#Handling Encodingx
x_train = one_hot_encoding(x_train)
x_test = one_hot_encoding(x_test)
#print(x_train)
#print(y_train)
#print(x_test)

#%%
######################################################################################
#x_test.to_csv("Edited_test_set_values.csv")
#save_df_to_file(x_train,"Edited_trainning_set_values.csv")
#x_train.to_csv("Edited_trainning_set_values.csv")
#y_train.to_csv("Edited_test_trainning_set_labels.csv")

mapping = {'functional': 1, 'non functional': 0, 'functional needs repair':2}
y_train = y_train.replace({'status_group': mapping})

#Feature Scaling of datasets
st_x= StandardScaler(with_mean=False)
x_train= st_x.fit_transform(x_train)
print(' X train data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1]))
X_train = st_x.transform(x_train)
#X_test = st_x.transform(x_test)
#y_train= st_x.fit_transform(y_train)
print('y_train data set has got {} rows and {} columns'.format(y_train.shape[0],y_train.shape[1]))

#%%

LR = LogisticRegression()
SVML = svm.SVC(kernel='linear')
RF = RandomForestClassifier(max_depth=20, random_state=0)
fold = 0.2
modelAcc(RF,x_train,y_train['status_group'],fold)
#logistic(x_train,y_train['status_group'])




# %%
import torch

# %%
