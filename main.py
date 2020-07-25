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
import lightgbm as lgb
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

#%% Gerekli sınıflar
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """ bos degerleri mean ile doldur    """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#%% Preprocess vs. için gerekli fonksiyonlar

def labelEnc(x_train):
    train = pd.DataFrame()
    label = LabelEncoder()
    for c in x_train.columns:
        if (x_train[c].dtype == 'object'):
            train[c] = label.fit_transform(x_train[c])
        else:
            train[c] = x_train[c]
    return train


def oneHotEnd(x_train):
    one = OneHotEncoder()
    one.fit(x_train)
    train = one.transform(x_train)
    print('----train data set has got {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    return train


def dropId(train,id_column_name):
    return  train.drop(columns=[id_column_name])

def findCossVars(train,treshold):
    threshold = 0.9
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are %d columns to remove.' % (len(to_drop)))
    return to_drop


def findZeroImpFeatures(train, train_labels, iterations=2):
    train_labels_ravel = train_labels.values.ravel()
    feature_importances = np.zeros(train.shape[1])

    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    for i in range(iterations):
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels_ravel, test_size=0.25,
                                                                            random_state=i)

        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)],
                  eval_metric='multi_logloss', verbose=200)
        feature_importances += model.feature_importances_ / iterations

    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values(
        'importance', ascending=False)

    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))

    return zero_features, feature_importances


def takeFI(df, threshold=0.9):

    df = df.sort_values('importance', ascending=False).reset_index()

    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index , threshold))
    return df

def keepColumns(train,keepColumns):
    return train[keepColumns]

def applyFI(train,train_labels,threshold):
    second_round_zero_features, feature_importances = findZeroImpFeatures(train, train_labels)
    norm_feature_importances = takeFI(feature_importances, threshold=0.95)
    features_to_keep = list(
        norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])
    return features_to_keep

def removeColumns(train,to_drop):
    print(to_drop)
    train = train.drop(columns=to_drop)
    return train

def fitTransform(x_train, y_train, x_test):
    x1 = DataFrameImputer().fit_transform(x_train)
    y1 = DataFrameImputer().fit_transform(y_train)
    x2 = DataFrameImputer().fit_transform(x_test)
    return x1, y1, x2
    print("fit transform done")

# one hot encode train ve test beraber
def ohEncoding(x_train, x_test):
    result = oneHotEnd(pd.concat([x_train,x_test]))
    """x1 = oneHotEnd(x_train)
    x2 = oneHotEnd(x_test)"""
    x1 = result[:59400,:]
    x2 = result[59400:,:]
    print("one hot encoding done")
    return x1, x2

def lEncoding(x_train, y_train, x_test):
    x1 = labelEnc(x_train)
    y1 = labelEnc(y_train)
    x2 = labelEnc(x_test)
    print("label encoding done")
    return x1, y1, x2

#%% train acc hesaplama 5 fold ile
def modelAcc(clf,X,y,fold):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=fold)
    clf.fit(X_train,y_train)
    y_pre=clf.predict(X_test)
    acc = accuracy_score(y_test,y_pre)
    #print(f'Accuracy : {acc}',)
    return acc

# test verisinden predictionları alıp dosyaya çıkar
def extractTestResults(clf, x_train,y_train,x_test,x_test_or):
    testMapping = {1:'functional', 0:'non functional' , 2:'functional needs repair'}


    clf.fit(x_train,y_train)
    y_pre=clf.predict(x_test)

    testDf = pd.DataFrame()
    testDf['id'] = x_test_or['id']
    testDf['status_group'] = y_pre
    testDf = testDf.replace({'status_group': testMapping})
    
    
    submission = "submission.csv"
    testDf.to_csv(submission, columns=['id','status_group'],index=None,sep=',')
    print(f"test result for {str(clf)} in file {submission}")



#%%  verileri oku 
x_train = pd.read_csv("piu_train.csv")
y_train = pd.read_csv("piu_train_label.csv")
x_test = pd.read_csv("piu_test.csv")
x_test_or = pd.read_csv("piu_test.csv")
mapping = {'functional': 1, 'non functional': 0, 'functional needs repair':2}
y_train = y_train.replace({'status_group': mapping})
x_train = dropId(x_train,'id')
y_train = dropId(y_train,'id')
x_test = dropId(x_test,'id')

print('x_train: {} rows {} columns; x_test {} rows {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))

###################### encoding sec: one hot=ohEncoding; label=lEncoding ###############################
x_train, y_train, x_test = fitTransform(x_train,y_train,x_test)
#x_train, x_test = ohEncoding(x_train,x_test)
x_train, y_train, x_test = lEncoding(x_train,y_train,x_test)

###################### drop feature selection yapmak icin #######################################
"""colums_to_drop = findCossVars(x_train,0.9)
x_train = removeColumns(x_train,colums_to_drop)
x_test = removeColumns(x_test,colums_to_drop)
print('drop: x_train {} rows {} columns; x_test {} rows {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))"""

###################### keep feature selection yapmak icin #######################################
"""columns_to_keep = applyFI(x_train,y_train,threshold=0.90)
x_train = keepColumns(x_train,columns_to_keep)
x_test = keepColumns(x_test,columns_to_keep)
print(list(x_train.columns))
print('keep: x_train  {} rows and {} columns; x_test {} rows {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))
"""


######################## scale data 0 1 arasında ################################################
scaler = StandardScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
scaler.fit(x_test)
x_test = scaler.transform(x_test)


#%% classifierları test et
fold = 0.2

depths= [ 0.01, 0.1, 1, 10]
for d in depths:
    RF = LogisticRegression( C=d )
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"Log Res Acc for C {d}: {acc}")

depths = [1,3,5,8]
for d in depths:
    RF = KNeighborsClassifier(n_neighbors=d)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"knn for depth {d}: {acc}")


depths = [1, 2, 4, 8]
for d in depths:
    RF = RandomForestClassifier( min_samples_leaf=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"Random Forest Acc for mss {d}: {acc}")


depths = [20, 40, 100, 150]
for d in depths:
    RF = RandomForestClassifier( n_estimators=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"Random Forest Acc for tree {d}: {acc}")

depths = [ 10, 20, 30, 40]
for d in depths:
    RF = RandomForestClassifier(max_depth=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"Random Forest Acc for depth {d}: {acc}")

depths = [1, 2, 4, 8]
for d in depths:
    RF = tree.DecisionTreeClassifier( min_samples_leaf=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"DecisionTree Acc for mss {d}: {acc}")

depths = [ 10, 20, 30, 40]
for d in depths:
    RF = tree.DecisionTreeClassifier(max_depth=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"DecisionTree Acc for depth {d}: {acc}")


depths = [100, 200, 300, 400]
for d in depths:
    RF = MLPClassifier(hidden_layer_sizes=(d,), random_state=1)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"MLPClassifier hidden layer size {d}: {acc}")


depths = [ 100, 200, 300, 400]
for d in depths:
    RF = MLPClassifier( random_state=1,  max_iter=d )
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"MLPClassifier max iter {d}: {acc}")


#%% secilen classifier ile test verisinden sonuçları al
clf = RandomForestClassifier(min_samples_leaf=1, n_estimators=150,max_depth=20, random_state=0)
extractTestResults(clf, x_train, y_train['status_group'], x_test, x_test_or)



