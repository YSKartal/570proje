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

def sort_wrt_id_drop_id(train,id_column_name):
    train.sort_values(id_column_name)
    return  train.drop(columns=[id_column_name])

def collect_correlated_variables(train,treshold):
    threshold = 0.9
    corr_matrix = train.corr().abs()
    #corr_matrix.head()
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #upper.head()
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print('There are %d columns to remove.' % (len(to_drop)))
    return to_drop


def identify_zero_importance_features(train, train_labels, iterations=2):
    train_labels_ravel = train_labels.values.ravel()
    """
    Identify zero importance features in a training dataset based on the
    feature importances from a gradient boosting model.

    Parameters
    --------
    train : dataframe
        Training features

    train_labels : np.array
        Labels for training data

    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """

    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):
        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels_ravel, test_size=0.25,
                                                                            random_state=i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)],
                  eval_metric='multi_logloss', verbose=200)
    #eval function internette auc idi hata verdi, multi_logloss yaptim, arity de oluyormus
        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations

    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values(
        'importance', ascending=False)

    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))

    return zero_features, feature_importances


def take_feature_importances(df, threshold=0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.

    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances

    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column

    """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index , threshold))
    return df

def keep_columns(train,keep_columns):
    return train[keep_columns]

def apply_feature_importance(train,train_labels,threshold):
    second_round_zero_features, feature_importances = identify_zero_importance_features(train, train_labels)
    norm_feature_importances = take_feature_importances(feature_importances, threshold=0.95)
    features_to_keep = list(
        norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])
    return features_to_keep

def remove_columns(train,to_drop):
    train = train.drop(columns=to_drop)
    return train

def fitTransform(x_train, y_train, x_test):
    x1 = DataFrameImputer().fit_transform(x_train)
    y1 = DataFrameImputer().fit_transform(y_train)
    x2 = DataFrameImputer().fit_transform(x_test)
    return x1, y1, x2
    print("fit transform done")

def ohEncoding(x_train, x_test):
    x1 = one_hot_encoding(x_train)
    x2 = one_hot_encoding(x_test)
    print("one hot encoding done")
    return x1, x2

def lEncoding(x_train, y_train, x_test):
    x1 = label_encoding(x_train)
    y1 = label_encoding(y_train)
    x2 = label_encoding(x_test)
    print("label encoding done")
    return x1, y1, x2

#%% model fonksiyonları
def modelAcc(clf,X,y,fold):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=fold)
    clf.fit(X_train,y_train)
    y_pre=clf.predict(X_test)
    acc = accuracy_score(y_test,y_pre)
    #print(f'Accuracy : {acc}',)
    return acc


def extractTestResults(clf, x_train,y_train,x_test):
    testMapping = {1:'functional', 0:'non functional' , 2:'functional needs repair'}

    clf.fit(x_train,y_train)
    y_pre=clf.predict(x_test)

    testDf = pd.DataFrame()
    testDf['id'] = x_test['id']
    testDf['status_group'] = y_pre
    testDf = testDf.replace({'status_group': testMapping})
    
    testDf.to_csv(str(clf)+'.csv', columns=['id','status_group'],index=None,sep=',')
    print(f"test result in file {str(clf)+'.csv'}")


#%%  verileri oku
x_train = pd.read_csv("piu_train.csv")
y_train = pd.read_csv("piu_train_label.csv")
x_test = pd.read_csv("piu_test.csv")
mapping = {'functional': 1, 'non functional': 0, 'functional needs repair':2}
y_train = y_train.replace({'status_group': mapping})

print('initial: x_train data set has got {} rows and {} columns and x_test data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))

x_train, y_train, x_test = fitTransform(x_train,y_train,x_test)
#x_train, x_test = ohEncoding(x_train,x_test)
x_train, y_train, x_test = lEncoding(x_train,y_train,x_test)


colums_to_drop = collect_correlated_variables(x_train,0.9)
x_train = remove_columns(x_train,colums_to_drop)
x_test = remove_columns(x_test,colums_to_drop)
print('column drop: x_train data set has got {} rows and {} columns and x_test data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))

"""columns_to_keep = apply_feature_importance(x_train,y_train,threshold=0.95)
x_train = keep_columns(x_train,columns_to_keep)
x_test = keep_columns(x_test,columns_to_keep)
"""
print('x_train data set has got {} rows and {} columns and x_test data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1],x_test.shape[0],x_test.shape[1]))

#%% classifierları test et
fold = 0.2

LR = LogisticRegression()
acc= modelAcc(LR,x_train, y_train['status_group'],fold)
print(f"Log Res Acc: {acc}")

#SVML = svm.SVC(kernel='linear')

depths = [2, 5, 10, 20, 30, 40]
for d in depths:
    RF = RandomForestClassifier(max_depth=d, random_state=0)
    acc= modelAcc(RF,x_train, y_train['status_group'],fold)
    print(f"Random Forest Acc for depth {d}: {acc}")


#%% secilen classifier ile test verisinden sonuçları al

clf = RandomForestClassifier(max_depth=40, random_state=0)
extractTestResults(clf, x_train, y_train['status_group'], x_test)











#%%
#Feature Scaling of datasets
"""st_x= StandardScaler(with_mean=False)
x_train= st_x.fit_transform(x_train)
print(' X train data set has got {} rows and {} columns'.format(x_train.shape[0],x_train.shape[1]))
X_train = st_x.transform(x_train)
#X_test = st_x.transform(x_test)
#y_train= st_x.fit_transform(y_train)
print('y_train data set has got {} rows and {} columns'.format(y_train.shape[0],y_train.shape[1]))
"""

"""
def testClassifier(clf, X, y, fold):
    LR = LogisticRegression()
    acc= modelAcc(LR,X,y,fold)
    print(f"Log Res Acc: {acc}")

    #SVML = svm.SVC(kernel='linear')

    depths = [2, 5, 10, 20, 30]
    for d in depths:
        RF = RandomForestClassifier(max_depth=d, random_state=0)
        acc= modelAcc(RF,X,y,fold)
        print(f"Random Forest Acc for depth {d}: {acc}")
"""
#%%






