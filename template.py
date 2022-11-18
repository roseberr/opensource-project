#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_dataset(dataset_path):
	#To-Do: Implement this function
 	return pd.read_csv(dataset_path)
	

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
    df=dataset_df
    n_feats=df.shape[1]-1
    #n_feats=2
    n_class0=df[df['target']==0]['target'].count()
    n_class1=df[df['target']==1]['target'].count()
    return n_feats,n_class0,n_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
    x=dataset_df.iloc[:,:-1].values
    y=dataset_df.iloc[:,-1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =testset_size,stratify=y)

    return x_train,x_test,y_train,y_test



def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    dec_cls = DecisionTreeClassifier(random_state=0)
    dec_cls.fit(x_train, y_train)
    
    #y_pred =dec_cls.predict(x_train)
    #print('Accuracy for train set for logistic regression={}'.format(sklearn.metrics.accuracy_score(y_train,y_pred)))

    y_pred = dec_cls.predict(x_test)
    acc=sklearn.metrics.accuracy_score(y_test,y_pred)
    prec=sklearn.metrics.precision_score(y_test,y_pred,zero_division='warn',average='binary')
    recall=sklearn.metrics.recall_score(y_test,y_pred,average='binary',zero_division='warn')
    #print(confusion_matrix(y_test,y_pred))
    return acc,prec,recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    rf_cls= RandomForestClassifier(random_state=0)
    rf_cls.fit(x_train,y_train)
    
    #y_pred = rf_cls.predict(x_train)

    y_pred = rf_cls.predict(x_test)
    #print("y_pred",y_pred)
    acc=sklearn.metrics.accuracy_score(y_test,y_pred)
    prec=sklearn.metrics.precision_score(y_test,y_pred,zero_division='warn',average='binary')
    recall=sklearn.metrics.recall_score(y_test,y_pred,average='binary',zero_division='warn')
    #print(confusion_matrix(y_test,y_pred))
    return acc,prec,recall

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
    #svm_cls = SVC()
    #scaler = StandardScaler()
        
    #x_train = scaler.fit_transform(x_train)#스케일링 standardscaler 적용 평균이 0과 표준편차가 1이되도록 변환
    #x_test = scaler.transform(x_test)
    #svm_cls.fit(x_train, y_train)
    
    #y_pred = svm_cls.predict(x_test)

    pipe_SVC = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
    pipe_SVC.fit(x_train, y_train)
    y_pred = pipe_SVC.predict(x_test)

    acc=sklearn.metrics.accuracy_score(y_test,y_pred)
    prec=sklearn.metrics.precision_score(y_test,y_pred,zero_division='warn')
    recall=sklearn.metrics.recall_score(y_test,y_pred,zero_division='warn')
    #print(confusion_matrix(y_test,y_pred))


    return acc,prec,recall

def print_performances(acc, prec, recall):
    #Do not modify this function!
    print ("Accuracy: ", acc)
    print ("Precision: ", prec)
    print ("Recall: ", recall)

if __name__ == '__main__':
        
    #Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print ("Number of features: ", n_feats)
    print ("Number of class 0 data entries: ", n_class0)
    print ("Number of class 1 data entries: ", n_class1)

    print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print ("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print ("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print ("\nSVM Performances")
    print_performances(acc, prec, recall)