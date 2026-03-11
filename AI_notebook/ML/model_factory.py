import numpy as np
import os 
import os
import time
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit 
from joblib import dump, load

class SVM:
    def __init__(self, log_path, model_name, tuned_parameters=None, random_state=1234):
        self.tuned_parameters = [{'kernel': ['rbf'], 
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100], 'class_weight':['balanced']},
                        {'kernel': ['sigmoid'], 
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100], 'class_weight':['balanced']}]

        self.random_state = random_state
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.model_path = log_path+model_name+'_model.gz'
        
    def evaluation_metrics(self, y_true, y_pred):
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print("Verifying recall {} & precision {} & f1-score {}".format(recall_weighted, precision_weighted, f1_weighted))
        return recall_weighted, precision_weighted, f1_weighted
    
    def fit(self, X_train, y_train, X_val, y_val):
        print("Dimension of training set is: {} and label is: {}".format(X_train.shape, y_train.shape))
        print("Dimension of validation set is: {} and label is: {}".format(X_val.shape, y_val.shape))
        print("[SVM] Starting hyperparameter search...")
        t0 = time.perf_counter()
        
        X_all = np.concatenate((X_train, X_val),axis=0)
        y_all = np.concatenate((y_train, y_val),axis=0)
        
        # Create a list where train data indices are -1 and validation data indices are 0
        tr_index = np.full((X_train.shape[0]), -1)
        val_index = np.full((X_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()
        
        # Use the list to create PredefinedSplit
        pds = PredefinedSplit(test_fold = split_index)
        clf = GridSearchCV(estimator=SVC(), param_grid=self.tuned_parameters, cv=pds, n_jobs=-1, scoring='accuracy', verbose=2)
        clf.fit(X_all , y_all)
        print("[SVM] Grid search complete in {:.2f}s".format(time.perf_counter() - t0))
        
        #Clasifying with an optimal parameter set
        Optimal_params = clf.best_params_
        print("[SVM] Best params:", Optimal_params)
        classifier = SVC(**Optimal_params)
        classifier.fit(X_train, y_train)
        dump(classifier, self.model_path)
        print("[SVM] Saved model to {}".format(self.model_path))
        

    def predict(self, X_test, y_test):
        print("Dimesion of testing set is: {} and label is: {}".format(X_test.shape, y_test.shape))
        print("Type of classes: ", np.unique(y_test))
        classifier = load(self.model_path)
        classifier_acc = classifier.score(X_test, y_test)
        y_true, y_pred = y_test, classifier.predict(X_test)
        print(classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy from SVM's evaluation: {:04f} and from sklean metric: {:04f}".format(classifier_acc, accuracy))
        evaluation = {'accuracy': classifier_acc, 
                      'recall':recall_weighted, 
                      'precision': precision_weighted,
                      'f1-score-weighted': f1_weighted}
        Y = {'y_true': y_true, 'y_pred': y_pred}
        return Y, evaluation

class KNN:
    def __init__(self, log_path, model_name, tuned_parameters=None, random_state=1234):
        self.tuned_parameters = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27,28,29,30],
                                    'weights': ['uniform', 'distance'],
                                    'metric': ['euclidean', 'manhattan', 'minkowski']}
        
        self.random_state = random_state
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.model_path = log_path+model_name+'_model.gz'
        
    def evaluation_metrics(self, y_true, y_pred):
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print("Verifying recall {} & precision {} & f1-score {}".format(recall_weighted, precision_weighted, f1_weighted))
        return recall_weighted, precision_weighted, f1_weighted
    
    def fit(self, X_train, y_train, X_val, y_val):
        print("Dimension of training set is: {} and label is: {}".format(X_train.shape, y_train.shape))
        print("Dimension of validation set is: {} and label is: {}".format(X_val.shape, y_val.shape))
        print("[KNN] Starting hyperparameter search...")
        t0 = time.perf_counter()
        
        X_all = np.concatenate((X_train, X_val),axis=0)
        y_all = np.concatenate((y_train, y_val),axis=0)
        
        # Create a list where train data indices are -1 and validation data indices are 0
        tr_index = np.full((X_train.shape[0]), -1)
        val_index = np.full((X_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()
        
        # Use the list to create PredefinedSplit
        pds = PredefinedSplit(test_fold = split_index)
        clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=self.tuned_parameters, cv=pds, n_jobs=-1,
                   scoring='accuracy', verbose=2)
        clf.fit(X_all , y_all)
        print("[KNN] Grid search complete in {:.2f}s".format(time.perf_counter() - t0))
        
        #Clasifying with an optimal parameter set
        Optimal_params = clf.best_params_
        print("[KNN] Best params:", Optimal_params)
        classifier = KNeighborsClassifier(**Optimal_params)
        classifier.fit(X_train, y_train)
        dump(classifier, self.model_path)
        print("[KNN] Saved model to {}".format(self.model_path))

    def predict(self, X_test, y_test):
        print("Dimesion of testing set is: {} and label is: {}".format(X_test.shape, y_test.shape))
        print("Type of classes: ", np.unique(y_test))
        classifier = load(self.model_path)
        classifier_acc = classifier.score(X_test, y_test)
        y_true, y_pred = y_test, classifier.predict(X_test)
        print(classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy from KNN's evaluation: {:04f} and from sklean metric: {:04f}".format(classifier_acc, accuracy))
        evaluation = {'accuracy': classifier_acc, 
                      'recall':recall_weighted, 
                      'precision': precision_weighted,
                      'f1-score-weighted': f1_weighted}
        Y = {'y_true': y_true, 'y_pred': y_pred}
        return Y, evaluation

class RandomForest:
    def __init__(self, log_path, model_name, tuned_parameters=None, random_state=1234):
        self.tuned_parameters = {'bootstrap': [True, False],
             'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             'min_samples_leaf': [2, 5, 8, 11, 14],
             'min_samples_split': [2, 5, 8, 11, 14],
             'class_weight':['balanced'],
             'n_estimators': [100, 300, 500]}
        
        self.random_state = random_state
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.model_path = log_path+model_name+'_model.gz'
        
    def evaluation_metrics(self, y_true, y_pred):
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        print("Verifying recall {} & precision {} & f1-score {}".format(recall_weighted, precision_weighted, f1_weighted))
        return recall_weighted, precision_weighted, f1_weighted
    
    def fit(self, X_train, y_train, X_val, y_val):
        print("Dimension of training set is: {} and label is: {}".format(X_train.shape, y_train.shape))
        print("Dimension of validation set is: {} and label is: {}".format(X_val.shape, y_val.shape))
        print("[RF] Starting hyperparameter search...")
        t0 = time.perf_counter()
        
        X_all = np.concatenate((X_train, X_val),axis=0)
        y_all = np.concatenate((y_train, y_val),axis=0)
        
        # Create a list where train data indices are -1 and validation data indices are 0
        tr_index = np.full((X_train.shape[0]), -1)
        val_index = np.full((X_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()
        
        # Use the list to create PredefinedSplit
        pds = PredefinedSplit(test_fold = split_index)
        clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=self.tuned_parameters, cv=pds, n_jobs=-1,
                   scoring='accuracy', verbose=2)
        clf.fit(X_all , y_all)
        print("[RF] Grid search complete in {:.2f}s".format(time.perf_counter() - t0))
        
        #Clasifying with an optimal parameter set
        Optimal_params = clf.best_params_
        print("[RF] Best params:", Optimal_params)
        classifier = RandomForestClassifier(**Optimal_params)
        classifier.fit(X_train, y_train)
        dump(classifier, self.model_path)
        print("[RF] Saved model to {}".format(self.model_path))
    
    def predict(self, X_test, y_test):
        print("Dimesion of testing set is: {} and label is: {}".format(X_test.shape, y_test.shape))
        print("Type of classes: ", np.unique(y_test))
        classifier = load(self.model_path)
        classifier_acc = classifier.score(X_test, y_test)
        y_true, y_pred = y_test, classifier.predict(X_test)
        print(classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy from RandomForest's evaluation: {:04f} and from sklean metric: {:04f}".format(classifier_acc, accuracy))
        evaluation = {'accuracy': classifier_acc, 
                      'recall':recall_weighted, 
                      'precision': precision_weighted,
                      'f1-score-weighted': f1_weighted}
        Y = {'y_true': y_true, 'y_pred': y_pred}
        return Y, evaluation