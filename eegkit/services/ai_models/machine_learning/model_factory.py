"""Classic ML model wrappers with validation-driven hyperparameter tuning."""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


LOGGER = logging.getLogger(__name__)


class SVM:
    """Train and evaluate an SVM classifier with grid search."""

    def __init__(self, model_name: str, random_state: int = 1234):
        """Initialize SVM search parameters and model identity."""
        self.tuned_parameters = [
            {
                "kernel": ["rbf"],
                "gamma": [1e-2, 1e-3],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "class_weight": ["balanced"],
            },
            {
                "kernel": ["sigmoid"],
                "gamma": [1e-2, 1e-3],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "class_weight": ["balanced"],
            },
        ]
        self.random_state = random_state
        self.model_name = model_name
        self.classifier = None

    def evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Compute recall, precision, and f1-score in weighted mode."""
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        LOGGER.info(
            "Verifying recall %s & precision %s & f1-score %s",
            recall_weighted,
            precision_weighted,
            f1_weighted,
        )
        return recall_weighted, precision_weighted, f1_weighted

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, object]:
        """Run grid search on train+val and fit best SVM on train split."""
        LOGGER.info("Dimension of training set is: %s and label is: %s", x_train.shape, y_train.shape)
        LOGGER.info("Dimension of validation set is: %s and label is: %s", x_val.shape, y_val.shape)
        LOGGER.info("[SVM] Starting hyperparameter search...")
        t0 = time.perf_counter()

        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)

        tr_index = np.full((x_train.shape[0]), -1)
        val_index = np.full((x_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()

        pds = PredefinedSplit(test_fold=split_index)
        clf = GridSearchCV(
            estimator=SVC(),
            param_grid=self.tuned_parameters,
            cv=pds,
            n_jobs=-1,
            scoring="accuracy",
            verbose=0,
        )
        clf.fit(x_all, y_all)
        LOGGER.info("[SVM] Grid search complete in %.2fs", time.perf_counter() - t0)

        best_params = clf.best_params_
        LOGGER.info("[SVM] Best params: %s", best_params)
        classifier = SVC(**best_params)
        classifier.fit(x_train, y_train)
        self.classifier = classifier
        return {"best_params": best_params, "model": classifier, "model_name": self.model_name}

    def predict(self, x_test: np.ndarray, y_test: np.ndarray, classifier=None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Predict on test split and return labels with evaluation metrics."""
        LOGGER.info("Dimension of testing set is: %s and label is: %s", x_test.shape, y_test.shape)
        LOGGER.info("Type of classes: %s", np.unique(y_test))
        clf = classifier or self.classifier
        if clf is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        classifier_acc = clf.score(x_test, y_test)
        y_true, y_pred = y_test, clf.predict(x_test)
        LOGGER.info("\n%s", classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        LOGGER.info(
            "Accuracy from SVM evaluation: %.4f and from sklearn metric: %.4f",
            classifier_acc,
            accuracy,
        )
        evaluation = {
            "accuracy": classifier_acc,
            "recall": recall_weighted,
            "precision": precision_weighted,
            "f1-score-weighted": f1_weighted,
        }
        labels = {"y_true": y_true, "y_pred": y_pred}
        return labels, evaluation


class KNN:
    """Train and evaluate a KNN classifier with grid search."""

    def __init__(self, model_name: str, random_state: int = 1234):
        """Initialize KNN search parameters and model identity."""
        self.tuned_parameters = {
            "n_neighbors": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                            19,20,21,22,23,24,25,26,27,28,29,30],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }
        self.random_state = random_state
        self.model_name = model_name
        self.classifier = None

    def evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Compute recall, precision, and f1-score in weighted mode."""
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        LOGGER.info(
            "Verifying recall %s & precision %s & f1-score %s",
            recall_weighted,
            precision_weighted,
            f1_weighted,
        )
        return recall_weighted, precision_weighted, f1_weighted

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, object]:
        """Run grid search on train+val and fit best KNN on train split."""
        LOGGER.info("Dimension of training set is: %s and label is: %s", x_train.shape, y_train.shape)
        LOGGER.info("Dimension of validation set is: %s and label is: %s", x_val.shape, y_val.shape)
        LOGGER.info("[KNN] Starting hyperparameter search...")
        t0 = time.perf_counter()

        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)

        tr_index = np.full((x_train.shape[0]), -1)
        val_index = np.full((x_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()

        pds = PredefinedSplit(test_fold=split_index)
        clf = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=self.tuned_parameters,
            cv=pds,
            n_jobs=-1,
            scoring="accuracy",
            verbose=0,
        )
        clf.fit(x_all, y_all)
        LOGGER.info("[KNN] Grid search complete in %.2fs", time.perf_counter() - t0)

        best_params = clf.best_params_
        LOGGER.info("[KNN] Best params: %s", best_params)
        classifier = KNeighborsClassifier(**best_params)
        classifier.fit(x_train, y_train)
        self.classifier = classifier
        return {"best_params": best_params, "model": classifier, "model_name": self.model_name}

    def predict(self, x_test: np.ndarray, y_test: np.ndarray, classifier=None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Predict on test split and return labels with evaluation metrics."""
        LOGGER.info("Dimension of testing set is: %s and label is: %s", x_test.shape, y_test.shape)
        LOGGER.info("Type of classes: %s", np.unique(y_test))
        clf = classifier or self.classifier
        if clf is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        classifier_acc = clf.score(x_test, y_test)
        y_true, y_pred = y_test, clf.predict(x_test)
        LOGGER.info("\n%s", classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        LOGGER.info(
            "Accuracy from KNN evaluation: %.4f and from sklearn metric: %.4f",
            classifier_acc,
            accuracy,
        )
        evaluation = {
            "accuracy": classifier_acc,
            "recall": recall_weighted,
            "precision": precision_weighted,
            "f1-score-weighted": f1_weighted,
        }
        labels = {"y_true": y_true, "y_pred": y_pred}
        return labels, evaluation


class RandomForest:
    """Train and evaluate a Random Forest classifier with grid search."""

    def __init__(self, model_name: str, random_state: int = 1234):
        """Initialize Random Forest search parameters and model identity."""
        self.tuned_parameters = {
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "min_samples_leaf": [2, 5, 8, 11, 14],
            "min_samples_split": [2, 5, 8, 11, 14],
            "class_weight": ["balanced"],
            "n_estimators": [100, 300, 500],
        }
        self.random_state = random_state
        self.model_name = model_name
        self.classifier = None

    def evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Compute recall, precision, and f1-score in weighted mode."""
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        LOGGER.info(
            "Verifying recall %s & precision %s & f1-score %s",
            recall_weighted,
            precision_weighted,
            f1_weighted,
        )
        return recall_weighted, precision_weighted, f1_weighted

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, object]:
        """Run grid search on train+val and fit best Random Forest on train split."""
        LOGGER.info("Dimension of training set is: %s and label is: %s", x_train.shape, y_train.shape)
        LOGGER.info("Dimension of validation set is: %s and label is: %s", x_val.shape, y_val.shape)
        LOGGER.info("[RF] Starting hyperparameter search...")
        t0 = time.perf_counter()

        x_all = np.concatenate((x_train, x_val), axis=0)
        y_all = np.concatenate((y_train, y_val), axis=0)

        tr_index = np.full((x_train.shape[0]), -1)
        val_index = np.full((x_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()

        pds = PredefinedSplit(test_fold=split_index)
        clf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=self.tuned_parameters,
            cv=pds,
            n_jobs=-1,
            scoring="accuracy",
            verbose=0,
        )
        clf.fit(x_all, y_all)
        LOGGER.info("[RF] Grid search complete in %.2fs", time.perf_counter() - t0)

        best_params = clf.best_params_
        LOGGER.info("[RF] Best params: %s", best_params)
        classifier = RandomForestClassifier(**best_params)
        classifier.fit(x_train, y_train)
        self.classifier = classifier
        return {"best_params": best_params, "model": classifier, "model_name": self.model_name}

    def predict(self, x_test: np.ndarray, y_test: np.ndarray, classifier=None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Predict on test split and return labels with evaluation metrics."""
        LOGGER.info("Dimension of testing set is: %s and label is: %s", x_test.shape, y_test.shape)
        LOGGER.info("Type of classes: %s", np.unique(y_test))
        clf = classifier or self.classifier
        if clf is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")
        classifier_acc = clf.score(x_test, y_test)
        y_true, y_pred = y_test, clf.predict(x_test)
        LOGGER.info("\n%s", classification_report(y_true, y_pred))
        recall_weighted, precision_weighted, f1_weighted = self.evaluation_metrics(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        LOGGER.info(
            "Accuracy from RandomForest evaluation: %.4f and from sklearn metric: %.4f",
            classifier_acc,
            accuracy,
        )
        evaluation = {
            "accuracy": classifier_acc,
            "recall": recall_weighted,
            "precision": precision_weighted,
            "f1-score-weighted": f1_weighted,
        }
        labels = {"y_true": y_true, "y_pred": y_pred}
        return labels, evaluation


def build_estimator(name: str, model_name: str, random_state: int):
    """Build and return a classic estimator instance by name."""
    key = str(name).lower()
    if key == "svm":
        return SVM(model_name=model_name, random_state=random_state)
    if key == "knn":
        return KNN(model_name=model_name, random_state=random_state)
    if key in {"random_forest", "rf"}:
        return RandomForest(model_name=model_name, random_state=random_state)
    raise ValueError(f"Unsupported estimator: {name}")
