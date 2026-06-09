import pytest

from utils import *

def test_get_classifier_random_forest():
    classifier = get_classifier("random_forest")
    assert isinstance(classifier, RandomForestClassifier)

def test_get_classifier_decision_tree():
    classifier = get_classifier("decision_tree")
    assert isinstance(classifier, DecisionTreeClassifier)

def test_get_classifier_knn():
    classifier = get_classifier("knn")
    assert isinstance(classifier, KNeighborsClassifier)

def test_random_forest_best_params():
    classifier = get_classifier("random_forest")

    assert classifier.n_estimators == 200
    assert classifier.min_samples_leaf == 2


def test_knn_best_params():
    classifier = get_classifier("knn")

    assert classifier.n_neighbors == 3

def test_knn_without_best_params():
    classifier = get_classifier("knn", use_best_params=False)

    assert isinstance(classifier, KNeighborsClassifier)
    assert classifier.n_neighbors == 5  # sklearn default

def test_get_classifier_invalid_name():
    with pytest.raises(ValueError):
        get_classifier("banana")

from utils import get_params


def test_get_params_random_forest():
    params = get_params("random_forest")

    assert "model__n_estimators" in params
    assert "model__max_depth" in params


def test_get_params_knn():
    params = get_params("knn")

    assert params["model__n_neighbors"] == [3, 5, 9]

def test_get_params_invalid_name():
    with pytest.raises(ValueError):
        get_params("raspberry")
