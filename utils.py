from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def get_classifier(model_name, use_best_params=True):
    """
    Returns the classifier instance based on the model_name
    :param model_name: Name of the model to use (can be 'random_forest', 'knn', 'decision_tree')
    :param use_best_params: Whether to use the best params for the model. Default is True
    :return: The classifier instance
    """
    if model_name == 'random_forest':
        if use_best_params:
            return RandomForestClassifier(
                max_depth=None,
                min_samples_leaf=2,
                min_samples_split=2,
                n_estimators=200,
            )
        else:
            return RandomForestClassifier()
    elif model_name == 'decision_tree':
        if use_best_params:
            return DecisionTreeClassifier(
                criterion='gini',
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=1,
            )
        else:
            return DecisionTreeClassifier()
    elif model_name == 'knn':
        if use_best_params:
            return KNeighborsClassifier(n_neighbors=3)
        else:
            return KNeighborsClassifier()
    else:
        raise ValueError('Invalid model name')

def get_params(model_name):
    """
    Returns the params grid per model that can be used for GridSearchCV
    :param model_name: Name of the model to use (can be 'random_forest', 'knn', 'decision_tree')
    :return: Params grid for the model
    """
    if model_name == 'random_forest':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [10, None, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [2, 4, 8],
        }
    elif model_name == 'decision_tree':
        return {
            'model__max_depth': [10, None, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 3, 5, 7],
            'model__criterion': ['gini', 'entropy'],
        }
    elif model_name == 'knn':
        return {
            'model__n_neighbors': [3, 5, 9],
        }
    else:
        raise ValueError('Invalid model name')

def use_grid_search_cv(model_name, pipeline, train_set, results_set, scoring='accuracy'):
    """
    Searches the best training params for the model and returns the best model
    :param model_name: Name of the model to use (can be 'random_forest', 'knn', 'decision_tree')
    :param pipeline: IMB Pipeline instance with the classifier and other steps in it
    :param train_set: X_train set with the features
    :param results_set: y_train set with the target values
    :param scoring: Scoring to use for the GridSearchCV. Default is 'accuracy'
    :return: Best model with the best parameters
    """
    params = get_params(model_name)
    grid = GridSearchCV(pipeline, param_grid=params, scoring=scoring, n_jobs=-1, cv=5)
    grid.fit(train_set, results_set)

    print(f"Best parameters for {model_name}:")
    print(grid.best_params_)

    return grid.best_estimator_