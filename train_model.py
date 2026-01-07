from imblearn.metrics import specificity_score
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbPipeline
from utils import get_classifier, use_grid_search_cv


def train_model(df, model_name, use_grid_search=False, skip_evaluation=False, use_best_params_for_models=True):
    """
    Prepares, trains, predicts and evaluates the model
    :param df: Dataset
    :param model_name: Name of the model to use (can be 'random_forest', 'knn', 'decision_tree')
    :param use_grid_search: Whether to use the grid search or not. Default is False
    :param skip_evaluation: If True, will skip the evaluation of the model. Default is False
    :param use_best_params_for_models: Will use the best parameters for the model. Default is True
    :return: Evaluation results
    """

    # 1. Preprocessing
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['gender', 'smoking_history']),
            ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']),
        ]
    )
    # 2. Resampling
    resampler = SMOTETomek(random_state=0)
    # 3. Classifier
    classifier = get_classifier(model_name, use_best_params_for_models)

    # 4. Pipeline creation
    pipeline = imbPipeline(steps=[
        ('column_transformer', column_transformer),
        ('resampler', resampler),
        ('model', classifier),
    ])

    # 5. Split features & target
    features = df.drop('diabetes', axis=1)
    target = df['diabetes']

    # 6. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # 7. Train the model OR Use grid search CV to find the best params for the model
    if use_grid_search:
        best_model = use_grid_search_cv(model_name, pipeline, X_train, y_train)
    else:
        # Train the model
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # If the evaluation step should be skipped, return the model.
    if skip_evaluation:
        return best_model

    # 8. Predict
    y_pred = best_model.predict(X_test)

    # 9. Evaluate
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)

    return {
        'model_name': model_name,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
    }
