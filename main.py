import sys
import time

from analyze import show_dataset_info, show_columns_info, detect_outliers
from file_utils import save_model
from graphs import show_relationships, show_distibutions, conf_matrix
from prepare_data import diabetes_prediction_dataset
from print_utils import separator
from train_model import train_model

if __name__ == '__main__':
    args = sys.argv[1:]

    use_grid_search_cv = '-use-grid-search-cv' in args
    save_models = '-save-models' in args
    use_best_params = '-use-best-params' in args

    models = ['random_forest', 'decision_tree', 'knn']
    if '-exclude-random-forest' in args:
        models.remove('random_forest')
    if '-exclude-decision-tree' in args:
        models.remove('decision_tree')
    if '-exclude-knn' in args:
        models.remove('knn')

    if '-analyze-data' in args:
        print('Analyzing Data Stage')
        start_time = time.time()
        show_dataset_info()
        show_columns_info()
        detect_outliers()
        print('The analyzing data stage has been completed in %s seconds.' % (time.time() - start_time))
        separator()

    if '-show-graphs' in args:
        print('Showing Graphs Stage')
        start_time = time.time()
        show_distibutions()
        show_relationships()
        print('The showing graph stage has been completed in %s seconds.' % (time.time() - start_time))
        separator()

    if '-train-models' in args:
        print('Training Models Stage')
        for model_name in models:
            print(f'Training {model_name} model....')
            start_time = time.time()
            model = train_model(
                diabetes_prediction_dataset,
                model_name,
                False,
                True,
                use_best_params
            )
            print('The model training stage has been completed in %s seconds.' % (time.time() - start_time))
            if save_models:
                save_model(model, model_name)
                print('The model has been saved.')
            separator()

    if '-evaluate-models' in args:
        print('Evaluating Models Stage')
        for model_name in models:
            print(f'Evaluating {model_name} model....')
            start_time = time.time()
            results = train_model(
                diabetes_prediction_dataset,
                model_name,
                use_grid_search_cv,
                False,
                False,
            )
            print(f'Evaluation results for {model_name}:')
            print(f'Accuracy: {results["accuracy"]}')
            print(f'Precision: {results["precision"]}')
            print(f'Recall: {results["recall"]}')
            print(f'F1-score: {results["f1"]}')
            print(f'Specificity: {results["specificity"]}')
            print(f'Confusion matrix: {results["confusion_matrix"]}')
            if '-dont-show-cm-graph' not in args:
                conf_matrix(results['confusion_matrix'])
            print('The model evaluation stage has been completed in %s seconds.' % (time.time() - start_time))
            separator()
