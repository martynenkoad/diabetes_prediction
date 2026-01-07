# Diabetes Prediction App

## Description

This is a study project the result of which is an app that can be used to predict the diabetes based on the personal data such as age, bmi, HbA1c level etc.
The app consists of the main part where all the logic is implemented, and the user interface part that can be used for the predictions.

## Models

The models used for these project and their names in the app are:

1. Random forest - `random_forest`
2. Decision tree - `decision_tree`
3. KNN - `knn`


## Main Part

In the main part, the ML process for the diabetes prediction is implemented.
This part also takes care of some utility tasks such as model saving etc. 
The main stages of the Machine Learning implemented within this project are listed below.

### Stages
1. **Data analysis** - The analysis of the given dataset. The following steps are present:
    - General dataset information (column types, data count etc.)
    - Null values per column
    - Duplicates analysis
    - Columns general description
    - Unique values of each column
    - Outliers analysis (IQR)
    - Columns distribution (visually displayed)
    - Relationships and correlations (visually displayed) 
2. **Data preprocessing** - The modification of the dataset so that it is analyzed better by the model. The following steps are present:
    - Duplicates removal
    - Columns cleaning
    - One-hot encoding of the categorical columns
    - Standard scaling of the numerical columns
    - Resampling using SMOTETomek algorithm
    - Train/test split
3. **Hyperparameter tuning** - GridSearchCV can be optionally used to find the best parameters for each model.
4. **Model(s) training** - The model is trained using the train set and then tested using the test set.
5. **Model(s) evaluation** - Analysis of the model performance. It involves the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Specificity
    - Confusion matrix

### Configuration

The application stages and configurations can be defined in the application args for the convenience purposes. This is the list with the available stages and configurations.

#### Available stages

1. `-analyze-data`: Will show dataset, columns information and the detected outliers.
2. `-show-graphs`: Will show graphs, such as:
   - Columns distributions for each column
   - Correlation of all the columns (heatmap)
   - Target correlation (with diabetes, heatmap)
   - Scatter plot (age vs bmi, hued by diabetes)
   - Boxplot (diabetes & HbA1c_level, diabetes & blood_glucose_level)
3. `-train-models`: Will train models skipping the evaluation stage.
4. `-evaluate-models`: Will evaluate models skipping the model saving stage.

#### Available configurations

1. `-use-grid-search-cv`: Will apply GridSearchCV to detect the best parameters for the models.
2. `-save-models`: Will save all the models as .pkl files in the local models folder.
3. `-use-best-params`: Will use the best parameters for all models.
4. `-exclude-random-forest`: Will not train random forest model.
5. `-exclude-decision-tree`: Will not train decision tree model.
6. `-exclude-knn`: Will not train knn model.
7. `-dont-show-cm-graph`: Will not show confusion matrix as graph.

#### Some possible configurations:

1. Apply all args: `-analyze-data -show-graphs -train-models -evaluate-models -use-grid-search-cv -save-models -use-best-params -exclude-random-forest -exclude-decision-tree -exclude-knn -dont-show-cm-graph`
2. Run every stage for every model with every configuration: `-analyze-data -show-graphs -train-models -evaluate-models -use-grid-search-cv -save-models`
3. Run every stage, but skip GridSearchCV without losing models quality: `-analyze-data -show-graphs -train-models -evaluate-models -save-models -use-best-params`


### Run

`python [filename] [args]`

## User Interface

The user interface part is present in the `app/app.py` file. 
It uses streamlit lib to provide the basic functionality that is needed for the diabetes prediction based on the personal data such as age, bmi etc.
The app will only work if at least 1 model is saved in the `models` folder.

### Run Script

`streamlit run app/app.py`
(make sure to be in venv)

## `models` folder

The `models` folder contains the .pkl files of the models that have been trained and can be loaded for the further testing and predictions.
It is important to have at least 1 .pkl model file in this directory to run the diabetes prediction app.

### Saving the models

To save the models, run the main application using at least these 2 args: `-train-models` and `save-models`.
The models names are generated using this template: `models/[model_name].pkl`.
To not train/save the model(s), use the `-exclude-[model_name]` arg when running the main app.

### Loading the models

When the streamlit app is run, the available models are loaded automatically and are available in the selected_model dropdown.

