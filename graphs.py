from matplotlib import pyplot as plt
import seaborn as sns
from prepare_data import original_diabetes_prediction_dataset

def conf_matrix(cm):
    """
    Displays the confusion matrix as heatmap
    :param cm: Confusion matrix
    :return: Nothing
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def distribution(column_name, _bins):
    """
    Displays the distribution of the values of the column as a histogram
    :param column_name: Name of the column
    :param _bins: How many bins to use for the histogram
    :return: Nothing
    """
    bins = _bins
    if column_name == 'bmi':
        bins = 40
    plt.hist(original_diabetes_prediction_dataset[column_name], color='skyblue', edgecolor='black', bins=bins)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f"Distribution of {column_name}")
    plt.show()

def pie(column_name='diabetes', label1='No Diabetes', label2='Diabetes', title='Diabetes Distribution'):
    """
    Displays the distribution of the values of the column as a pie chart
    :param column_name: Name of the column
    :param label1: Label 1, will be used for labels
    :param label2: Label 2, will be used for labels
    :param title: Title of the plot
    :return: Nothing
    """
    sizes = original_diabetes_prediction_dataset[column_name].value_counts()
    labels = [label1, label2]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
    plt.title(title)
    plt.show()

def correlation_of_columns():
    """
    Displays the correlation of all the columns with each other as a heatmap
    :return: Nothing
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(original_diabetes_prediction_dataset.corr(numeric_only=True), annot=True, fmt='.2f')
    plt.show()

def target_correlation(target):
    """
    Displays the correlation of all the columns with the target column as a heatmap
    :param target: Name of the column that is the target
    :return: Nothing
    """
    target_corr = original_diabetes_prediction_dataset.corr(numeric_only=True)[target].drop(target)
    plt.figure(figsize=(10, 6))
    sns.heatmap(target_corr.to_frame(), annot=True, fmt='.2f')
    plt.title(f'Correlation with {target} column')
    plt.show()

def scatterplot(x, y, hue=None):
    """
    Displays the scatter plot of data points of the provided x and y variables
    :param x: X variable
    :param y: Y variable
    :param hue: By which parameter to highlight data points
    :return: Nothing
    """
    sns.scatterplot(x=x, y=y, hue=hue, data=original_diabetes_prediction_dataset)
    plt.title(f'{x} vs {y}')
    plt.show()

def boxplot(x, y):
    """
    Shows the boxplot of x and y
    :param x: X variable
    :param y: Y variable
    :return: Nothing
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=original_diabetes_prediction_dataset, x=x, y=y)
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.show()

def show_distibutions():
    """
    Shows the distributions of all the columns
    :return: Nothing
    """
    for column in original_diabetes_prediction_dataset.columns:
        if column != 'diabetes':
            distribution(column, len(original_diabetes_prediction_dataset[column].unique().tolist()))
        else:
            pie(column)

def show_relationships():
    """
    Shows all the graphs that describe the relationships in the dataset
    :return: Nothing
    """
    correlation_of_columns()
    target_correlation('diabetes')
    scatterplot('age', 'bmi', 'diabetes')
    boxplot('diabetes', 'HbA1c_level')
    boxplot('diabetes', 'blood_glucose_level')