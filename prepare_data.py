import pandas as pd

def alter_smoking_column(smoking_status):
    """
    Alters the smoking_history columns so that the meaning of its values is more logical
    :param smoking_status: Current value of smoking_history
    :return: The changed value of the column per row
    """
    if smoking_status in ['never', 'No Info']:
        return 'not_smoker'
    elif smoking_status == 'current':
        return 'smoker'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

# Read the original data from the .csv file
original_diabetes_prediction_dataset = pd.read_csv('data/diabetes_prediction_dataset.csv')
# Remove the duplicates from the dataset
diabetes_prediction_dataset = original_diabetes_prediction_dataset.drop_duplicates()
# Apply the modifications to the smoking_history column
diabetes_prediction_dataset['smoking_history'] = (diabetes_prediction_dataset['smoking_history']
                                                  .apply(alter_smoking_column))
