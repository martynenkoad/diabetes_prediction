import pickle

def save_model(model, filename="model", folder="models", mimetype="pkl"):
    """
    Saves a model to a pickle file.
    :param model: Model to save
    :param filename: Filename to save the model to. Default is 'model'
    :param folder: Folder to save the model to. Default is 'models'
    :param mimetype: Mimetype which should be applied to the model file. Default is 'pkl'
    :return: Nothing
    """
    with open(f"{folder}/{filename}.{mimetype}", 'wb') as f:
        pickle.dump(model, f)
