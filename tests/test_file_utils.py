import pickle
from file_utils import *

def test_save_model(tmp_path):
    model = {"accuracy": 0.95}

    save_model(
        model=model,
        filename="test_model",
        folder=str(tmp_path),
    )

    saved_file = tmp_path / "test_model.pkl"

    assert saved_file.exists()

    with open(saved_file, "rb") as f:
        loaded_model = pickle.load(f)

    assert loaded_model == model