from keras import layers
from keras import models
from keras.applications import VGG16
import json
import os
import shutil


def predict(test_dir,
            output_dir="q5_result"):
    """
    Args:
        - test_dir: test set directory, note that there are four sub-directories under this directory, i.e., c0, c1, c2, c3.
        - output_dir: output directory
    Important: Your model shall be stored in ``q5_model`` directory. Hence, in this function, you have to implement:
        1. Restore your model from ``q5_model``; and
        2. Make predictions for test set using the restored model.
            Your results will be stored in file named ``prediction.json`` under ``q5_result``.
            Make sure your results are formatted as dictionary, e.g., {"image_file_name": "label"}. See prediction.json for reference.
            You can save the results with json.dump.
        3. You can define your model architecture in this scripts.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists("q5_model"):
        raise Exception("Model not found.")

    ### Restore your model

    ### Make predictions with restored model

    ### Convert predicted results to dictionary

    ### Saved resutls to q5_result/prediction.json
    return True


def test_predict():
    shutil.rmtree("q5_result")
    predict(test_dir="/path/to/testset")
    result_path = "q5_result/prediction.json"
    if not os.path.isfile(result_path):
        raise FileNotFoundError()

    with open(result_path, encoding="utf-8") as fp:
        results = json.load(fp)

    assert isinstance(results, dict)

    print("\nPass.")


if __name__ == '__main__':
    test_predict()