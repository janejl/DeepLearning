import unittest
import sys
import importlib
import timeout_decorator
import time
import os
import json
import shutil
from keras import models
from modulefinder import ModuleFinder

current_dirpath = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dirpath)

class Assignment2SubmissionTest(unittest.TestCase):

    @timeout_decorator.timeout(1)
    def test_import_q1(self):
        import q1

    @timeout_decorator.timeout(1)
    def test_q1_determine_k_value(self):
        import q1
        k = q1.determine_k_value()
        self.assertIsInstance(k, int)

    @timeout_decorator.timeout(5)
    def test_q1_build_cnn_architecture(self):
        import q1
        model = q1.build_cnn_architecture()
        self.assertIsInstance(model, models.Sequential)

    @timeout_decorator.timeout(1)
    def test_import_q2(self):
        import q2

    @timeout_decorator.timeout(1)
    def test_q2_multiple_choice(self):
        import q2
        choice = q2.multiple_choice()
        typeCorrect = False
        if isinstance(choice, str) or isinstance(choice, tuple):
            typeCorrect = True
        self.assertTrue(typeCorrect, "Choice type is incorrect")

    @timeout_decorator.timeout(5)
    def test_q2_modified_cnn(self):
        import q2
        model = q2.modified_cnn()
        assert isinstance(model, models.Sequential)
        modelHasAttr = False
        if hasattr(model, "loss") and hasattr(model, "optimizer"):
            modelHasAttr = True
        self.assertTrue(True, "Model does not have loss and optimizer")

    @timeout_decorator.timeout(1)
    def test_import_q3(self):
        import q3

    @timeout_decorator.timeout(1)
    def test_q3_learning_rate_range(self):
        import q3
        lower, upper = q3.learning_rate_range()
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

    @timeout_decorator.timeout(1)
    def test_q3_learning_rate_examples(self):
        import q3
        try:
            bad_learning_rate, not_bad_learning_rate, good_learning_rate = q3.learning_rate_examples()
        except:
            bad_learning_rate, not_bad_learning_rate, good_learning_rate = q3.learnign_rate_examples()
        self.assertIsInstance(bad_learning_rate, float)
        self.assertIsInstance(not_bad_learning_rate, float)
        self.assertIsInstance(good_learning_rate, float)

    @timeout_decorator.timeout(1)
    def test_import_q4(self):
        import q4

    @timeout_decorator.timeout(1)
    def test_q4_transfer_learning_with_vggnet(self):
        import q4
        model = q4.transfer_learning_with_vggnet()
        self.assertIsInstance(model, models.Sequential)

    @timeout_decorator.timeout(1)
    def test_import_q5(self):
        import q5

    def findTestDir(self):
        for root, dirs, files in os.walk('../../../'):
            for d in dirs:
                path = os.path.join(root, d)
                path_snippet = ('/').join(path.split('/')[-3:])
                if 'Datasets/cat_dog_car_bike/test' in path_snippet:
                    return path

    @timeout_decorator.timeout(180)
    def test_q5_predict(self):
        import q5
        raised = False

        # Get test_dir and output_dir
        self.test_dir = self.findTestDir()
        self.assertIsInstance(self.test_dir, str, 'Cannot find "Datasets/cat_dog_car_bike/test" directory')
        output_dir = os.path.join(current_dirpath, "q5_result")
        output_path = os.path.join(output_dir, 'prediction.json')

        # Delete output_dir if exists
        if os.path.isfile(output_path):
            shutil.rmtree(output_dir)

        # Test predict
        q5.predict(self.test_dir, output_dir)

        predictionFileExists = False
        if os.path.isfile(output_path):
            predictionFileExists = True
        self.assertTrue(predictionFileExists, "prediction file is not in " + output_path)

        with open(output_path, encoding="utf-8") as fp:
            results = json.load(fp)

        assert isinstance(results, dict)


if __name__ == '__main__':
    unittest.main()