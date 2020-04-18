from keras import layers
from keras import models
from keras import optimizers


def multiple_choice():
    """Choose the right answers (one or more) from multiple choices.
    Note: if you think a is the right answer, return a;
            if your think a, b are the right answers, return a, b; ect.
    """
    # Here are 4 choices
    a = "Training accuracy increases over time and this is a problem."
    b = "Validation accuracy does not increase over time and this is a problem."
    c = "Training loss decreases over time and this is a problem."
    d = "Validation loss does not decrease over time and this is a problem."
    return a, b, c, d


def modified_cnn():
    """Return instance of keras.models.Sequential
    This is similar to build_cnn_architecture() in q1.py; however, you have to compile your model with
    a proper loss function and optimizer.
    """
    #######

    #######
    return model


def test_multiple_choice():
    choice = multiple_choice()
    assert isinstance(choice, str) or isinstance(choice, tuple)
    print("\nPass.")


def test_modified_cnn():
    model = modified_cnn()
    assert isinstance(model, models.Sequential)
    if hasattr(model, "loss") and hasattr(model, "optimizer"):
        print("\nPass.")


if __name__ == '__main__':
    test_multiple_choice()
    test_modified_cnn()