from keras import layers
from keras import models


def determine_k_value():
    """Return k value."""
    #######

    #######
    return k


def build_cnn_architecture():
    """Return instance of keras.models.Sequential."""
    #######

    #######
    return model


def test_determine_k_value():
    k = determine_k_value()
    assert isinstance(k, int)
    print("\nPass.")


def test_build_cnn_architecture():
    model = build_cnn_architecture()
    assert isinstance(model, models.Sequential)
    print("\nPass.")


if __name__ == '__main__':
    test_determine_k_value()
    test_build_cnn_architecture()