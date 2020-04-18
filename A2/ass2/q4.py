from keras import layers
from keras import models
from keras.applications import VGG16


def transfer_learning_with_vggnet():
    """Return model built on pre-trained VGG16."""
    #######

    #######
    return model


def test_transfer_learning_with_vggnet():
    model = transfer_learning_with_vggnet()
    assert isinstance(model, models.Sequential)
    print("\nPass.")


if __name__ == '__main__':
    test_transfer_learning_with_vggnet()