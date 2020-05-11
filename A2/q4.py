from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16


def transfer_learning_with_vggnet():
    """Return model built on pre-trained VGG16."""
    #######
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-3),
              metrics=['acc'])
    #######
    return model


def test_transfer_learning_with_vggnet():
    model = transfer_learning_with_vggnet()
    assert isinstance(model, models.Sequential)
    print("\nPass.")


if __name__ == '__main__':
    test_transfer_learning_with_vggnet()
