from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
import json
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator


def best_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        layer.trainable = set_trainable
    
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
    return model

def train_and_save_model():
    aug_train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    aug_train_generator = aug_train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    model = best_model()
    history = model.fit_generator(
        aug_train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=val_generator,
        validation_steps=50)

    model.save('/userhome/34/ljiang/deep_learning/A2/q5_model/q5_model.h5')


def predict(test_dir, output_dir="q5_result"):
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
    model = models.load_model('./q5_model/q5_model.h5')

    ### Make predictions with restored model
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False)
    predict = model.predict_classes(test_generator, batch_size=None)

    ### Convert predicted results to dictionary
    mapping = {
        0: 'cat',
        1: 'dog',
        2: 'car',
        3: 'motorbike'
    }
    result = {}
    for i in range(len(test_generator.filenames)):
        y = test_generator.filenames[i].split('/')[-1]
        y_predict = predict[i]
        result[y] = mapping[y_predict]

    ### Saved resutls to q5_result/prediction.json
    with open(os.path.join(output_dir, 'prediction.json'), 'w') as f:
        json.dump(result, f, indent=2)
        
    return True


def test_predict():
    shutil.rmtree("q5_result")
    predict(test_dir='/userhome/34/ljiang/deep_learning/A2/Datasets/cat_dog_car_bike/test/')
    result_path = "q5_result/prediction.json"
    if not os.path.isfile(result_path):
        raise FileNotFoundError()

    with open(result_path, encoding="utf-8") as fp:
        results = json.load(fp)

    assert isinstance(results, dict)

    print("\nPass.")


if __name__ == '__main__':
    test_predict()