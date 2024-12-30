import tensorflow as tf
from tensorflow.python import keras
import keras.models as models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AvgPool2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

from utils.best_model_callback import BestModelCallback

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Disable GPU boosting
os.environ['TF_METAL_ENABLE'] = '0'

# Am I using CPU or GPU?
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# env
plot_loss = False
plot_cm = True
allow_save_model = True
enable_best_model_search = True


# Paths
train_dir = 'DATA/default/train/'
test_dir = 'DATA/default/test/'
models_dir = 'MODELS/'
model_name = 'testmodel.h5'
best_model_name = 'best_' + model_name

# Parameters
img_size = (96, 96)
input_shape = (*img_size, 3)
batch_size = 32
epochs = 5

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for consistent evaluation
)

l2_lambda = 1e-3
activation = 'relu'


model = Sequential([
    Conv2D(32, (3, 3), activation=activation, input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (5, 5), activation=activation, kernel_regularizer=l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation=activation, kernel_regularizer=l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(512, (3, 3), activation=activation, kernel_regularizer=l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    
    Flatten(),
    Dropout(0.1),
    Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda)),
    Dropout(0.1),
    Dense(256, activation="tanh", kernel_regularizer=l2(l2_lambda)),
    Dropout(0.1),
    Dense(5, activation='softmax')  # 5 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

bmcallback = BestModelCallback()

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data,
    callbacks=[bmcallback]
)

best_model = bmcallback.best_model

if(plot_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Predict on Test Data
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

# Confusion Matrix
if(plot_cm):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

if enable_best_model_search:
    y_pred = np.argmax(best_model.predict(test_data), axis=1)
    y_true = test_data.classes

    # Confusion Matrix
    if(plot_cm):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices.keys())
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Best Model')
        plt.show()

if enable_best_model_search:

    # Classification Report
    print("Classification Report for BEST MODEL:")
    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))


model_path = models_dir + model_name
if allow_save_model and input(f"Are you sure you want to create/overwrite {model_path}? (y/n): ").strip().lower() == 'y':
    model.save(model_path)
    print(f"Model saved as {model_path}.")
else:
    print("Model saving skipped.")

best_model_path = models_dir + best_model_name
if enable_best_model_search and allow_save_model and input(f"Are you sure you want to create/overwrite {best_model_path}? (y/n): ").strip().lower() == 'y':
    best_model.save(best_model_path)
    print(f"Model saved as {best_model_path}.")
else:
    print("Model saving skipped.")