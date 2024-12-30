import tensorflow as tf
from tensorflow.python import keras
import keras.models as models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Disable GPU boosting
# os.environ['TF_METAL_ENABLE'] = '0'

# Am I using CPU or GPU?
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# env
plot_cm = True
allow_save_model = False


# Paths
train_dir = 'DATA/default/train/'
test_dir = 'DATA/default/test/'
models_dir = 'MODELS/'
model_name = 'testmodel.h5'

# Parameters
img_size = (96, 96)
input_shape = (*img_size, 3)
batch_size = 32
epochs = 1

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = test_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = train_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for consistent evaluation
)

# Load model from file
model = models.load_model(models_dir + model_name)

# Evaluate Model
print("Evaluating Model...")
loss, accuracy = model.evaluate(test_data)
print(f"Test Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

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


model_path = models_dir + model_name
if allow_save_model and input(f"Are you sure you want to create/overwrite {model_path}? (y/n): ").strip().lower() == 'y':
    model.save(model_path)
    print(f"Model saved as {model_path}.")
else:
    print("Model saving skipped.")