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

# Am I using CPU or GPU?
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Paths
test_dir = 'DATA/default/test/'
model_paths = ['MODELS/basemodel1/best_basemodel1.h5']  # Paths to model architectures
trainset_paths = ['DATA/default/train', 'DATA/augmented_0/train', 'DATA/augmented_1/train', 'DATA/augmented_2/train']  # Paths to different training sets
img_size = (96, 96)
input_shape = (*img_size, 3)
batch_size = 32
epochs = 5  # You can increase this for better results

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Combined Plot Storage
# fig_loss, ax_loss = plt.subplots(len(model_paths), 1, figsize=(10, 6 * len(model_paths)))
# fig_cm, ax_cm = plt.subplots(len(model_paths), len(trainset_paths), figsize=(15, 6 * len(model_paths)))

# Loop through each model architecture
for model_idx, model_path in enumerate(model_paths):
    base_model_name = os.path.basename(model_path).replace('.h5','')
    base_model = models.load_model(model_path)  # Load model architecture
    base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"\n\nProcessing Model: {model_path}")
    for train_idx, train_dir in enumerate(trainset_paths):
        trainset_name = train_dir.replace('/', '-')
        print(f"\nTraining with Dataset: {train_dir}")

        # Train Data Generator
        train_datagen = ImageDataGenerator(rescale=1.0/255)
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Copy architecture and initialize a fresh model
        model = tf.keras.models.clone_model(base_model)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=test_data,
            verbose=1
        )

        # Plot Training/Validation Loss
        # ax_loss[model_idx].plot(history.history['loss'], label=f'Training Loss ({train_dir})')
        # ax_loss[model_idx].plot(history.history['val_loss'], label=f'Validation Loss ({train_dir})')
        # ax_loss[model_idx].set_title(f"Training and Validation Loss - Model {model_idx + 1}")
        # ax_loss[model_idx].set_xlabel("Epochs")
        # ax_loss[model_idx].set_ylabel("Loss")
        # ax_loss[model_idx].legend()

        plt.plot(history.history['loss'], label=f'Training Loss ({train_dir})')
        plt.plot(history.history['val_loss'], label=f'Validation Loss ({test_dir})')
        plt.title(f"Model {base_model_name}, Trainset {trainset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'plots/test_augmentations/losses/{base_model_name}_vs_{trainset_name}.png')
        plt.clf()

        # Evaluate the model
        y_pred = np.argmax(model.predict(test_data), axis=1)
        y_true = test_data.classes

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices.keys())
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Model {base_model_name}, Trainset {trainset_name}")
        plt.savefig(f'plots/test_augmentations/CMs/{base_model_name}_vs_{trainset_name}.png')
        plt.clf()

        # Classification Report
        # Print classification report to a file
        report = classification_report(y_true, y_pred, target_names=test_data.class_indices.keys(), output_dict=True)

        # print(f"Classification Report - Model {model_idx + 1}, Trainset {train_idx + 1}")
        # print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

        with open(f'plots/test_augmentations/reports/{base_model_name}_vs_{trainset_name}.txt', 'w') as file:
            file.write(f"Classification Report - Model {model_idx + 1}, Trainset {train_idx + 1}\n")
            file.write(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

# Show plots
plt.tight_layout()
plt.show()