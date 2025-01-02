import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from utils.AdaBoostCNN import AdaBoostCNN

# Parameters
img_size = (96, 96)
input_shape = (*img_size, 3)
num_classes = 5
n_estimators = 5  # Number of CNN models to train in AdaBoost
learning_rate = 0.001
batch_size = 32
epochs_per_model = 3

model_dir = 'models/adaboost/simpleCNN'
train_dir = 'DATA/default/train/'
test_dir = 'DATA/default/test/'

# Define a simple CNN model
def cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Initialize AdaBoostCNN
adaboost = AdaBoostCNN(
    n_estimators=n_estimators
)

adaboost.set_builder(cnn)

# Train AdaBoost ensemble
adaboost.fit(train_dir)

# Evaluate the ensemble
adaboost.evaluate(test_dir)


# AdaBoost
models = []
alphas = []  # Weight of each model
for i in range(n_estimators):
    print(f"\nTraining model {i+1}/{n_estimators}...")
    
    # Create and train CNN model
    model = create_cnn()
    
    # Reweight samples for training
    train_data.reset()
    train_data_sampled = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    model.fit(train_data_sampled, epochs=epochs_per_model, steps_per_epoch=len(train_data))
    
    # Predict and calculate weighted error
    train_data.reset()
    # y_pred = np.argmax(model.predict(train_data), axis=1)
    # y_true = train_data.classes
    y_pred = np.argmax(model.predict(train_data), axis=1)
    y_true = train_data.classes
    incorrect = (y_pred != y_true).astype(np.float32)
    weighted_error = np.sum(weights * incorrect) / np.sum(weights)
    
    # Calculate alpha (model weight)
    if weighted_error == 0:
        alpha = 1
    else:
        alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
    
    # Update weights
    weights *= np.exp(alpha * incorrect)
    weights /= np.sum(weights)
    
    # Save model and alpha
    models.append(model)
    alphas.append(alpha)

# Ensemble Prediction
test_data.reset()
final_predictions = np.zeros((test_data.samples, num_classes))
for model, alpha in zip(models, alphas):
    test_data.reset()
    predictions = model.predict(test_data)
    final_predictions += alpha * predictions

# Final predictions (ensemble voting)
final_classes = np.argmax(final_predictions, axis=1)

# Evaluate
y_true = test_data.classes
accuracy = accuracy_score(y_true, final_classes)
print(f"\nFinal Ensemble Accuracy: {accuracy * 100:.2f}%")