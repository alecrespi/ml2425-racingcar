import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

class AdaBoostCNN:
    def __init__(self, model_dir, n_estimators=10):
        """
        Initializes the AdaBoostCNN utility.

        Parameters:
        - model_dir (str): Directory where AdaBoost models and weights will be stored/loaded.
        - n_estimators (int): Number of CNN models (weak learners) to train in the AdaBoost ensemble.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.n_estimators = n_estimators
        self.num_classes = -1
        self.input_shape = None
        self.models = []
        self.alphas = []
        self._builder_ready = False

    def set_builder(self, build_model, input_shape, num_classes):
        """
        Sets the model builder function for the AdaBoost ensemble.

        Parameters:
        - model (function): Function that returns a compiled Keras model.
        - input_shape (tuple): Shape of the input data (e.g., (96, 96, 3)).
        - num_classes (int): Number of output classes.
        """
        self._build_model = build_model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._builder_ready = False

    def predict(self, test_dir):
        """
        Makes predictions on the dataset located in `test_dir`.
        """
        datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_data = datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Ensemble Prediction
        final_predictions = np.zeros((test_data.samples, self.num_classes))
        for model, alpha in zip(self.models, self.alphas):
            test_data.reset()
            predictions = model.predict(test_data)
            final_predictions += alpha * predictions

        # Final predictions (ensemble voting)
        final_classes = np.argmax(final_predictions, axis=1)
        return final_classes, test_data.classes

    def evaluate(self, test_data, test_labels):
        """
        Evaluates the AdaBoost ensemble on test data.

        Parameters:
        - test_data (np.ndarray): Test data, shape should match model input.
        - test_labels (np.ndarray): True labels, one-hot encoded.

        Returns:
        - float: Accuracy score of the ensemble.
        """
        self.ready()
        predictions = self.predict(test_data)
        true_labels = np.argmax(test_labels, axis=1)
        return accuracy_score(true_labels, predictions)

    def save_model(self, models, alphas):
        """
        Saves the AdaBoost models and alpha values to the specified directory.

        Parameters:
        - models (list): List of trained Keras models.
        - alphas (list): List of alpha weights corresponding to the models.
        """
        self.ready()
        for idx, model in enumerate(models):
            model_path = os.path.join(self.model_dir, f"model_{idx}.h5")
            model.save(model_path)
            print(f"Model {idx} saved at {model_path}")

        alphas_path = os.path.join(self.model_dir, "alphas.npy")
        np.save(alphas_path, np.array(alphas))
        print(f"Alpha values saved at {alphas_path}")

    def load_model(self, overwrite=False):
        """
        Loads the AdaBoost models and alpha values from the specified directory.

        Returns:
        - list: Loaded models.
        - list: Loaded alpha values.
        """
        if self.models or self.alphas:
            print("Warning: Models and alphas already loaded. Overwriting...")
            if not overwrite:
                raise ValueError("Models and alphas already loaded. Set overwrite=True to reload.")

        self.models = []
        idx = 0

        while True:
            model_path = os.path.join(self.model_dir, f"model_{idx}.h5")
            if not os.path.exists(model_path):
                break
            self.models.append(tf.keras.models.load_model(model_path))
            print(f"Model {idx} loaded from {model_path}")
            idx += 1

        alphas_path = os.path.join(self.model_dir, "alphas.npy")
        if not os.path.exists(alphas_path):
            raise FileNotFoundError(f"Alphas file not found in {self.model_dir}")
        self.alphas = np.load(alphas_path).tolist()
        print(f"Alpha values loaded from {alphas_path}")

    def ready(self):
        if not self._builder_ready:
            raise ValueError("Model builder not set. Use set_builder() to set the model builder.")