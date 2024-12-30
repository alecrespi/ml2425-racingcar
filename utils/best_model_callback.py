import tensorflow as tf
# Define a custom callback to save the best model in a variable
class BestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, previous_best_model = None):
        super().__init__()
        self.best_model = None
        if previous_best_model:
            self.best_model = previous_best_model
            self.best_val_accuracy = previous_best_model.evaluate(self.validation_data)[1]
        else:
            self.best_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # Check if the current epoch has a better validation accuracy
        if logs['val_accuracy'] > self.best_val_accuracy:
            self.best_val_accuracy = logs['val_accuracy']
            # Save the current model as the best model
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_model.set_weights(self.model.get_weights())  # Copy weights
