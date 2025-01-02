from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback

# If a model can't predict class 4, stop training
class Class4EarlyStopping(Callback):
    def __init__(self, validation_data):
        super(Class4EarlyStopping, self).__init__()
        self.validation_data_x, self.validation_data_y = validation_data
        self.class_4_predicted = False
    
    def on_epoch_begin(self, epoch, logs=None):
        # print(self.validation_data_x.shape)
        # print(self.validation_data_y.shape)
        # print(np.argmax(self.validation_data_y, axis=1))
        # print(self.validation_data)
        pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:  # Start checking predictions from the second epoch onwards
            y_pred = np.argmax(self.model.predict(self.validation_data_x), axis=1)
            y_true = np.argmax(self.validation_data_y, axis=1)

            cm = confusion_matrix(y_true, y_pred)

            # Check if class 4 has been predicted
            if cm[4, 4] > 0:
                self.class_4_predicted = True

            if not self.class_4_predicted:
                print(f"Epoch {epoch}: Class 4 not predicted yet. Skipping model.")
                self.model.stop_training = True