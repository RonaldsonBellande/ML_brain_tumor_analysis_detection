from header_inputs import *


class brain_tumor_display(object):
    def __init__(self):
        

    # PLotting model
    def plot_model(self, brain_cancer_model):
        plt.plot(brain_cancer_model.history['accuracy'])
        plt.plot(brain_cancer_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()
        plt.plot(brain_cancer_model.history['loss'])
        plt.plot(brain_cancer_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()

