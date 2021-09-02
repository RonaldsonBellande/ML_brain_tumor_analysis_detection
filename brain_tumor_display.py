from header_inputs import *


class brain_tumor_display(object):
    def __init__(self, brain_cancer_model):
        
        self.brain_cancer_model = brain_cancer_model

        self.plot_model()
        

    # PLotting model
    def plot_model(self):

        # Brain cancer modeling
        plt.plot(self.brain_cancer_model.history['accuracy'])
        plt.plot(self.brain_cancer_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()


        plt.plot(self.brain_cancer_model.history['loss'])
        plt.plot(self.brain_cancer_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()


    def plot_random_examples():

        plt.figure( dpi=256)
        predicted_classes = model.predict_classes(X_test)

        for i in range(10):
            plt.subplot(5,5,i+1)
            fig=plt.imshow(X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(model_labels[predicted_classes[i]] ) + "\n Actual - {}".format(model_labels[Y_test_vec[i,0]] ),fontsize=3)
            plt.tight_layout()

