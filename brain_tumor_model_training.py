from header_inputs import *

class brain_tumor_training(object):
    def __init__(self):


    # Compile information
    def 



    #  Training model 
    def train_model(self):
       
        self.model = KerasClassifier(build_fn = ConvNeuralNetwork, verbose = 0)
        grid = GridSearchCV(estimator = self.model, param_grid = self.param_grid, n_jobs = 1, cv = 3, verbose = 10)

        self.brain_cancer_model = model.fit(self.X_train, self.Y_train,
          batch_size=self.batch_size[2],
          validation_split=0.15,
          epochs=self.epochs[2],
          callbacks=[self.callbacks],
          shuffle=True)
   

    # Evaluate model
    def evaluate_model(self):
        evaluation = model.evaluate(X_test, Y_test, verbose=1)
        print("Loss:", evaluation[0])
        print("Accuracy: ", evaluation[1])


