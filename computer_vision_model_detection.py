from header_imports import *


class computer_vision_localization_detection(object):
    def __init__(self, save_model, number_classes):
        
        self.image_file = []
        self.save_model = save_model
        self.model = keras.models.load_model("models/" + self.save_model)
        self.image_path = "brain_cancer_category_2/" + "Testing2/" 

        self.image_size = 240
        self.number_classes = int(number_classes)
        self.number_images_to_plot = 900
        self.split_size = 900

        self.graph_path = "graph_charts/" + "detection_localization/"

        if self.number_classes == 2:
            self.number_images_to_plot = 9
            self.model_categpory = ["False","True"]
            self.image_path = "brain_cancer_category_2/" + "Testing2/" 
       
        elif self.number_classes == 4:
            self.number_images_to_plot = 16
            self.model_categpory = ["False", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
            self.image_path = "brain_cancer_category_4/" + "Testing2/" 

        self.prepare_image_data()
        self.plot_prediction_with_model()

    
    def prepare_image_data(self):
        
        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            
            self.image_file.append(image_resized)
        
        print(self.image_file)
        self.image_file = np.array(self.image_file)
        self.X_test = self.image_file.astype("float32") / 255
    

    def split_images(self, image):

        for r in range(0,img.shape[0],30):
            for c in range(0,img.shape[1],30):
                cv2.imwrite(f"img{r}_{c}.png",img[r:r+30, c:c+30,:])


    def plot_prediction_with_model(self):

        plt.figure(dpi=1000)
        predicted_classes = self.model.predict(self.X_test)

        for i in range(self.number_images_to_plot):
            plt.subplot(math.sqrt(self.number_images_to_plot),math.sqrt(self.number_images_to_plot),i+1)
            fig=plt.imshow(self.X_test[i,:,:,:])
            plt.axis('off')
            plt.title("Predicted - {}".format(self.model_categpory[np.argmax(predicted_classes[i], axis=0)]), fontsize=1)
            plt.tight_layout()
            plt.savefig(self.graph_path + "model_detection_localization_with_model_trained_prediction_" + str(self.save_model) + '.png')

