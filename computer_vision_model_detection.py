from header_imports import *


class computer_vision_localization_detection(object):
    def __init__(self, save_model, number_classes):
        
        self.image_file = []
        self.predicted_classes_array = []
        self.save_model = save_model
        self.model = keras.models.load_model("models/" + self.save_model)
        self.image_path = "brain_cancer_category_2/" + "Testing2/" 

        self.image_size = 240
        self.number_classes = int(number_classes)
        self.split_size = 900
        self.color = [(255,0,0),(255,120,0),(255,0,120),(255,120.120)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
  
        self.thickness = 1
        self.thickness_fill = -1
        self.graph_path = "graph_charts/" + "detection_localization/" 
        self.graph_path_localization = "graph_charts/" + "detection_localization/" + "localization/"
        self.graph_path_detection = "graph_charts/" + "detection_localization/" + "detection/"
        
        if self.number_classes == 2:
            self.model_categpory = ["False","True"]
            self.image_path = "brain_cancer_category_2/" + "Testing2/" 
       
        elif self.number_classes == 4:
            self.model_categpory = ["False", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
            self.image_path = "brain_cancer_category_4/" + "Testing2/" 

        self.prepare_image_data()
        self.plot_prediction_with_model()
        # self.segmentation()
        self.localization()

    
    def prepare_image_data(self):
        
        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.split_images(image_resized)
        
        self.image_file = np.array(self.image_file)
        self.X_test = self.image_file.astype("float32") / 255
    

    def split_images(self, image):

        for r in range(0,image.shape[0],int(math.sqrt(self.split_size))):
            for c in range(0,image.shape[1],int(math.sqrt(self.split_size))):
                image_split = image[r:r+int(math.sqrt(self.split_size)), c:c+int(math.sqrt(self.split_size)),:]
                image_split = cv2.resize(image_split,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
                self.image_file.append(image_split)


    def plot_prediction_with_model(self):

        predicted_classes = self.model.predict(self.X_test)

        for i in range(len(self.image_file)):
            if self.number_classes == 2:
                self.predicted_classes_array.append([np.argmax(predicted_classes[i])][0])
            elif self.number_classes == 4:
                self.predicted_classes_array.append([np.argmax(predicted_classes[i])][0])

        self.predicted_classes_array = np.reshape(self.predicted_classes_array, ((int(math.sqrt(len(self.image_file)))), (int(math.sqrt(len(self.image_file))))))
        print(self.predicted_classes_array)
    

    def prepare_prediction(self):
        
        self.image_file_image = []
        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file_image.append(image_resized)
        
        self.image_file_image = np.array(self.image_file_image)
        self.X_test_image = self.image_file_image.astype("float32") / 255
        self.predicted_classes = self.model.predict(self.X_test)


    def segmentation(self):

        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        
        self.prepare_prediction()
        
        for i in range(len(self.image_file_image)):

            if self.number_classes == 2:
                for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                    for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                        start_point = (int(r), int(c))
                        end_point = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                        if self.predicted_classes_array[int(r/(self.image_size/(math.sqrt(len(self.image_file)))))][int(c/(self.image_size/(math.sqrt(len(self.image_file)))))] == [np.argmax(self.predicted_classes[i])][0]:
                            cv2.rectangle(image_resized, start_point, end_point, self.color[np.argmax(self.predicted_classes[i])][0], self.thickness)
        
            if self.number_classes == 4:
                for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                    for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                        start_point = (int(r), int(c))
                        end_point = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                        if self.predicted_classes_array[int(r/(self.image_size/(math.sqrt(len(self.image_file)))))][int(c/(self.image_size/(math.sqrt(len(self.image_file)))))] == [np.argmax(self.predicted_classes[i])][0]:
                            cv2.rectangle(image_resized, start_point, end_point, self.color[np.argmax(self.predicted_classes[i])][0], self.thickness)


        cv2.imwrite(self.graph_path_detection + "model_segmenation_with_model_trained_prediction_" + str(self.save_model) + '.png', image_resized)


    def localization(self):
        
        first_predicting_position = None
        first_prediction = False
        last_predicting_position = None

        for image in os.listdir(self.image_path):
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        
        self.prepare_prediction()
        
        for i in range(len(self.image_file_image)):
            if self.number_classes == 2:
                for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                    for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                        if first_prediction == False:
                            if self.predicted_classes_array[int(r/(self.image_size/(math.sqrt(len(self.image_file)))))][int(c/(self.image_size/(math.sqrt(len(self.image_file)))))] == [np.argmax(self.predicted_classes[i])][0]:
                                first_predicting_position = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                                first_prediction = True
                        last_predicting_position = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                        
                        if r == int(math.sqrt(self.split_size)) and c == int(math.sqrt(self.split_size)):
                            print(first_predicting_position)
                            print(last_predicting_position)
                            cv2.rectangle(image_resized, first_predicting_position, (210,210), self.color[np.argmax(self.predicted_classes[i])][0], self.thickness)
                            cv2.putText(image, 'OpenCV', org, self.font,fontScale, self.color[np.argmax(self.predicted_classes[i])][0], thickness, cv2.LINE_AA)
        
            if self.number_classes == 4:
                for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
                    for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
                        if first_prediction == False:
                            if self.predicted_classes_array[int(r/(self.image_size/(math.sqrt(len(self.image_file)))))][int(c/(self.image_size/(math.sqrt(len(self.image_file)))))] == [np.argmax(self.predicted_classes[i])][0]:
                                first_predicting_position = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                                first_prediction = True
                        last_predicting_position = (int(r+(self.image_size/math.sqrt(len(self.image_file)))), int(c+(self.image_size/math.sqrt(len(self.image_file)))))
                        
                        if r == int(math.sqrt(self.split_size)) and c == int(math.sqrt(self.split_size)):
                            print("here")
                            cv2.rectangle(image_resized, first_predicting_position, last_predicting_position, self.color[np.argmax(self.predicted_classes[i])][0], self.thickness)

        cv2.imwrite(self.graph_path_localization + "model_segmenation_with_model_trained_prediction_" + str(self.save_model) + '.png', image_resized)



