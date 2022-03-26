from header_imports import *

class model_building(models):
    def __init__(self, number_classes, model_type, image_type):

        """
        False - 0
        glioma_tumor or False - 1
        meningioma_tumor - 2
        pituitary_tumor - 3
        """

        self.image_file = []
        self.label_name = []
        self.number_classes = int(number_classes)
        self.image_size = 240
        self.number_of_nodes = 16
        self.true_path  = "brain_cancer_category_2/"
        self.image_type = image_type

        self.valid_images = [".jpg",".png"]
        self.categories = ["False","True"]
        self.advanced_categories = ["False", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        self.model = None
        self.model_summary = "model_summary/"
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model_type = model_type
        
        self.labelencoder = LabelEncoder()
        self.setup_structure() 
        self.splitting_data_normalize()
        
        if self.model_type == "model1":
            self.model = self.create_models_1()
        elif self.model_type == "model2":
            self.model = self.create_models_2()
        elif self.model_type == "model3":
            self.model = self.create_model_3()

        self.save_model_summary()

    
    def setup_structure(self):
        
        if self.image_type == "normal":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2/"
        elif self.image_type == "edge_1":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2_edge_1/"
        elif self.image_type == "edge_2":
            self.true_path = self.true_path + "brain_cancer_seperate_category_2_edge_2/"

        if self.number_classes == 2:

            self.category_names = self.categories 
            
            self.check_valid(self.categories[0])
            self.check_valid(self.categories[1])
            
            self.resize_image_and_label_image(self.categories[0])
            self.resize_image_and_label_image(self.categories[1])

        elif self.number_classes == 4:
            
            self.category_names = self.advanced_categories 
            self.true_path = "brain_cancer_category_4/"
            if self.image_type == "normal":
            	self.true_path = self.true_path + "brain_cancer_seperate_category_4/"
            elif self.image_type == "edge_1":
                self.true_path = self.true_path + "brain_cancer_seperate_category_4_edge_1/"
            elif self.image_type == "edge_2":
                self.true_path = self.true_path + "brain_cancer_seperate_category_4_edge_2/"
             
            self.check_valid(self.advanced_categories[0])
            self.check_valid(self.advanced_categories[1])
            self.check_valid(self.advanced_categories[2])
            self.check_valid(self.advanced_categories[3])
            
            self.resize_image_and_label_image(self.advanced_categories[0])
            self.resize_image_and_label_image(self.advanced_categories[1])
            self.resize_image_and_label_image(self.advanced_categories[2])
            self.resize_image_and_label_image(self.advanced_categories[3])

        else:
            print("Detection Variety out of bounds")
        
        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.image_file = np.array(self.image_file)
        self.label_name = np.array(self.label_name)
        self.label_name = self.label_name.reshape((len(self.image_file),1))

    
    def check_valid(self, input_file):
        for img in os.listdir(self.true_path + input_file):
            ext = os.path.splitext(img)[1]
            if ext.lower() not in self.valid_images:
                continue
    

    def resize_image_and_label_image(self, input_file):
        for image in os.listdir(self.true_path + input_file):
            
            image_resized = cv2.imread(os.path.join(self.true_path + input_file,image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            self.image_file.append(image_resized)
            self.label_name.append(input_file)


    def splitting_data_normalize(self):
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.image_file, self.label_name, test_size = 0.10, random_state = 42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
        self.X_train = self.X_train.astype("float32") /255
        self.X_test = self.X_test.astype("float32") / 255

   
    
    def save_model_summary(self):
        with open(self.model_summary + self.model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
