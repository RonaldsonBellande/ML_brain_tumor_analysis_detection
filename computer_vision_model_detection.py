from header_imports import *


class computer_vision_localization_detection(utils.Dataset):
    def __init__(self, category, currently_build_model = "normal_category_1_model1_computer_vision_categories_43_model.h5"):
        
        self.boxes = None
        self.class_ids = list()
        self.number_images_to_plot = 20
        self.image_path = "traffic_signs" + "/Test"
        self.category = category
        
        self.path_to_model = "models/"
        self.save_model_path = self.path_to_model + "/transfer_learning_model/"
        self.save_model_path = os.path.join(self.save_model_path, "logs")
        self.currently_build_model = self.path_to_model + currently_build_model

        self.prediction_config = detection()
        self.model = modellib.MaskRCNN(mode='inference', model_dir=self.save_model_path, config=self.prediction_config)
        self.model.load_weights(self.currently_build_model, by_name=True) 

        self.category_names = traffic_sign_categories.category_names
        self.categories = traffic_sign_categories.categories

        self.name_directory = {self.category_names[0]: 1, self.category_names[1]: 2, self.category_names[2]: 3, self.category_names[3]: 4, self.category_names[4]: 5, self.category_names[5]: 6, self.category_names[6]: 7, self.category_names[7]: 8, self.category_names[8]: 9, self.category_names[9]: 10,
                self.category_names[10]: 11, self.category_names[11]: 12, self.category_names[12]: 13, self.category_names[13]: 14, self.category_names[14]: 15, self.category_names[15]: 16, self.category_names[16]: 17, self.category_names[17]: 18, self.category_names[18]: 19, self.category_names[19]: 20,
                self.category_names[20]: 21, self.category_names[21]: 22, self.category_names[22]: 23, self.category_names[23]: 24, self.category_names[24]: 25, self.category_names[25]: 26, self.category_names[26]: 27, self.category_names[27]: 28, self.category_names[28]: 29, self.category_names[29]: 30,
                self.category_names[30]: 31, self.category_names[31]: 32, self.category_names[32]: 33, self.category_names[33]: 34, self.category_names[34]: 35, self.category_names[35]: 36, self.category_names[36]: 37, self.category_names[37]: 38, self.category_names[38]: 39, self.category_names[39]: 40,
                self.category_names[40]: 41, self.category_names[41]: 42, self.category_names[42]: 43}

        for i in range(0, 43):
            self.add_class(detection_name, i, self.category_names[i])

        if self.category == "normal":
            self.category_names = self.categories
        elif self.category == "regular":
            self.category_names = self.category_names 

        self.localization_detection()


    def localization_detection(self):

        count = os.listdir(self.image_path)
        for i in range(0,len(count)):
            path = os.path.join(self.image_path, count[i])
            
            if os.path.isfile(path):
                file_names = next(os.walk(self.image_path))[2]
                image = skimage.io.imread(os.path.join(self.image_path, count[i]))
            
                detect = model.detect([image], verbose=1)[0]
                visualize.display_instances(count[i],image, detect['rois'], detect['masks'], detect['class_ids'], self.category_names, detect['scores'])

    

    def load_custom(self, dataset_dir, subset):

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations1 = json.load(open('D:/MaskRCNN-aar/Dataset/train/demo_json.json'))
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
           
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            name_dict = {"laptop": 1,"tab": 2,"phone": 3}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("object", image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons, num_ids=num_ids)


    def load_mask(self, image_id):
       
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
