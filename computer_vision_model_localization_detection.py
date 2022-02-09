from header_imports import *

class computer_vision_localization_detection(object):
    def __init__(self, save_model, number_classes):
        
        self.image_file = []
        self.predicted_classes_array = []
        self.save_model = save_model
        self.model = keras.models.load_model("models/" + self.save_model)
        self.image_path = "brain_cancer_category_2/" + "Testing2/"

        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.objness = None
        self.classes = None
        self.label = -1
        self.score = -1
        self.number_box = 3

        self.label_map = {
            "False": "blue",
            "True": "yellow", 
            "glioma_tumor": "red",
            "meningioma_tumor": "green",
            "pituitary_tumor": "white", 
            }

        self.class_threshold = 0.3
        self.class_threshold_max = 0.5

        self.image_size = 240
        self.number_classes = int(number_classes)
        self.split_size = 25
        self.validation_size = 10
        
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        self.box_index = []
        self.color = [(0,255,255),(0,0,255),(0,255,0),(255,0,0)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alpha = 0.4
        self.fontScale = 0.5
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
        
        self.localization()
        # self.prepare_image_data()
        # self.plot_prediction_with_model()


    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label


    def get_score(self, xmin, ymin, xmax, ymax, objness, classes):

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes

        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score
 

    def decode_netout(self, predicted_classes_network, anchors):
        print(predicted_classes_network)
        grid_h, grid_w = predicted_classes_network.shape[:2]
        predicted_classes_network = predicted_classes_network.reshape((grid_h, grid_w, self.number_box, -1))
        nb_class = predicted_classes_network.shape[-1] - 5

        boxes = []
        predicted_classes_network[..., :2]  = 1.0 / (1.0 + np.exp(-(predicted_classes_network[..., :2])))
        predicted_classes_network[..., 4:]  = 1.0 / (1.0 + np.exp(-(predicted_classes_network[...,4:])))
        predicted_classes_network[..., 5:]  = predicted_classes_network[..., 4][..., np.newaxis] * predicted_classes_network[..., 5:]
        predicted_classes_network[..., 5:] *= predicted_classes_network[..., 5:] > self.class_threshold
 
        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = predicted_classes_network[int(row)][int(col)][b][4]
                if(objectness.all() <= self.class_threshold): continue
                x, y, w, h = predicted_classes_network[int(row)][int(col)][b][:4]
                x = (col + x) / grid_w 
                y = (row + y) / grid_h 
                w = anchors[2 * b + 0] * np.exp(w) / self.image_size
                h = anchors[2 * b + 1] * np.exp(h) / self.image_size
                classes = predicted_classes_network[int(row)][col][b][5:]
                box = self.get_score(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                boxes.append(box)
        return boxes
 

    def correct_yolo_boxes(self, boxes):

        for i in range(len(boxes)):
            x_offset, x_scale = (self.image_size - self.image_size)/2.0/self.image_size, float(self.image_size)/self.image_size
            y_offset, y_scale = (self.image_size - self.image_size)/2.0/self.image_size, float(self.image_size)/self.image_size
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * self.image_size)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * self.image_size)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * self.image_size)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * self.image_size)

        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return

        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= self.class_threshold_max:
                        boxes[index_j].classes[c] = 0


    def interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3


    def bbox_iou(box1, box2):
        intersect_w = self.interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union


    # get all of the results above a threshold
    def draw_boxes(boxes):
        v_boxes, v_labels, v_scores = list(), list(), list()
        for box in boxes:
            # enumerate all possible labels
            for i in range(len(self.model_categpory)):
                # check if the threshold for this label is high enough
                if box.classes[i] > self.class_threshold_max:
                    v_boxes.append(box)
                    v_labels.append(self.model_categpory[i])
                    v_scores.append(box.classes[i]*100)


        pyplot.imshow(pyplot.imread(filename))
        ax = pyplot.gca()
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color=self.label_map[v_labels[i]])
            ax.add_patch(rect)
            label = "%s (%.3f)" % (v_labels[i], v_scores[i])
            pyplot.text(x1, y1, label, color=self.label_map[v_labels[i]])
        pyplot.show()


    def localization(self):

        for image in os.listdir(self.image_path):
            image_name = self.image_path + image
    
            image_resized = cv2.imread(os.path.join(self.image_path, image))
            image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
            image_resized = image_resized.astype('float32') / 255.0
            image_resized = expand_dims(image_resized, 0)
            predicted_classes = self.model.predict(image_resized)
    
            boxes = list()
            for i in range(len(predicted_classes)):
                boxes += self.decode_netout(predicted_classes[i][0], self.anchors[i])

            self.correct_yolo_boxes(boxes)
            self.draw_boxes(boxes, class_threshold)


        


    # def localization(self):
    #     
    #     predicting_position = None
    #     first_prediction = False
    #     validation_matrix = []
    #     percentage_list = []

    #     for image in os.listdir(self.image_path):
    #         image_resized = cv2.imread(os.path.join(self.image_path, image))
    #         image_resized = cv2.resize(image_resized,(self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
    #     
    #         self.prepare_prediction()

    #         for i in range(len(self.image_file_image)):
    #             if self.number_classes == 2:
    #                 for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
    #                     for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
    #                         for j in range(self.validation_size):
    #                             for jj in range(self.validation_size):
    #                                 if self.predicted_classes_array[int(r/((math.sqrt(self.split_size)*(j+1))))][int(c/((math.sqrt(self.split_size)*(jj+1))))] == [np.argmax(self.predicted_classes[i], axis=0)]:
    #                                     validation_matrix.append(1)
    #                                 else:
    #                                     validation_matrix.append(0)

    #                         percentage = validation_matrix.count(1) / len(validation_matrix)
    #                         percentage_list.append(percentage)
    #                         
    #                         if first_prediction == False:
    #                             if percentage > 0.75:
    #                                 predicting_position[0] = self.predicted_classes_array[int(r/(math.sqrt(self.split_size)))][int(c/(math.sqrt(self.split_size)))]
    #                                 first_prediction = True

    #                         elif percentage < 0.45:
    #                             predicting_position[1] = self.predicted_classes_array[int(r/(math.sqrt(self.split_size)))][int(c/(math.sqrt(self.split_size)))]

    #                 print(percentage_list)

    #                 for jjj in range(len(self.box_index)):
    #                     prediction = self.predict_parts_images(self.box_index)
    #                     image_resized=cv2.rectangle(image_resized, self.box_index[jjj][0], self.box_index[jjj][1], self.color[np.argmax(prediction[i], axis=0)], self.thickness)
    #                     cv2.putText(image_resized, str((self.model_categpory[np.argmax(prediction[i], axis=0)])), first_predicting_position, self.font, self.fontScale, self.color[np.argmax(self.predicted_classes[i], axis=0)], self.thickness, cv2.LINE_AA)

    #                 cv2.imwrite(self.graph_path_localization + "model_segmenation_with_model_trained_prediction_" + str(self.save_model) + str(image) + '.png', image_resized)

    #             if self.number_classes == 4:
    #                 for r in range(0,image_resized.shape[0],int(math.sqrt(self.split_size))):
    #                     for c in range(0,image_resized.shape[1],int(math.sqrt(self.split_size))):
    #                         if first_prediction == False:
    #                             if self.predicted_classes_array[int(r/(math.sqrt(self.split_size)))][int(c/(math.sqrt(self.split_size)))] == [np.argmax(self.predicted_classes[i], axis=0)]:
    #                                 first_predicting_position = (int(r+(math.sqrt(self.split_size))), int(c+(math.sqrt(self.split_size))))
    #                                 first_prediction = True

    #                         elif self.predicted_classes_array[int(r/(math.sqrt(self.split_size)))][int(c/(math.sqrt(self.split_size)))] == [np.argmax(self.predicted_classes[i], axis=0)]:
    #                             last_predicting_position = (int(r+(math.sqrt(self.split_size))), int(c+(math.sqrt(self.split_size))))
    #                     
    #                         if c == int(self.image_size-(math.sqrt(self.split_size))) and r == int(self.image_size-(math.sqrt(self.split_size))):
    #                             image_resized=cv2.rectangle(image_resized, first_predicting_position, last_predicting_position, self.color[np.argmax(self.predicted_classes[i], axis=0)], self.thickness)
    #                             cv2.putText(image_resized, str((self.model_categpory[np.argmax(self.predicted_classes[i], axis=0)])), first_predicting_position, self.font, self.fontScale, self.color[np.argmax(self.predicted_classes[i], axis=0)], self.thickness, cv2.LINE_AA)


    #                 cv2.imwrite(self.graph_path_localization + "model_segmenation_with_model_trained_prediction_" + str(self.save_model) + str(image) + '.png', image_resized)



