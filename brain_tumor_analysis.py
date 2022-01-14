from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:
        if sys.argv[1] == "create":
            util = utilities()
            util.seperate_image_base_on_image(nested_folders = "True")

        if sys.argv[1] == "seperate":
            util = utilities()
            util.seperate_image_into_file()

        if sys.argv[1] == "model_building":
            brain_analysis_obj = brain_tumor_building(number_classes=sys.argv[2], model_type=sys.argv[3], image_type=sys.argv[4])

        if sys.argv[1] == "model_training":
            brain_analysis_obj = brain_tumor_training(number_classes=sys.argv[2], model_type=sys.argv[3], image_type=sys.argv[4])

        if sys.argv[1] == "image_prediction":
            
            if sys.argv[2] == "2":
                if sys.argv[3] == "model1":
                    input_model = "model1_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model2":
                    input_model = "model2_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model3":
                    input_model = "model3_computer_vision_categories_10_model.h5"

            elif sys.argv[2] == "4":
                if sys.argv[3] == "model1":
                    input_model = "model1_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model2":
                    input_model = "model2_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model3":
                    input_model = "model3_computer_vision_categories_10_model.h5"

            computer_vision_analysis_obj = classification_with_model(save_model=input_model)

        if sys.argv[1] == "transfer_learning":

            if sys.argv[2] == "2":
                if sys.argv[3] == "model1":
                    input_model = "model1_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model2":
                    input_model = "model2_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model3":
                    input_model = "model3_computer_vision_categories_10_model.h5"

            elif sys.argv[2] == "4":
                if sys.argv[3] == "model1":
                    input_model = "model1_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model2":
                    input_model = "model2_computer_vision_categories_10_model.h5"
                elif sys.argv[3] == "model3":
                    input_model = "model3_computer_vision_categories_10_model.h5"
            
            computer_vision_analysis_obj = computer_vision_transfer_learning(save_model=input_model, model_type=sys.argv[4])



