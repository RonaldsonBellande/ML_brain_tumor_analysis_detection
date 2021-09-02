from header_inputs import *
from brain_tumor_model_training import *

if __name__ == "__main__":
    
    # Determine if you want to create new files, second augument is where you create a new folder
    util = utilities()
    if len(sys.argv) != 1:
        if sys.argv[1] == "create":
            util.seperate_image_base_on_image(nested_folders = "True")

        # Seperate images base on names
        if sys.argv[1] == "seperate":
            util.seperate_image_into_file()


    # Begin analysis for building model or training it
    if len(sys.argv) != 1:
        if sys.argv[1] == "model_building":
            brain_analysis_obj = brain_tumor_building(number_classes = sys.argv[2], model_type = sys.argv[3])

        # Seperate images base on names
        if sys.argv[1] == "model_training":

            brain_analysis_obj = brain_tumor_training(number_classes = sys.argv[2], model_type = sys.argv[3])
