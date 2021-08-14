from header_inputs import *

if __name__ == "__main__":
    
    # Determine if you want to create new files, second augument is where you create a new folder
    util = utilities()
    if len(sys.argv) != 1:
        if sys.argv[1] == "create":
            util.seperate_image_base_on_image(nested_folders = "True")

        # Seperate images base on names
        if sys.argv[1] == "seperate":
            util.seperate_image_into_file()


    # Begin analysis
    brain_analysis = brain_cancer_analysis()
