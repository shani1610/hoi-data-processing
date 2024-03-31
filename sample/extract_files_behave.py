import os
import shutil
import argparse
import cv2

def organize_files(root_dir, output_color, output_inpaint):
    for date_folder in os.listdir(root_dir):
        date_folder_path = os.path.join(root_dir, date_folder)
        
        color_folder = os.path.join(output_color, f"{date_folder}_color")
        inpaint_folder = os.path.join(output_inpaint, f"{date_folder}")
        mask_resized_folder = os.path.join(inpaint_folder, f"{date_folder}_mask_resized")
        color_resized_folder = os.path.join(inpaint_folder, f"{date_folder}_color_resized")

        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(inpaint_folder, exist_ok=True)
        os.makedirs(mask_resized_folder, exist_ok=True)
        os.makedirs(color_resized_folder, exist_ok=True)

        if os.path.isdir(date_folder_path):
            for sub_folder in os.listdir(date_folder_path):
                sub_folder_path = os.path.join(date_folder_path, sub_folder)
                if os.path.isdir(sub_folder_path):
                    W_value = sub_folder[3:]  # Extract W value from the subfolder name
                    for file in os.listdir(sub_folder_path):
                        if file.startswith("k0.color.jpg"):
                            destination_filename_color = f"k0_color_W{W_value}.jpg"
                            shutil.copy2(os.path.join(sub_folder_path, file), os.path.join(color_folder, destination_filename_color))
                            destination_filename_color_resized = f"k0_color_W{W_value}_resized.jpg"
                            image = cv2.imread(os.path.join(sub_folder_path, file))
                            resized_image = cv2.resize(image, (432, 240))
                            cv2.imwrite(os.path.join(color_resized_folder, destination_filename_color_resized), resized_image)
                        elif file.startswith("k0.obj_rend_mask.jpg"):
                            destination_filename_mask_resized = f"k0_obj_rend_mask_W{W_value}_resized.jpg"
                            mask_image = cv2.imread(os.path.join(sub_folder_path, file))
                            resized_mask_image = cv2.resize(mask_image, (432, 240))
                            cv2.imwrite(os.path.join(mask_resized_folder, destination_filename_mask_resized), resized_mask_image)

if __name__ == "__main__":

    root_directory = "./data/preprocessed_behave"

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Your script description here")
    # Add argument
    parser.add_argument("-r", "--root_directory", default="./data/preprocessed_behave", help="Specify the root directory path")
    parser.add_argument("-c", "--output_color", default="./data/images_for_gdino/", help="Specify the output color path for grounding dino object detection, full size images")
    parser.add_argument("-i", "--output_inpaint", default="./data/images_for_ppainter/", help="Specify the output path for inpaint path")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the argument value
    output_color = args.output_color
    output_inpaint = args.output_inpaint

    organize_files(root_directory, output_color, output_inpaint)

    # in terminal run: 
    # python ./sample/extract_files_behave.py --root_directory "./data/preprocessed_behave" --output_color "./data/images_for_gdino/" --output_inpaint "./data/images_for_ppainter/"
    # python ./sample/extract_files_behave.py 
    
