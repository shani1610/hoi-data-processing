import os
import shutil

def organize_files(root_dir):
    output_path = "./data/behave_seq_for_inpaint"

    for date_folder in os.listdir(root_dir):
        date_folder_path = os.path.join(root_dir, date_folder)  # Date0X_Sub0Y_Z
        color_folder = os.path.join(output_path, f"{date_folder}_color")
        mask_folder = os.path.join(output_path, f"{date_folder}_obj_rend_mask")
        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        if os.path.isdir(date_folder_path):
            for sub_folder in os.listdir(date_folder_path):
                sub_folder_path = os.path.join(date_folder_path, sub_folder)  # t000W.000
                if os.path.isdir(sub_folder_path):
                    W_value = sub_folder[3:]  # Extract W value from the subfolder name
                    for file in os.listdir(sub_folder_path):
                        if file.startswith("k0.color.jpg"):
                            destination_filename_color = f"k0_color_W{W_value}.jpg"
                            shutil.copy2(os.path.join(sub_folder_path, file), os.path.join(color_folder, destination_filename_color))
                        elif file.startswith("k0.obj_rend_mask.jpg"):
                            destination_filename_mask = f"k0_obj_rend_mask_W{W_value}.jpg"
                            shutil.copy2(os.path.join(sub_folder_path, file), os.path.join(mask_folder, destination_filename_mask))

if __name__ == "__main__":
    root_directory = "./data/behave_seq"
    organize_files(root_directory)