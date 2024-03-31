import os
import torch
import supervision as sv
import os
from groundingdino.util.inference import Model
import cv2
import supervision as sv
import tqdm
import argparse

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

def bb_for_image(SOURCE_IMAGE_PATH, CLASS, annotate=False):
    if type(CLASS) != list:
        CLASS = [CLASS]

    image = cv2.imread(SOURCE_IMAGE_PATH)
    height, width = image.shape[:2]
    caption = ", ".join(CLASS)
    detections, labels = model.predict_with_caption(
        image=image,
        caption=caption,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    yolo_bbox = []
    if detections:
        yolo_bbox = convert_to_yolo(detections.xyxy[0], width, height)
        #print(detections.xyxy)
    #else:
        #print(f"No detections found for {SOURCE_IMAGE_PATH}.")

    if annotate:
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{label} {confidence:0.2f}" 
            for label, (_, confidence, class_id, _) 
            in zip(labels, detections)]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        sv.plot_image(annotated_frame, (16, 16))
    
    return yolo_bbox

def run_on_one_folder(data_folder, class_label, name_of_subject, annotation_output_folder):
    # Iterate through files in the current directory
    for idx, file in enumerate(sorted(os.listdir(data_folder))):
        # Check if the file name matches the pattern "kN_color_W0M.000.jpg"
        if file.startswith("k") and file.endswith(".jpg"):
            # Process the image (perform your desired operation)
            image_path = os.path.join(data_folder, file)
            yolo_bbox = bb_for_image(image_path, CLASS=class_label, annotate=False)
            class_num = get_class_number(class_label, conversion_dict)
            output_path = os.path.join(annotation_output_folder, name_of_subject+str(idx).zfill(4)+'.txt')
            save_annotations(image_path, yolo_bbox, class_num, annotation_output_folder, output_path)

def convert_to_yolo(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    
    # Calculate the center of the bounding box
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    
    # Normalize the coordinates relative to the image dimensions
    normalized_bbox_center_x = bbox_center_x / image_width
    normalized_bbox_center_y = bbox_center_y / image_height
    normalized_bbox_width = (x2 - x1) / image_width
    normalized_bbox_height = (y2 - y1) / image_height
    
    # Format the coordinates in YOLO format
    yolo_format = [
        normalized_bbox_center_x,
        normalized_bbox_center_y,
        normalized_bbox_width,
        normalized_bbox_height
    ]
    
    return yolo_format

# check the correctness
def check_the_yolo_conversion(img_path, lab_path):
    # Load the image
    image = cv2.imread(img_path)
    yolo_bbox = []
    # Assuming bbox is a list of YOLO format coordinates [center_x, center_y, width, height]
    # Open the file for reading
    with open(lab_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into a list of numbers
            numbers = line.split()
            # Extract the last four columns from the list of numbers
            last_four_columns = list(map(float, numbers[-4:]))
            # Append the last four columns to the data list
            yolo_bbox.append(last_four_columns)
    print(yolo_bbox[0])
    if not yolo_bbox:
        # If data is empty, break out of the function
        print("Data is empty. Exiting function.")
    
    else:
        # Draw bounding box on the image
        image_with_bbox = draw_bbox(image, yolo_bbox[0])

        display_works = False
        if display_works:
            # Display the image with bounding box
            cv2.imshow("Image with Bounding Box", image_with_bbox)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        # Specify the file path where you want to save the image
        output_path = './output_image.jpg'

        # Save the image with bounding boxes as a JPEG file
        cv2.imwrite(output_path, image_with_bbox)

        print(f"Image saved as {output_path}")

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on the image.
    Args:
        image: Image to draw on.
        bbox: Bounding box coordinates in YOLO format [center_x, center_y, width, height].
        color: Color of the bounding box (BGR format).
        thickness: Thickness of the bounding box lines.
    Returns:
        image: Image with bounding box drawn.
    """
    height, width = image.shape[:2]
    
    if len(bbox) == 0:
        return image

    # Convert YOLO format to bounding box coordinates
    bbox_center_x = int(bbox[0] * width)
    bbox_center_y = int(bbox[1] * height)
    bbox_width = int(bbox[2] * width)
    bbox_height = int(bbox[3] * height)
    #print("bbox_center_x", bbox_center_x, bbox_center_y, bbox_width, bbox_height)
    # Calculate top-left and bottom-right coordinates of the bounding box
    x1 = int(bbox_center_x - bbox_width / 2)
    y1 = int(bbox_center_y - bbox_height / 2)
    x2 = int(bbox_center_x + bbox_width / 2)
    y2 = int(bbox_center_y + bbox_height / 2)

    #print("x1", x1, y1, x2, y2)

    # Draw bounding box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return image

def save_annotations(image_path, yolo_bbox, class_number, output_folder, output_path=None):
    """
    Save annotations (class number and bounding box) to a text file.
    """
    # Construct output file path
    if output_path is None:
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        output_path = os.path.join(output_folder, output_filename)
    
    # Open the file for writing
    with open(output_path, "w") as file:
        if len(yolo_bbox) == 0:
            line = f"\n"
        else:
        # Write class label and bounding box coordinates for each detection   
            line = f"{class_number} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
        file.write(line)

conversion_dict = {
    "backpack": 0,
    "basketball": 1,
    "boxlarge": 2,
    "boxlong": 3,
    "boxmedium": 4,
    "boxsmall": 5,
    "boxtiny": 6,
    "chairblack": 7,
    "chairwood": 8,
    "keyboard": 9,
    "monitor": 10,
    "plasticcontainer": 11,
    "stool": 12,
    "suitcase": 13,
    "tablesmall": 14,
    "tablesquare": 15,
    "toolbox": 16,
    "trashbin": 17,
    "yogaball": 18,
    "yogamat": 19,
}

def get_class_number(class_label, conversion_dict):
    """
    Get the class number from the conversion dictionary based on the class label.
    """
    if type(class_label) == list:
        class_label = class_label[0]
    # print(class_label)
    if class_label in conversion_dict:
        return conversion_dict[class_label]
    else:
        return None  # Return None if class label is not found in the dictionary

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--img_path', type=str, help='Path of the input video or image folder.')
    parser.add_argument(
        '-l', '--lab_path', type=str, help='Path of the mask(s) or mask folder.')    
    args = parser.parse_args()

    if args.img_path is not None:
        img_path = args.img_path
    if args.lab_path is not None:
        lab_path = args.lab_path  
    check_the_yolo_conversion(img_path, lab_path)

# we will run: 

# python check_img_lab_corr.py --img_path "./data/hoi_dataset/Date01_Sub01_backpack_back/images/Date01_Sub01_backpack_back0022.jpg" --lab_path "./data/hoi_dataset/Date01_Sub01_backpack_back/labels/Date01_Sub01_backpack_back0022.txt"
