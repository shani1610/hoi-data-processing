import cv2
import json
import os
import sys
import numpy as np

def get_combined_bounding_box(mask_path):
    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine all contours to find the bounding box that includes all pieces
    combined_contour = None
    for contour in contours:
        if combined_contour is None:
            combined_contour = contour
        else:
            combined_contour = cv2.convexHull(np.vstack((combined_contour, contour)))

    # Get bounding box
    x, y, w, h = cv2.boundingRect(combined_contour)
    bounding_box = {"x": x, "y": y, "width": w, "height": h}

    return bounding_box

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <mask_directory>")
        sys.exit(1)

    # Get mask directory from command-line arguments
    mask_directory = sys.argv[1]

    # Create a new directory for images with bounding boxes
    output_directory = os.path.join(mask_directory + "_with_boxes")
    os.makedirs(output_directory, exist_ok=True)

    # Dictionary to store bounding boxes for each frame
    bounding_boxes_dict = {}

    # Iterate through mask images in the directory
    for filename in sorted(os.listdir(mask_directory)):
        if filename.endswith(".jpg"):  # Adjust the file extension if needed
            frame_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(mask_directory, filename)

            # Get combined bounding box for the current frame
            bounding_box = get_combined_bounding_box(mask_path)

            # Store bounding box in the dictionary
            bounding_boxes_dict[frame_name] = [bounding_box]

            # Draw bounding box on the mask image (for visualization purposes)
            frame = cv2.imread(mask_path)
            x, y, w, h = bounding_box["x"], bounding_box["y"], bounding_box["width"], bounding_box["height"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the image with bounding box in the new directory
            output_path = os.path.join(output_directory, f"{frame_name}_with_boxes.jpg")
            cv2.imwrite(output_path, frame)

    # Save bounding boxes to JSON file in the new directory
    json_file_path = os.path.join(output_directory, "bounding_boxes.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(bounding_boxes_dict, json_file, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()