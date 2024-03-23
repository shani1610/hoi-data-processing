import os
import cv2
import sys

def resize_images(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_directory):
        # Construct the full path of the input image
        input_path = os.path.join(input_directory, filename)

        # Read the image
        image = cv2.imread(input_path)

        # Resize the image to 432x240 pixels
        resized_image = cv2.resize(image, (432, 240))

        # Construct the full path of the output image
        output_path = os.path.join(output_directory, filename)

        # Save the resized image
        cv2.imwrite(output_path, resized_image)

        #print(f'Image resized and saved: {output_path}')

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    # Get input and output directories from command-line arguments
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Call the function with command-line argument directories
    resize_images(input_directory, output_directory)

    # resizing from 1920 × 1080 pixels to 432 × 240 pixels since CUDA error