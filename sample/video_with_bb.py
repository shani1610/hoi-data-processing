import cv2
import json
import os
import sys

def draw_bounding_boxes(frame_directory, json_file_path, output_directory, output_video_path):
    # Load bounding boxes from JSON file
    with open(json_file_path, 'r') as json_file:
        bounding_boxes_dict = json.load(json_file)

    # Get the frame names in order
    frame_names = [f"frame_{i:02d}.jpg" for i in range(len(bounding_boxes_dict))]

    # Get the first frame to set up the VideoWriter
    first_frame_name = frame_names[0]
    first_frame_path = os.path.join(frame_directory, first_frame_name)
    first_frame = cv2.imread(first_frame_path)
    frame_height, frame_width, _ = first_frame.shape

    # Set up VideoWriter with explicit file names
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    output_video_path = os.path.join(output_directory, output_video_path)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

    # Iterate through frames in order
    for frame_name in frame_names:
        # Read the original frame
        frame_path = os.path.join(frame_directory, frame_name)
        frame = cv2.imread(frame_path)

        # Get bounding boxes for the current frame
        bounding_boxes = bounding_boxes_dict.get(frame_name, [])

        # Draw bounding boxes on the original frame
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box["x"], bounding_box["y"], bounding_box["width"], bounding_box["height"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the frame with bounding boxes
        output_path = os.path.join(output_directory, f"{os.path.splitext(frame_name)[0]}_with_boxes.jpg")
        cv2.imwrite(output_path, frame)

        # Write frame to video
        video_writer.write(frame)

    # Release VideoWriter
    video_writer.release()

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python script.py <frame_directory> <json_file_path> <output_directory> <output_video_filename>")
        sys.exit(1)

    # Get frame directory, JSON file path, output directory, and output video filename from command-line arguments
    frame_directory = sys.argv[1]
    json_file_path = sys.argv[2]
    output_directory = sys.argv[3]
    output_video_filename = sys.argv[4]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Call the function to draw bounding boxes on frames and create the video
    draw_bounding_boxes(frame_directory, json_file_path, output_directory, output_video_filename)

    # python script.py /path/to/frame_directory /path/to/bounding_boxes.json /path/to/output_directory /path/to/output_video.mp4

    # python ./sample/video_with_bb.py "./output/inpaint_out" "./data/behave_seq_for_inpaint/resized/Date01_Sub01_backpack_back/Date01_Sub01_backpack_back_obj_rend_mask_resized_with_boxes/bounding_boxes.json" "./data/frames_with_bb" "./data/frames_with_bb"