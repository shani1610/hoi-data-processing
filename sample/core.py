import cv2
import os
import json
import sys

def video_to_frames(subfolder, video_id):
    video_path = f"data/vidor/training-video-part8/video/{subfolder}/{video_id}.mp4"
    json_file_path = f"data/vidor/training-annotation/training/{subfolder}/{video_id}.json"
    output_folder = f"output/{subfolder}/{video_id}"

    # Read video information from the JSON file
    with open(json_file_path, 'r') as json_file:
        video_info = json.load(json_file)

    # Check if required categories are present in the JSON file
    categories = {"adult", "racket"}
    if "subject/objects" in video_info and any(obj.get("category") in categories for obj in video_info["subject/objects"]):
        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get video properties from the provided information
        fps = video_info.get("fps", int(video_capture.get(cv2.CAP_PROP_FPS)))
        frame_count = video_info.get("frame_count", int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Read frames and save as images
        for i in range(frame_count):
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame to grayscale (optional)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save frame as an image
            image_name = f"{i:05d}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)

        # Release video capture object
        video_capture.release()
        print(f"Frames extracted and saved successfully to {output_folder}.")
    else:
        print(f"Skipping frame extraction for {video_path} as required categories are not present in the JSON file.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <video_subfolder> <video_id>")
    else:
        subfolder = sys.argv[1]
        video_id = sys.argv[2]
        video_to_frames(subfolder, video_id)