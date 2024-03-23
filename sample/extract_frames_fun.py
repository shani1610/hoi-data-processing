import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the padding size for naming frames
    pad_size = len(str(total_frames))

    # Loop through each frame and save it as an image
    start_frame_index = 4
    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Determine the frame index
        frame_index = frame_number + start_frame_index  # Assuming start_frame_index is defined somewhere
        
        # Save frame as an image in the output folder
        if 4 <= frame_index <= 49:
            # Construct the frame name with the desired structure
            frame_name = f"k0_color_W{str(frame_index).zfill(2)}.000.jpg"        
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Get video path from the user
    video_path = input("Enter the path to the video file: ").strip()

    # Check if the file exists
    if not os.path.exists(video_path):
        print("Error: The specified video file does not exist.")
    else:
        # Create output folder with the same name as the video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join("output", video_name)

        # Call the function to extract frames
        extract_frames(video_path, output_folder)

        print(f"Frames extracted and saved in: {output_folder}")
