import os
import random
import shutil
import argparse

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(input_dir, output_dir, move_files):
    print("Input directory:", input_dir)
    print("Output directory:", output_dir)
    print("Move files:", move_files)

    # Define the directories for train, test, and valid sets
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    test_images_dir = os.path.join(output_dir, 'test', 'images')
    valid_images_dir = os.path.join(output_dir, 'valid', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    test_labels_dir = os.path.join(output_dir, 'test', 'labels')
    valid_labels_dir = os.path.join(output_dir, 'valid', 'labels')

    print("Train images directory:", train_images_dir)
    print("Test images directory:", test_images_dir)
    print("Valid images directory:", valid_images_dir)
    print("Train labels directory:", train_labels_dir)
    print("Test labels directory:", test_labels_dir)
    print("Valid labels directory:", valid_labels_dir)

    # Create train, test, and valid directories if they don't exist
    create_directory(train_images_dir)
    create_directory(test_images_dir)
    create_directory(valid_images_dir)
    create_directory(train_labels_dir)
    create_directory(test_labels_dir)
    create_directory(valid_labels_dir)

    # Function to either move or copy files
    action_func = shutil.move if move_files else shutil.copy

    # Iterate over each directory in the input_dir
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for dirname in dirnames:
            # Check if the directory starts with 'Date01_Sub01_'
            if dirname.startswith('Date01_Sub01_'):
                # Define the source directories for images and labels
                images_dir = os.path.join(dirpath, dirname, 'images')
                labels_dir = os.path.join(dirpath, dirname, 'labels')

                print("Processing directory:", dirname)
                print("Images directory:", images_dir)
                print("Labels directory:", labels_dir)

                # Gather list of image and label files
                image_files = sorted([file for file in os.listdir(images_dir) if file.endswith('.jpg')])
                label_files = sorted([file for file in os.listdir(labels_dir) if file.endswith('.txt')])

                # Calculate the number of files for train, test, and valid sets
                total_files = len(image_files)
                train_count = int(total_files * 0.8)
                test_count = int(total_files * 0.1)
                valid_count = total_files - train_count - test_count

                # Randomly select files for train, test, and valid sets
                selected_files = random.sample(range(total_files), total_files)
                train_files = selected_files[:train_count]
                test_files = selected_files[train_count:train_count + test_count]
                valid_files = selected_files[train_count + test_count:]

                # Move or copy selected files to train set
                for idx in train_files:
                    image_file = image_files[idx]
                    label_file = label_files[idx]
                    action_func(os.path.join(images_dir, image_file), os.path.join(train_images_dir, image_file))
                    action_func(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))

                # Move or copy selected files to test set
                for idx in test_files:
                    image_file = image_files[idx]
                    label_file = label_files[idx]
                    action_func(os.path.join(images_dir, image_file), os.path.join(test_images_dir, image_file))
                    action_func(os.path.join(labels_dir, label_file), os.path.join(test_labels_dir, label_file))

                # Move or copy selected files to validation set
                for idx in valid_files:
                    image_file = image_files[idx]
                    label_file = label_files[idx]
                    action_func(os.path.join(images_dir, image_file), os.path.join(valid_images_dir, image_file))
                    action_func(os.path.join(labels_dir, label_file), os.path.join(valid_labels_dir, label_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, test, and validation sets for YOLO")
    parser.add_argument("input_dir", help="Path to the input directory containing the dataset")
    parser.add_argument("output_dir", help="Path to the output directory where the split dataset will be saved")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them (default is copy)")
    args = parser.parse_args()

    # Split the dataset
    split_dataset(args.input_dir, args.output_dir, args.move)
