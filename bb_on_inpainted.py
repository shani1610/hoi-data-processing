# Example usage:
import cv2


def draw_resized_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on a resized image.
    Args:
        image: Resized image to draw on.
        bbox: Bounding box coordinates in YOLO format [center_x, center_y, width, height].
        color: Color of the bounding box (BGR format).
        thickness: Thickness of the bounding box lines.
    Returns:
        image: Resized image with bounding box drawn.
    """

    label, xc, yc, w, h = bbox
    new_h, new_w = image.shape[:2]

    xcn = int(xc*new_w) # center_x new
    ycn = int(yc*new_h) # center_y new
    wn = int(w*new_w) # width
    hn = int(h*new_h) # height

    # Calculate top-left and bottom-right coordinates of the bounding box
    x1 = (xcn - (wn / 2))
    y1 = (ycn - (hn / 2))
    x2 = (xcn + (wn / 2))
    y2 = (ycn + (hn / 2))
    
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Draw bounding box on the resized image
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return image



def extract_bbox_from_txt(txt_file_path):
    """
    Extract bounding box coordinates from a text file.
    Args:
        txt_file_path: Path to the text file containing bounding box information.
    Returns:
        bbox: List containing bounding box coordinates [center_x, center_y, width, height].
    """
    with open(txt_file_path, "r") as file:
        line = file.readline().strip()  # Read the first line (assuming there's only one line)
        bbox = list(map(float, line.split()))  # Convert the space-separated values to floats
    return bbox

original_image = cv2.imread("./data/Date01_Sub01_backpack_back_color/k0_color_W07.000.jpg")
resized_image = cv2.imread("./data/Date01_Sub01_backpack_back_inpainted/k0_color_W07.000.jpg")
#resized_inpainted_image = cv2.imread("./data/Date01_Sub01_backpack_back_color_inpainted/k0_color_W07.000.jpg")

# YOLO format bounding box coordinates for the original image
bbox = extract_bbox_from_txt("./output/k0_color_W07.000.txt")

# Draw bounding box on the resized image
image_with_resized_bbox = draw_resized_bbox(resized_image, bbox)

# Display the resized image with bounding box
cv2.imshow("Resized Image with Bounding Box", image_with_resized_bbox)
cv2.waitKey(3520)
cv2.destroyAllWindows()