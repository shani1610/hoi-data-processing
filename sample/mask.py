import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os

model = YOLO("yolov8n-seg.pt")  # segmentation model
names = model.model.names
cap = cv2.VideoCapture("video_w0_a1_c0.MOV")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

output_folder = "output_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

out = cv2.VideoWriter('instance-segmentation.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

frame_count = 0
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            annotator.seg_bbox(mask=mask,
                               mask_color=colors(int(cls), True),
                               det_label=names[int(cls)])

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    # Save each frame separately
    frame_filename = f"{output_folder}/frame_{frame_count:04d}.png"
    cv2.imwrite(frame_filename, im0)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
