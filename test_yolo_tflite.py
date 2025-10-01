# test_yolo_tflite.py
import argparse
import cv2
import numpy as np
import tensorflow as tf
import os
from collections import defaultdict

# ------------ CONFIG ------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 3
BOX_THICKNESS = 3
CONF_THRES = 0.25
IOU_THRES = 0.45
# ---------------------------------

def get_colors(num_classes):
    np.random.seed(42)
    return {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(num_classes)}

def non_max_suppression(boxes, scores, classes, conf_thres=0.25, iou_thres=0.45):
    if len(boxes) == 0:
        return [], [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),  # [[x,y,w,h], ...]
        scores=scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )
    if len(idxs) == 0:
        return [], [], []
    idxs = idxs.flatten()
    return boxes[idxs].tolist(), scores[idxs].tolist(), classes[idxs].tolist()


def run_tflite(model_path, image_path, labels_path, save_out=None):
    # load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    colors = get_colors(len(labels))

    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load image
    img = cv2.imread(image_path)
    H, W, _ = img.shape
    in_h, in_w = input_details[0]['shape'][1:3]

    img_resized = cv2.resize(img, (in_w, in_h))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])  # [1, 64, 8400]
    preds = np.squeeze(preds).T  # -> (8400, 64)

    # split boxes/scores
    boxes_xywh = preds[:, :4]
    scores_all = preds[:, 4:]
    class_ids = np.argmax(scores_all, axis=1)
    confidences = np.max(scores_all, axis=1)

    # filter by conf
    mask = confidences > CONF_THRES
    boxes_xywh = boxes_xywh[mask]
    scores = confidences[mask]
    class_ids = class_ids[mask]

    # convert xywh (normalized) -> xywh pixel
    boxes = []
    for (x, y, w_box, h_box) in boxes_xywh:
        x1 = int((x - w_box/2) * W / in_w * in_w)  # scale theo W
        y1 = int((y - h_box/2) * H / in_h * in_h)
        x2 = int((x + w_box/2) * W / in_w * in_w)
        y2 = int((y + h_box/2) * H / in_h * in_h)
        boxes.append([x1, y1, x2 - x1, y2 - y1])

    # nms
    boxes, scores, class_ids = non_max_suppression(boxes, scores, class_ids, CONF_THRES, IOU_THRES)

    # draw
    for (box, score, cls) in zip(boxes, scores, class_ids):
        x, y, w_box, h_box = box
        color = colors[cls]
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), color, BOX_THICKNESS)

    # summary label box
    summary = defaultdict(int)
    for c in class_ids:
        summary[c] += 1
    y0 = 30
    for c, cnt in summary.items():
        text = f"{labels[c]}: {cnt}"
        cv2.putText(img, text, (10, y0), FONT, FONT_SCALE, colors[c], FONT_THICKNESS, cv2.LINE_AA)
        y0 += 30

    if save_out:
        cv2.imwrite(save_out, img)
        print(f"Saved: {save_out}")
    else:
        cv2.imshow("YOLO TFLite", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to .tflite")
    parser.add_argument("--image", type=str, required=True, help="path to image")
    parser.add_argument("--labels", type=str, default="labels.txt", help="labels file")
    parser.add_argument("--save_out", type=str, default=None, help="save output image path")
    args = parser.parse_args()

    run_tflite(args.model, args.image, args.labels, save_out=args.save_out)
