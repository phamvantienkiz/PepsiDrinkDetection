# test_yolo_improved.py
import argparse
import cv2
import numpy as np
import random
from ultralytics import YOLO
import tensorflow as tf
from collections import defaultdict
import os

# ------------ CONFIG ------------
MAX_DISPLAY_WIDTH = 800   # max width of window (pixels). chỉnh nếu muốn
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0           # base font scale (tweak nếu cần)
FONT_THICKNESS = 3         # font thickness (bold)
BOX_THICKNESS = 3          # bbox thickness
TEXT_PADDING = 8           # padding inside the background box for text
LINE_SPACING = 10          # vertical spacing between summary lines
# ---------------------------------

def get_colors(names):
    random.seed(42)
    colors = {}
    for i, _ in enumerate(names if isinstance(names, dict) == False else list(names.values())):
        # nicer colors: choose from HSV -> BGR for visibility
        hue = int(180 * i / max(1, len(names)))
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0,0])
        colors[i] = (int(color[0]), int(color[1]), int(color[2]))
    return colors

def draw_results(img, boxes, classes, class_names, colors, confs=None):
    """
    boxes: Nx4 numpy array x1,y1,x2,y2 in pixels (image coordinates)
    classes: list or array of class indices
    class_names: dict idx->name or list
    colors: dict idx->(B,G,R)
    confs: optional confidences list for ordering (not printed)
    """
    h, w = img.shape[:2]
    summary = defaultdict(int)
    # draw boxes
    for i, (box, cls_idx) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = colors.get(int(cls_idx), (0,255,0))
        # clip coords
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w-1,x2), min(h-1,y2)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, BOX_THICKNESS, lineType=cv2.LINE_AA)
        summary[int(cls_idx)] += 1

    # prepare summary lines sorted by class idx or by count desc
    # We'll sort by class index to keep consistent ordering
    lines = []
    for idx in sorted(summary.keys()):
        name = class_names[idx] if isinstance(class_names, dict) else class_names[idx]
        lines.append((idx, f"{name}: {summary[idx]}"))

    # Draw a translucent background rectangle to improve readability
    # compute total height needed
    line_height = int(FONT_SCALE * 30) + LINE_SPACING
    total_h = len(lines) * line_height + TEXT_PADDING*2
    total_w = 0
    for _, text in lines:
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        total_w = max(total_w, tw)
    total_w += TEXT_PADDING*2

    # Background rectangle position (top-left)
    bg_x1, bg_y1 = 10, 10
    bg_x2, bg_y2 = bg_x1 + total_w, bg_y1 + total_h

    # draw semi-transparent rect
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,0,0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # put each line with colored left square and text
    for i, (idx, text) in enumerate(lines):
        y = bg_y1 + TEXT_PADDING + int((i+0.8)*line_height)
        color = colors.get(idx, (255,255,255))
        # small color box
        box_size = int(line_height * 0.6)
        cb_x1 = bg_x1 + TEXT_PADDING//2
        cb_y1 = y - int(box_size*0.75)
        cb_x2 = cb_x1 + box_size
        cb_y2 = cb_y1 + box_size
        cv2.rectangle(img, (cb_x1, cb_y1), (cb_x2, cb_y2), color, -1)
        # text (to the right of color box)
        text_x = cb_x2 + TEXT_PADDING//2
        cv2.putText(img, text, (text_x, y), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS, cv2.LINE_AA)

    return img

def show_image_window(img, window_name="YOLO Result", wait=True, max_width=MAX_DISPLAY_WIDTH, save_out=None):
    # scale down if too wide
    h, w = img.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, disp)
    # optionally resize window to image size to allow user to rescale
    try:
        cv2.resizeWindow(window_name, disp.shape[1], disp.shape[0])
    except Exception:
        pass

    if save_out:
        cv2.imwrite(save_out, img)
        print(f"Saved output image to {save_out}")

    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

def run_pt(model_path, image_path, save_out=None):
    model = YOLO(model_path)
    results = model(image_path)[0]

    names = model.names  # dict idx->name
    colors = get_colors(names)

    img = cv2.imread(image_path)
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    out = draw_results(img, boxes, classes, names, colors)
    show_image_window(out, window_name="YOLO PT Result", save_out=save_out)

def run_tflite(model_path, image_path, labels_path, save_out=None, conf_thres=0.25):
    # load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    names = {i: label for i, label in enumerate(labels)}
    colors = get_colors(names)

    # load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get expected input size (H,W)
    in_shape = input_details[0]['shape']
    # support dynamic or fixed shapes
    if len(in_shape) == 4:
        in_h, in_w = int(in_shape[1]), int(in_shape[2])
    else:
        # fallback
        in_h, in_w = 640, 640

    # IMPORTANT: use letterbox (keep aspect ratio) -> model expects square input  (simple resize used here)
    resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Attempt to parse outputs: many TF-Lite exports have different output orders.
    # Here we try common patterns: boxes, classes, scores
    out_tensors = [interpreter.get_tensor(o['index']) for o in output_details]
    # naive heuristic: find boxes as [1,N,4] and scores [1,N] and classes [1,N]
    boxes, classes, scores = None, None, None
    for t in out_tensors:
        if t.ndim == 3 and t.shape[2] == 4:
            boxes = t[0]
        elif t.ndim == 2 and t.shape[1] == 1:
            classes = t[0].squeeze().astype(int)
        elif t.ndim == 2 and t.shape[1] == 4:
            # alternative
            pass
        elif t.ndim == 2 and t.shape[1] >= 1:
            # could be scores
            if scores is None and (t.max() <= 1.0):
                scores = t[0]
    # fallback by output_details order if heuristic fails
    if boxes is None:
        try:
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
        except Exception as e:
            print("Could not parse TFLite outputs automatically. Output details:")
            for o in output_details:
                print(o)
            raise e

    H, W, _ = img.shape
    final_boxes, final_classes = [], []
    for box, cls, score in zip(boxes, classes, scores):
        if score > conf_thres:
            # many TFLite models use [ymin, xmin, ymax, xmax] normalized
            ymin, xmin, ymax, xmax = box
            final_boxes.append([xmin*W, ymin*H, xmax*W, ymax*H])
            final_classes.append(int(cls))

    out = draw_results(img, np.array(final_boxes), np.array(final_classes), names, colors)
    show_image_window(out, window_name="YOLO TFLite Result", save_out=save_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model (.pt or .tflite)")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument("--labels", type=str, default="labels.txt", help="labels file (for tflite only)")
    parser.add_argument("--save_out", type=str, default=None, help="path to save output image (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for detections")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    if args.model.endswith(".pt"):
        run_pt(args.model, args.image, save_out=args.save_out)
    elif args.model.endswith(".tflite"):
        if not os.path.exists(args.labels):
            raise FileNotFoundError(f"Labels file not found: {args.labels}")
        run_tflite(args.model, args.image, args.labels, save_out=args.save_out, conf_thres=args.conf)
    else:
        print("Unsupported model format. Use .pt or .tflite")
