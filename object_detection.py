import cv2
import tensorflow as tf
import numpy as np
import os

def load_labels(labels_path):
    """
    Load COCO labels from a text file or .pbtxt file.
    """
    labels = []
    if not isinstance(labels_path, str):
        raise ValueError(f"labels_path must be a string, got {type(labels_path)}")
    
    with open(labels_path, 'r') as f:
        content = f.read()
    
    if 'item {' in content and 'display_name:' in content:
        lines = content.splitlines()
        for line in lines:
            if 'display_name:' in line:
                label = line.split('"')[1]
                labels.append(label)
    else:
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    
    return labels

def detect_objects_in_image(image_path, model, labels):
    """
    Detect objects in a single image and display results.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image at {image_path}")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb_frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = model(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        num_detections = int(detections.pop('num_detections'))

        im_height, im_width, _ = frame.shape

        for i in range(num_detections):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                xmin_scaled = int(xmin * im_width)
                xmax_scaled = int(xmax * im_width)
                ymin_scaled = int(ymin * im_height)
                ymax_scaled = int(ymax * im_height)
                cv2.rectangle(frame, (xmin_scaled, ymin_scaled), (xmax_scaled, ymax_scaled), (0, 255, 0), 2)
                label = labels[classes[i] - 1]
                cv2.putText(frame, label, (xmin_scaled, ymin_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(f"Object Detection - {os.path.basename(image_path)}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def detect_objects_in_video(video_path, model, labels):
    """
    Detect objects in a video and display results frame by frame.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: FPS={fps}, Total frames={frame_count}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Reached end of video or failed to read frame at frame {frame_idx}")
                break

            frame_idx += 1
            print(f"Processing frame {frame_idx}/{frame_count}")

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(rgb_frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            try:
                detections = model(input_tensor)
            except Exception as e:
                print(f"Error during model inference on frame {frame_idx}: {e}")
                break

            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            scores = detections['detection_scores'][0].numpy()
            num_detections = int(detections.pop('num_detections'))

            im_height, im_width, _ = frame.shape

            for i in range(num_detections):
                if scores[i] > 0.5:
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin_scaled = int(xmin * im_width)
                    xmax_scaled = int(xmax * im_width)
                    ymin_scaled = int(ymin * im_height)
                    ymax_scaled = int(ymax * im_height)
                    cv2.rectangle(frame, (xmin_scaled, ymin_scaled), (xmax_scaled, ymax_scaled), (0, 255, 0), 2)
                    label = labels[classes[i] - 1]
                    cv2.putText(frame, label, (xmin_scaled, ymin_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Object Detection - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted video playback")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video processing completed")

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def detect_objects_in_directory(image_directory, model_path, labels_path):
    """
    Detects objects in all image and video files within a given directory using the specified model.
    """
    try:
        print(f"Loading model from: {model_path}")
        model = tf.saved_model.load(model_path)
        infer = model.signatures["serving_default"]
        print("Model loaded successfully")

        print(f"Loading labels from: {labels_path}")
        labels = load_labels(labels_path)
        print(f"Loaded {len(labels)} labels: {labels}")

        VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        VALID_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}

        files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
        print(f"Found {len(files)} files in directory: {files}")

        for file in files:
            file_path = os.path.join(image_directory, file)
            file_ext = os.path.splitext(file)[1].lower()
            print(f"Checking file: {file_path} (extension: {file_ext})")

            if file_ext in VALID_IMAGE_EXTENSIONS:
                print(f"Processing as image: {file_path}")
                detect_objects_in_image(file_path, infer, labels)
            elif file_ext in VALID_VIDEO_EXTENSIONS:
                print(f"Processing as video: {file_path}")
                detect_objects_in_video(file_path, infer, labels)
            else:
                print(f"Skipping non-supported file: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Choose which model to use ---
model_choice = "faster_rcnn"  # Options: "mobilenet", "faster_rcnn"

image_directory = r"E:/SATHVIK/study/Projects/Object Detection Using Tensorflow"
labels_path = r"E:/SATHVIK/study/Projects/Object Detection Using Tensorflow/coco_labels.txt"

if model_choice == "mobilenet":
    model_path = r"E:/SATHVIK/study/Projects/Object Detection Using Tensorflow/mobilenet-v2-tensorflow2-035-128-classification-v2"
    print("Warning: You've selected a classification model. It will not perform object detection with bounding boxes.")
elif model_choice == "faster_rcnn":
    model_path = r"E:/SATHVIK/study/Projects/Object Detection Using Tensorflow/faster-rcnn-inception-resnet-v2-tensorflow2-1024x1024-v1"
    detect_objects_in_directory(image_directory, model_path, labels_path)
else:
    print("Invalid model choice. Please select 'mobilenet' or 'faster_rcnn'.")