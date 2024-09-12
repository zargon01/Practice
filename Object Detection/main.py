import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load pre-trained object detection model
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite/saved_model')

# Load the label map for the model
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
}
# Function to run object detection
def detect_objects(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    
    detections = model(input_tensor)
    
    return detections

# Function to display detection results
def visualize_detection(image_np, detections):
    image_np_with_detections = image_np.copy()

    for i in range(int(detections['num_detections'])):
        score = detections['detection_scores'][i]
        if score > 0.5:  # Threshold to filter out weak detections
            bbox = detections['detection_boxes'][i]
            class_id = int(detections['detection_classes'][i])
            
            # Draw bounding boxes and label on the image
            (h, w, _) = image_np.shape
            (ymin, xmin, ymax, xmax) = bbox
            xmin, xmax, ymin, ymax = int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h)
            
            image_np_with_detections = cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            image_np_with_detections = cv2.putText(image_np_with_detections, category_index[class_id]['name'], 
                                                   (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return image_np_with_detections


# Load an image
image_path = 'image.png'
image_np = cv2.imread(image_path)

# Convert BGR (OpenCV format) to RGB (required for TensorFlow)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Run object detection
detections = detect_objects(image_np)

# Visualize the detection results
image_np_with_detections = visualize_detection(image_np, detections)

# Display the image
plt.figure(figsize=(10,10))
plt.imshow(image_np_with_detections)
plt.show()
