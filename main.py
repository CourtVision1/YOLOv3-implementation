import cv2 as cv2
import numpy as np
import os


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def make_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(object_names[class_id])
    if label not in ('person', 'car', 'truck', 'bus', 'motorbike', 'bicycle'):
        return
    label_with_confidence = f'{label}: {round(confidence, 4)}'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label_with_confidence, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':

    input_path = 'images'
    output_path = 'output-images'
    file_length = len(os.listdir(input_path))
    file_num = 1

    for filename in os.listdir(input_path):

        image = cv2.imread(f'images/{filename}')

        width = image.shape[1]
        height = image.shape[0]
        scale = 0.00392

        # Puts coco.names into a list
        object_names = open('coco.names').read().strip().split('\n')

        # Colors for bounding boxes
        colors = np.random.uniform(0, 255, size=(len(object_names), 3))

        # Creating network using pretrained weights and config file
        net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        layers = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for layer in layers:
            for detection in layer:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Ignore weak detections
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Applying non-max suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw bounding box
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            make_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        output_name = f'{output_path}/output-image-{file_num}.jpg'
        file_num += 1
        cv2.imwrite(output_name, image)
        cv2.destroyAllWindows()
