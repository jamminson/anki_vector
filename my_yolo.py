#!/usr/bin/env python3


import numpy as np
import argparse
import imutils
import time
import cv2 as cv
import os
import anki_vector
import time
from timeit import default_timer as timer
from anki_vector import audio
from anki_vector.connection import ControlPriorityLevel


def main():
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial,
                           default_logging=False,
                           show_viewer=False,
                           show_3d_viewer=False,
                           enable_nav_map_feed=False,
                           behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
        robot.audio.set_master_volume(audio.RobotVolumeLevel.LOW)

        # labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        labelsPath = "‎/Users/jamin/desktop/yolo-object-detection/yolo-coco/coco.names"
        labelsPath = labelsPath.strip('\u200e')
        # weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
        weightsPath = "/Users/jamin/desktop/yolo-object-detection/yolo-coco/yolov3.weights"
        # configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
        configPath = "/Users/jamin/desktop/yolo-object-detection/yolo-coco/yolov3.cfg"
        LABELS = open(labelsPath).read().strip().split("\n")

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype="uint8")

        print("[INFO] loading YOLO from disk...")
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        (W, H) = (None, None)

        while True:
            start = timer()
            image = robot.camera.capture_single_image()
            print(f"Displaying image with id {image.image_id}, received at {image.image_recv_time}")
            # image.raw_image.show()
            raw_image = image.raw_image
            raw_rgb = np.array(raw_image)
            raw_bgr = cv.cvtColor(raw_rgb, cv.COLOR_RGB2BGR)
            # hsv_img = cv.cvtColor(raw_bgr, cv.COLOR_BGR2HSV)
            # cv.imshow("hsv_image", hsv_img)
            # cv.imshow("raw_image", raw_bgr)
            # threshold_image = cv.inRange(hsv_img, (80, 50, 50), (110, 255, 255))
            # cv.imshow("threshold image", threshold_image)

            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = raw_bgr.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv.dnn.blobFromImage(raw_bgr, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
            net.setInput(blob)
            start_nn = timer()
            layerOutputs = net.forward(ln)
            end_nn = timer()
            print(end_nn-start_nn)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    confidence_threshold = 0.5
                    if confidence > confidence_threshold:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            nms_threshold = 0.3
            idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                                   nms_threshold)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv.rectangle(raw_bgr, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                    cv.putText(raw_bgr, text, (x, y - 5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.imshow("image_labelled", raw_bgr)
            cv.waitKey(1)
            end = timer()
            print(end-start)


if __name__ == "__main__":
    main()
