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
from anki_vector.util import degrees, radians, distance_mm, speed_mmps, Pose
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
        color_labels = np.random.randint(0, 255, size=(len(LABELS), 3),
                                         dtype="uint8")

        print("[INFO] loading YOLO from disk...")
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        robot.behavior.drive_off_charger()
        print("Robot has driven off charger.")

        while True:
            error_x = 1000000
            error_y = 1000000
            while abs(error_x) > 20 or abs(error_y) > 20:
                raw_bgr = get_BGR_image(robot)
                (center_detection_x, center_detection_y) = get_yolo_detections(raw_bgr, net, ln, color_labels, LABELS)

                (H, W) = grab_image_dimensions(raw_bgr)
                center_image_x = 0.5 * W
                center_image_y = 0.5 * H

                error_x = center_detection_x - center_image_x
                error_y = center_detection_y - center_image_y

                # print("Difference x = {}, Difference y = {}".format(error_x, error_y))
                scaled_error_x = -0.1 * error_x
                scaled_error_y = -0.1 * error_y
                # if error_x > 0:
                robot.behavior.turn_in_place(degrees(scaled_error_x))
                robot.behavior.set_head_angle(degrees(scaled_error_y) + radians(robot.head_angle_rad))
                # elif error_x < 0:

            # print(error_x)
            # print(error_y)
            rand_pose_x = 100 * np.random.rand()
            rand_pose_y = 100 * np.random.rand()
            rand_pose_z = 0
            rand_pose_degrees = 360 * np.random.rand()

            print("This is the proximity distance: {}".format(get_distance_from_proximity_sensor(robot)))

            rand_pose = Pose(x=rand_pose_x, y=rand_pose_y, z=rand_pose_z,
                             angle_z=anki_vector.util.Angle(degrees=rand_pose_degrees))
            print("I am going to pose: {}".format(rand_pose))
            robot.behavior.go_to_pose(rand_pose)

            # robot.behavior.drive_straight(distance_mm(50), speed_mmps(50))
            # robot.behavior.turn_in_place(degrees(90))




def get_BGR_image(robot):
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

    return raw_bgr


def running_nn_on_image(bgr_image, net, ln):
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv.dnn.blobFromImage(bgr_image, 1 / 255.0, (416, 416),
                                swapRB=True, crop=False)
    net.setInput(blob)
    start_nn = timer()
    layerOutputs = net.forward(ln)
    end_nn = timer()
    print(end_nn - start_nn)

    return layerOutputs


def grab_image_dimensions(bgr_image):
    # if the frame dimensions are empty, grab them
    (H, W) = bgr_image.shape[:2]

    return H, W


def get_score_ID_confidence(detection):
    # extract the class ID and confidence (i.e., probability)
    # of the current object detection
    scores = detection[5:]
    classID = np.argmax(scores)
    confidence = scores[classID]

    return scores, classID, confidence


def derive_image_box_bounds(detection, W, H):
    # scale the bounding box coordinates back relative to
    # the size of the image, keeping in mind that YOLO
    # actually returns the center (x, y)-coordinates
    # the bounding box followed by the boxes' width and
    # height
    box = detection[0:4] * np.array([W, H, W, H])
    (centerX, centerY, width, height) = box.astype("int")

    # use the center (x, y)-coordinates to derive the top
    # and and left corner of the bounding box
    x = int(centerX - (width / 2))
    y = int(centerY - (height / 2))

    return x, y, width, height


def update_box_bounds(boxes, x, y, width, height):
    # update our list of bounding box coordinates,
    boxes.append([x, y, int(width), int(height)])


def update_confidences(confidences, confidence):
    # update list of confidences
    confidences.append(float(confidence))


def update_classIDs(classIDs, classID):
    # update list of classIDs
    classIDs.append(classID)


def define_color(COLORS, classIDs, i):
    color = [int(c) for c in COLORS[classIDs[i]]]
    return color


def draw_box(image, x, y, w, h, color):
    # Draws box around the image
    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)


def define_text(LABELS, classIDs, interation_number, confidences):
    text = "{}: {:.4f}".format(LABELS[classIDs[interation_number]],
                               confidences[interation_number])
    return text


def write_text_on_box(image, text, placement, text_font, text_color, ):
    cv.putText(image, text, placement, text_font, 0.5, text_color, 2)


def get_distance_from_proximity_sensor(robot):
    proximity_data = robot.proximity.last_sensor_reading
    if proximity_data is not None:
        return proximity_data.distance


def get_yolo_detections(image, neural_network, network_labels, color_labels, labels):
    start = timer()
    (H, W) = grab_image_dimensions(image)

    layerOutputs = running_nn_on_image(image, neural_network, network_labels)
    # print(layerOutputs)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    confidence_threshold = 0.5

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            (scores, classID, confidence) = get_score_ID_confidence(detection)

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability

            if confidence > confidence_threshold:
                (x, y, width, height) = derive_image_box_bounds(detection, W, H)
                update_box_bounds(boxes, x, y, width, height)
                update_confidences(confidences, confidence)
                update_classIDs(classIDs, classID)

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

            color = define_color(color_labels, classIDs, i)
            draw_box(image, x, y, w, h, color)

            text = define_text(labels, classIDs, i, confidences)
            write_text_on_box(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, color)

    # Show image
    cv.imshow("image_labelled", image)
    cv.waitKey(1)
    end = timer()
    print(end - start)
    if len(idxs) > 0:
        (x, y) = (boxes[0][0], boxes[0][1])
        (w, h) = (boxes[0][2], boxes[0][3])
        (center_x, center_y) = (x + (0.5 * w), y + (0.5 * h))
        return center_x, center_y

    return 0, 0

    # print("Robot is at {}.".format(robot.pose.position))

    # if len(idxs) > 0:
    #     for i in idxs.flatten():
    #         (x, y) = (boxes[i][0], boxes[i][1])
    #         (w, h) = (boxes[i][2], boxes[i][3])
    #         print("Robot has seen {} with confidence {}. {} has pixel location ({},{}).".format(
    #             labels[classIDs[i]],
    #             confidences[i],
    #             labels[classIDs[i]],
    #             x + (0.5 * w),
    #             y + (0.5 * h)))


if __name__ == "__main__":
    main()
