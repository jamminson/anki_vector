#!/usr/bin/env python3

import asyncio
import PIL.Image
import PIL.ImageFont
import PIL.ImageTk
import cv2 as cv
import numpy as np
import tkinter as tk
import sys
import anki_vector
import time
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

        while True:

            image = robot.camera.capture_single_image()
            print(f"Displaying image with id {image.image_id}, received at {image.image_recv_time}")
            # image.raw_image.show()
            raw_image = image.raw_image
            raw_rgb = np.array(raw_image)
            raw_bgr = cv.cvtColor(raw_rgb, cv.COLOR_RGB2BGR)
            hsv_img = cv.cvtColor(raw_bgr, cv.COLOR_BGR2HSV)
            cv.imshow("hsv_image", hsv_img)
            cv.imshow("raw_image", raw_bgr)
            threshold_image = cv.inRange(hsv_img, (80, 50, 50), (110, 255, 255))
            cv.imshow("threshold image", threshold_image)
            print("1 iteration")
            cv.waitKey(1)


            # try:
            #     while True:
            #         time.sleep(0.1)
            # except KeyboardInterrupt:
            #     pass


if __name__ == "__main__":
    main()
