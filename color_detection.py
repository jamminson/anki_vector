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


args = anki_vector.util.parse_command_args()
with anki_vector.Robot() as robot:
    image = robot.camera.capture_single_image()
    print(f"Displaying image with id {image.image_id}, received at {image.image_recv_time}")
    image.raw_image.show()
    raw_image = image.raw_image
    raw_rgb = np.array(raw_image)
    hsv_img = cv.cvtColor(raw_rgb, cv.COLOR_RGB2HSV)
    cv.imshow("raw_image", hsv_img)
    threshold_image = cv.inRange(hsv_img, (0, 0, 0), (50, 255, 255))
    cv.imshow("threshold image", threshold_image)




