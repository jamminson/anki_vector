#!/usr/bin/env python3

import time

import anki_vector
from anki_vector.objects import CustomObjectMarkers, CustomObjectTypes


def handle_object_appeared(robot, event_type, event):
    # This will be called whenever an EvtObjectAppeared is dispatched -
    # whenever an Object comes into view.
    robot.behavior.set_eye_color(hue=0, saturation=1)
    print(f"--------- Vector started seeing an object --------- \n{event.obj}")


def handle_object_disappeared(robot, event_type, event):
    # This will be called whenever an EvtObjectDisappeared is dispatched -
    # whenever an Object goes out of view.
    robot.behavior.set_eye_color(hue=0.21, saturation=1)
    print(f"--------- Vector stopped seeing an object --------- \n{event.obj}")


def main():
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial,
                           default_logging=False,
                           show_viewer=True,
                           show_3d_viewer=True,
                           enable_nav_map_feed=True) as robot:
        # Add event handlers for whenever Vector sees a new object
        robot.events.subscribe(handle_object_appeared, anki_vector.events.Events.object_appeared)
        robot.events.subscribe(handle_object_disappeared, anki_vector.events.Events.object_disappeared)

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
