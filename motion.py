#!/usr/bin/env python3

import anki_vector
import time
from anki_vector.util import degrees, distance_mm, speed_mmps


def main():
    args = anki_vector.util.parse_command_args()
    with anki_vector.Robot(args.serial, show_3d_viewer=False, show_viewer=False) as robot:
        print(robot.pose.position)
        robot.behavior.drive_off_charger()
        print(robot.pose.position)

        for i in range(4):
            print(i)
            print("Drive Vector straight...")
            robot.behavior.drive_straight(distance_mm(50), speed_mmps(50))

            print("Turn Vector in place...")
            robot.behavior.turn_in_place(degrees(90))

        robot.behavior.drive_on_charger()


# Driving by coordinates.
# Detect single colour object

if __name__ == "__main__":
    main()
