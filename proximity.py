import anki_vector
from anki_vector import audio
import time

with anki_vector.Robot() as robot:
    proximity_data = robot.proximity.last_sensor_reading
    if proximity_data is not None:
        print('Proximity distance: {}'.format(proximity_data.distance))

    robot.audio.set_master_volume(audio.RobotVolumeLevel.LOW)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
