#!/usr/bin/env python
# ----------------------------------------------------------------------------------------------- #
# ------------------- This file will group write the motors synchronously ----------------------- #
# ----------------------------------------------------------------------------------------------- #
import time, os, sys
from time import sleep
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from project_utilities import Octopus_tentacle_actuation
from project_utilities import extract_aurora_data
from sksurgerynditracker.nditracker import NDITracker


# Physical units
POS_UNIT = 0.088                        # (DEG/unit) A rotation of 1 (motor command) is 0.088 deg
PHI_PULLEY = 0.0125                     # (M) pulley radius
VEL_UNIT = 0.229                        # (RPM/unit) A command of 1 in profile velocity is 0.229 rpm
# Conversion of rotation per unit vs length
def unit2length(units):
    lengths = np.zeros(len(units))
    for i in range(0,len(units)-1):
        lengths[i] = np.deg2rad(units[i]*POS_UNIT)*PHI_PULLEY
    return lengths

def length2unit(lengths):
    units = np.zeros(len(lengths))
    for i in range(0,len(lengths)-1):
        units[i] = np.rad2deg(lengths[i]/PHI_PULLEY)/POS_UNIT
    # units = round(units,0)
    return units

# --------------------------------------------------------------------------------- #
# ----------------------------- Octopus class start ------------------------------- #
# --------------------------------------------------------------------------------- #
fig1, ax1 = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
data_directory = "Data_collection_rev1/"
chosen_shape = "random_robot2"
loaded_matrix_ = loadmat(data_directory + "dataset_" + chosen_shape + ".mat")
# loaded_matrix_ = loadmat(data_directory + chosen_shape + ".mat")
for_plot = np.squeeze(loaded_matrix_["random"])
# for_plot = np.squeeze(loaded_matrix_[chosen_shape])
for_plot = for_plot[:4500, :]
for_plot = for_plot.astype(int)
print('dimension of the traj is: ', for_plot.shape)
ax1[0].plot(for_plot[:, 0], '*-')
ax1[1].plot(for_plot[:, 1], '*-')
ax1[2].plot(for_plot[:, 2], '*-')

# breakpoint()

dynamixel_port = '/dev/ttyUSB0'
print("Connected COM ports: " + dynamixel_port)
N_motors = 3
octopus_operation = Octopus_tentacle_actuation(N_MOTORS=N_motors, DEVICENAME=dynamixel_port)

# Port is opened in the initialization of the class instant
# Baudrate is set in the initialization of the class instant
octopus_operation.torque_enable()

# now set desired initial positions for the motors
desired_init_motor_pos = np.asarray([1800, 400, 2700], dtype=int)
octopus_operation.reset_dxl = desired_init_motor_pos
octopus_operation.reset_dxl_positions()
print('Robot class ready...')

max_array = desired_init_motor_pos + np.asarray([3300, 3300, 3300], dtype=int)
ax1[0].axhline(y=desired_init_motor_pos[0], linestyle=':', color='blue')
ax1[0].axhline(y=max_array[0], linestyle=':', color='blue')

ax1[1].axhline(y=desired_init_motor_pos[1], linestyle=':', color='blue')
ax1[1].axhline(y=max_array[1], linestyle=':', color='blue')

ax1[2].axhline(y=desired_init_motor_pos[2], linestyle=':', color='blue')
ax1[2].axhline(y=max_array[2], linestyle=':', color='blue')
print("Maximum allowed tendons: ", max_array)
plt.show()
# --------------------------------------------------------------------------------- #
# ----------------------------- Aurora class starts ------------------------------- #
# --------------------------------------------------------------------------------- #
aurora_port = '/dev/ttyUSB1'
settings_aurora = {
    "tracker type": "aurora",
    "serial port": aurora_port,
    "verbose": True,
}
tracker = NDITracker(settings_aurora)
print("Tracking device is successfully initialized...")

tracker.start_tracking()
print("Tracking is initiated...")

position = None
for _ in range(20):
    init_frame = tracker.get_frame()
    position = extract_aurora_data(init_frame)
    sleep(0.02)

print("Robot initial position is: ", position)
# breakpoint()
# Compute velocity for the motors movement
robot_frequency = 2
motor_frequency = robot_frequency * 10      # Hz
custom_velocity = np.array(motor_frequency/VEL_UNIT*np.ones(N_motors), dtype='int32')
print("Desired velocity: ", custom_velocity)

device_closed = False
time_sample = 1 / robot_frequency
try:
    data_file = data_directory + "positions_" + chosen_shape + "_freq" + str(robot_frequency) + "_OutsideWater.mat"

    collected_positions = []
    collected_motors = []
    time_rec = time.time()
    for iter in range(for_plot.shape[0]):
        motor_values = for_plot[iter, :]
        # print("sent: ", motor_values)
        if np.isnan(motor_values).any():
            print("NaN was seen here: ", motor_values)
            break
        else:
            if np.any(motor_values > max_array):
                print("unsafe action was sent above the max limit: ", motor_values)
                print("-------------------- ROBOT NOT ACTUATED -----------------------")
            else:
                octopus_operation.write_dxl_goal_velocity(vel=custom_velocity, pos=motor_values)

        time_to_sleep = time_sample - (time.time() - time_rec)
        if np.sign(time_to_sleep) != -1:
            sleep(time_to_sleep)
            if iter == for_plot.shape[0] - 1:
                print("instant", iter, ", acquired position: ", position, ", frequency: ",
                      round(1 / (time.time() - time_rec)), 4)
        time_rec = time.time()
        frame_instant = tracker.get_frame()
        position = extract_aurora_data(frame_instant)
        read_motor = octopus_operation.read_dxl_position()

        collected_positions.append(position)
        collected_motors.append(read_motor)

    collected_positions = np.array(collected_positions)
    collected_motors = np.array(collected_motors)

    # Here reset the robot
    octopus_operation.reset_dxl_positions()

    collected_data_dict = {"collected_positions": collected_positions, "read_motors": collected_motors}
    savemat(file_name=data_file, mdict=collected_data_dict)
    print("File is saved. Going to sleep now...")
    sleep(5.0)

    # close the tracker now
    tracker.stop_tracking()
    tracker.close()
    print("Tracking device is successfully closed...")

    octopus_operation.reset_dxl_positions()
    # octopus_operation.torque_disable()
    octopus_operation.close_port()
    print('Motors are reset, torque disabled and port closed...')

    device_closed = True

finally:
    if not device_closed:
        tracker.stop_tracking()
        tracker.close()
        print("Tracking device is closed afterwards...")

        octopus_operation.reset_dxl_positions()
        # octopus_operation.torque_disable()
        octopus_operation.close_port()
        print('Motors are reset, torque disabled and port closed afterwards...')


print("Successful script...")





