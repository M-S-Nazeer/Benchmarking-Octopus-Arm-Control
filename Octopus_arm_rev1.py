#!/usr/bin/env python
# ----------------------------------------------------------------------------------------------- #
# ------------------- This file will group write the motors synchronously ----------------------- #
# ----------------------------------------------------------------------------------------------- #
from itertools import count
import time, os, sys
from time import sleep
import serial.tools.list_ports
import serial
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from project_utilities import get_motor_saturation
from project_utilities import Octopus_tentacle_actuation

import matlab.engine
import matlab


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



# ---------------- CREATE CLASS INSTANT AND USE FOR READING AND WRITING TO DYNAMIXELS ---------------- #
# comlist = serial.tools.list_ports.comports()
# connected = []
# for element in comlist:
#     connected.append(element.device)
#     print('device: ', element.device)
# print("All connected devices: ", connected)

# REQUIRED_SAMPLES = 100
# for_plot = get_motor_saturation(NUM_MOTORS=3, SAMPLES_PER_MOTOR=REQUIRED_SAMPLES)
# for_plot = np.array(for_plot, dtype=int)
# loaded_matrix_ = loadmat("dataset.mat")
# for_plot = np.squeeze(loaded_matrix_["store"])
# print('dimension of the input dataset: ', for_plot.shape)

N_TRAJS = 49
# fig1, ax1 = plt.subplots(3, 1, figsize=(15, 8))
# for i in range(1, N_TRAJS):
#     loaded_matrix_ = loadmat("Input_dataset/dataset" + str(i) + ".mat")
#     for_plot = np.squeeze(loaded_matrix_["dataset"])
#     print('dimension of traj: ', i, ' is: ', for_plot.shape)
#     ax1[0].plot(for_plot[:, 0], '*-')
#     ax1[1].plot(for_plot[:, 1], '*-')
#     ax1[2].plot(for_plot[:, 2], '*-')
# plt.show()
# breakpoint()

# --------------------------------------------------------------------------------- #
# ----------------------------- Octopus class start ------------------------------- #
# --------------------------------------------------------------------------------- #
dynamixel_port = '/dev/ttyUSB0'
print("Connected COM ports: " + dynamixel_port)
N_motors = 3
octopus_operation = Octopus_tentacle_actuation(N_MOTORS=N_motors, DEVICENAME=dynamixel_port)

# Port is opened in the initialization of the class instant
# Baudrate is set in the initialization of the class instant
octopus_operation.torque_enable()

# now set desired initial positions for the motors
octopus_operation.reset_dxl = np.asarray([2200, 3900, 2100], dtype=int)
octopus_operation.reset_dxl_positions()
print('Robot class ready...')

# Compute velocity for the motors movement
motor_frequency = 42      # Hz
custom_velocity = np.array(motor_frequency/VEL_UNIT*np.ones(N_motors), dtype='int32')
print("Desired velocity: ", custom_velocity)

# --------------------------------------------------------------------------------- #
# ----------------------------- Aurora class starts ------------------------------- #
# --------------------------------------------------------------------------------- #
lock_file = '/var/lock/LCK..ttyUSB1'
if os.path.exists(lock_file):
    try:
        os.remove(lock_file)
        print(f"Removed stale lock file: {lock_file}")
    except PermissionError:
        print(f"Permission denied when trying to remove lock file: {lock_file}")
        print("Try running the script with sudo or fix permissions.")
        exit(1)

aurora_port = '/dev/ttyUSB1'
eng = matlab.engine.start_matlab()
print("MATLAB engine started...")
matlab_path = '/home/user/MatlabProjects/Matlab_NDI_APIs-master/Tests/'
eng.addpath(matlab_path, nargout=0)

# initialize the device first
device_handle = eng.aurora_initialization(nargout=1)
sleep(2.0)
print("device is successfully initialized with: ", device_handle)

device_closed = False
robot_frequency = 2
time_sample = 1 / robot_frequency
try:
    file_directory = "Frequency_" + str(robot_frequency) + "/"
    for traj in range(1, N_TRAJS):
        loaded_matrix_ = loadmat("Input_dataset/dataset" + str(traj) + ".mat")
        for_plot = np.squeeze(loaded_matrix_["dataset"])
        for_plot = for_plot.astype(int)
        print('dimension of traj: ', traj, ' is: ', for_plot.shape)

        data_file = file_directory + "trajectory" + str(traj) + ".mat"

        collected_positions = []
        collected_motors = []
        time_rec = time.time()
        for iter in range(for_plot.shape[0]):
            motor_values = for_plot[iter, :]
            # print("sent: ", motor_values)
            octopus_operation.write_dxl_goal_velocity(vel=custom_velocity, pos=motor_values)

            time_to_sleep = time_sample - (time.time() - time_rec)
            if np.sign(time_to_sleep) != -1:
                sleep(time_to_sleep)
                # print("slept: ", round(time_to_sleep, 5), ", read motor: ", octopus_operation.read_dxl_position(),
                #       ", and velocity: ", custom_velocity)

            position = eng.get_aurora_data(device_handle, nargout=1)
            read_motor = octopus_operation.read_dxl_position()
            print("instant", iter, ", acquired position: ", position, ", frequency: ", int(1 / (time.time() - time_rec)))

            col_pos = np.squeeze(position)
            collected_positions.append(position)
            collected_motors.append([read_motor])
            time_rec = time.time()
        collected_positions = np.array(collected_positions)
        collected_motors = np.array(collected_motors)

        # Here reset the robot
        octopus_operation.reset_dxl_positions()

        collected_data_dict = {"collected_positions": collected_positions, "read_motors": collected_motors}
        savemat(file_name=data_file, mdict=collected_data_dict)
        print("File is saved. Going to sleep now...")
        sleep(5.0)

    # close the device now
    eng.close_aurora_device(device_handle, nargout=0)
    device_closed = True
    print("Device is successfully closed...")

    octopus_operation.reset_dxl_positions()
    # octopus_operation.torque_disable()
    octopus_operation.close_port()
    print('Motors are reset, torque disabled and port closed...')

finally:
    if not device_closed:
        eng.close_aurora_device(device_handle, nargout=0)
        print("Device is closed afterwards...")

        octopus_operation.reset_dxl_positions()
        # octopus_operation.torque_disable()
        octopus_operation.close_port()
        print('Motors are reset, torque disabled and port closed...')

    if eng:
        print("ending the engine")
        try:
            eng.quit()
            print("engine ended successfully")
        except Exception as e:
            print("Error during exiting: ", e)

eng.quit()





