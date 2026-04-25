import numpy as np
from project_utilities import (Octopus_tentacle_actuation,
                               normalize_it, un_normalize_it)

import tensorflow as tf
from tensorflow.keras import layers
# import keras_tuner
from tensorflow.keras import Model, initializers
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.python.platform import gfile

from scipy.io import loadmat, savemat
from project_utilities import extract_aurora_data
from sksurgerynditracker.nditracker import NDITracker
from time import sleep
import matplotlib.pyplot as plt, time


data_directory = "Data_collection_rev1/"
chosen_shape = "random_robot2"
model_freq = 2

# data_file_pos = data_directory + "dataset_" + chosen_shape + "_positions_" + str(model_freq) + "Hz.mat"
data_file_pos = data_directory + "positions_" + chosen_shape + "_freq" + str(model_freq) + ".mat"
overall_data_pos = loadmat(data_file_pos)
# robot_positions_un_ = np.squeeze(overall_data_pos["dataset_" + chosen_shape + "_positions"])
robot_positions_un_ = np.squeeze(overall_data_pos["collected_positions"])

nan_indices = np.any(np.isnan(robot_positions_un_), axis=1)
print("Found nan indices: ", nan_indices)

robot_positions_un = robot_positions_un_[~nan_indices]

model_directory = data_directory + chosen_shape + "_model/"
file_name = model_directory + "normalization_parameters_dict_" + str(model_freq) + "Hz.mat"
loaded_matrix = loadmat(file_name)
total_p_min = np.squeeze(loaded_matrix['positions_min'])
total_p_max = np.squeeze(loaded_matrix['positions_max'])
total_act_min = np.squeeze(loaded_matrix['actuations_min'])
total_act_max = np.squeeze(loaded_matrix['actuations_max'])
scaling_factor = np.squeeze(loaded_matrix['scaling_factor'])
print("Scaling factor: ", scaling_factor)
print("total p max: ", total_p_max)
print("total p min: ", total_p_min)

robot_positions_n = normalize_it(input_data=robot_positions_un, total_p_max=total_p_max, total_p_min=total_p_min, scaling_factor=scaling_factor)

# --------------------------------------------------------------------------------- #
# ----------------------------- Octopus class start ------------------------------- #
# --------------------------------------------------------------------------------- #
dynamixel_port = '/dev/ttyUSB0'
print("Connected COM ports: " + dynamixel_port)
N_motors = 3
VEL_UNIT = 0.229
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
print("Maximum allowed tendons: ", max_array)

robot_frequency = 2
motor_frequency = robot_frequency * 10      # Hz
custom_velocity = np.array(motor_frequency/VEL_UNIT*np.ones(N_motors), dtype='int')
print("Desired velocity: ", custom_velocity)


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
for _ in range(40):
    init_frame = tracker.get_frame()
    position = extract_aurora_data(init_frame)
    sleep(0.02)

position = np.squeeze(position)
position = position[:3]
print("Robot initial position is: ", position)
position_n = normalize_it(input_data=position, total_p_max=total_p_max, total_p_min=total_p_min, scaling_factor=scaling_factor)
position_n = np.squeeze(position_n)
print("positions norm1: ", position_n)
breakpoint()
# --------------------------------------------------------------------------------- #
# ----------------------------- Create the 3D line -------------------------------- #
# --------------------------------------------------------------------------------- #
# Example usage
Baseline_variants = "mainPolicy"    # ["B1", "B2", "B3", "mainPolicy"]
if Baseline_variants == "B1":
    Baseline_variant = 7
elif Baseline_variants == "B2":
    Baseline_variant = 1
elif Baseline_variants == "B3":
    Baseline_variant = 5
elif Baseline_variants == "mainPolicy":
    Baseline_variant = 8
else:
    print("Invalid Model Baseline")
    exit()

test_shape = "test_circle"
line_ = loadmat(data_directory + test_shape + ".mat")
line = line_[test_shape]
print("shape of the line: ", line.shape)

FIGSIZE = (8, 7)
fig1 = plt.figure(figsize=FIGSIZE)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], c=robot_positions_un[:, 2], cmap='viridis', marker='o', s=1)
ax1.scatter(line[:, 0], line[:, 1], line[:, 2], c='r', marker='*', s=100)
ax1.scatter(position[0], position[1], position[2], marker='o', c='m', s=400)
ax1.set_xlabel('X - axis')
ax1.set_ylabel('Y - axis')
ax1.set_zlabel('Z - axis')

normalized_line = normalize_it(input_data=line, total_p_min=total_p_min, total_p_max=total_p_max, scaling_factor=scaling_factor)

fig2 = plt.figure(figsize=FIGSIZE)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(robot_positions_n[:, 0], robot_positions_n[:, 1], robot_positions_n[:, 2], c=robot_positions_n[:, 2], cmap='viridis', marker='o', s=1)
ax2.scatter(normalized_line[:, 0], normalized_line[:, 1], normalized_line[:, 2], c='r', marker='*', s=100)
ax2.scatter(position_n[0], position_n[1], position_n[2], marker='o', c='m', s=400)
ax2.set_xlabel('X - axis')
ax2.set_ylabel('Y - axis')
ax2.set_zlabel('Z - axis')
plt.show()
# --------------------------------------------------------------------------------- #
# ------------------------------- Load the model ---------------------------------- #
# --------------------------------------------------------------------------------- #
# file_model = "INV_LSTM_model"
# ANN_file_name = model_directory + "INV_LSTM_model_weights_" + str(model_freq) + "Hz.keras"
# loaded_model = tf.keras.models.load_model(ANN_file_name)

# Load the frozen trained dynamics model for faster predictions
with gfile.GFile(model_directory + "IDM_dir_" + str(model_freq) + "Hz/INV_LSTM_model_rev" + str(Baseline_variant) + ".pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read(-1))

with tf.Graph().as_default() as graph1:
    tf.import_graph_def(graph_def)

sess1 = tf.compat.v1.Session(graph=graph1)
IDM_op_tensor = sess1.graph.get_tensor_by_name('import/Identity:0')

time_sample = 1 / robot_frequency
for trial_n in range(1, 4):
    new_data = (data_directory + chosen_shape + "_model/" + test_shape + "_freq" + str(robot_frequency) + "_rev" +
                str(Baseline_variant) + "_trial" + str(trial_n) + ".mat")
    actual_positions, actual_motors, desired_positions = [], [], []
    for l_n in range(0, 20):
        init_motor_position = octopus_operation.read_dxl_position()
        past_inst_tau = normalize_it(init_motor_position, total_p_min=total_act_min,
                                     total_p_max=total_act_max, scaling_factor=scaling_factor)
        past_inst_tau = np.squeeze(past_inst_tau)
        past_past_inst_tau = past_inst_tau.copy()
        past_past_past_inst_tau = past_inst_tau.copy()

        past_past_inst_pos = position_n.copy()
        past_inst_pos = position_n.copy()
        current_inst_pos = position_n.copy()
        next_inst_pos = normalized_line[l_n, :]

        record_position_un = position.copy()
        current_distance = None
        time_record = time.time()
        for itr_n in range(60):
            if Baseline_variant == 7:  # ---> B^(1)
                tau_pos_inp = (np.hstack((past_inst_pos, current_inst_pos, next_inst_pos))).reshape(1, 1, 9)
            elif Baseline_variant == 1:  # ---> B^(2)
                tau_pos_inp = np.stack((past_inst_tau, past_inst_pos, current_inst_pos, next_inst_pos)).reshape(1, 1, 12)
            elif Baseline_variant == 5:  # ---> B^(3)
                tau_pos_inp = (np.hstack((past_inst_tau, next_inst_pos))).reshape(1, 1, 6)
            elif Baseline_variant == 8:  # ---> Main Policy Results
                tau_pos_inp = (np.hstack((past_past_past_inst_tau, past_past_inst_tau, past_inst_tau, next_inst_pos))).reshape(1, 1, 12)
            else:
                print("Baseline_variant not implemented.....")
                exit()

            pred_temp = sess1.run(IDM_op_tensor, {'import/x:0': tau_pos_inp})
            predicted_action_n = np.squeeze((pred_temp).reshape(1, 3))
            # print("Action norm: ", predicted_action_n)
            predicted_action = un_normalize_it(input_data=predicted_action_n, total_p_max=total_act_max,
                                               total_p_min=total_act_min, scaling_factor=scaling_factor)
            predicted_action = predicted_action.astype(int)
            predicted_action = np.squeeze(predicted_action)
            # print("Action: ", predicted_action, "for state: ", next_inst_pos)
            if np.isnan(predicted_action).any():
                print("NaN was seen here: ", predicted_action)
                break
            else:
                if np.any(predicted_action > max_array):
                    print("unsafe action was sent above the max limit: ", predicted_action)
                    print("-------------------- ROBOT NOT ACTUATED -----------------------")
                else:
                    octopus_operation.write_dxl_goal_velocity(vel=custom_velocity, pos=predicted_action)

            time_to_sleep = time_sample - (time.time() - time_record)
            if np.sign(time_to_sleep) != -1:
                sleep(time_to_sleep)

            time_record = time.time()
            frame_instant = tracker.get_frame()
            position = extract_aurora_data(frame_instant)
            position = np.squeeze(position[:3])
            current_distance = np.around(np.linalg.norm(position - line[l_n, :]), 3)

            print("Point ", l_n, ", distance=", current_distance)
            if np.isnan(current_distance):
                print("Tracked position NaN: ", position, ", retaining previous position...")
                # break
                position = record_position_un.copy()
            position_n = normalize_it(input_data=position, total_p_max=total_p_max,
                                      total_p_min=total_p_min, scaling_factor=scaling_factor)
            position_n = np.squeeze(position_n)
            # print("Position norm2: ", position_n)
            read_motor = octopus_operation.read_dxl_position()
            read_motor_n = normalize_it(input_data=read_motor, total_p_max=total_act_max, total_p_min=total_act_min, scaling_factor=scaling_factor)
            read_motor_n = np.squeeze(read_motor_n)

            position = np.squeeze(position)
            actual_positions.append(position)
            actual_motors.append(read_motor)
            desired_positions.append(line[l_n, :])

            past_past_inst_pos = past_inst_pos.copy()
            past_inst_pos = current_inst_pos.copy()
            current_inst_pos = position_n.copy()
            next_inst_pos = normalized_line[l_n, :]
            # unnecessary position only to tackle NaN
            record_position_un = position.copy()

            past_past_past_inst_tau = past_past_inst_tau.copy()
            past_past_inst_tau = past_inst_tau.copy()
            past_inst_tau = read_motor_n.copy()
        print("-------------------- Steps finished --------------------")
        octopus_operation.reset_dxl_positions()
        sleep(3.0)
        # breakpoint()

        if np.isnan(current_distance):
            for _ in range(20):
                init_frame = tracker.get_frame()
                new_position = extract_aurora_data(init_frame)
                sleep(0.02)
            position = np.squeeze(new_position)[:3]
            print("Tracker position after NaN: ", position)
            # breakpoint()

    actual_positions = np.array(actual_positions)
    actual_motors = np.array(actual_motors)

    # SAve the data
    collected_data_dict = {"actual_positions": actual_positions, "actual_motors": actual_motors,
                           "desired_positions": desired_positions}
    savemat(file_name=new_data, mdict=collected_data_dict)
    print("File is saved. Going to sleep now...")

# close the tracker now
tracker.stop_tracking()
tracker.close()
print("Tracking device is successfully closed...")

# octopus_operation.torque_disable()
octopus_operation.close_port()
print('Motors are reset, torque disabled and port closed...')

fig1 = plt.figure(figsize=FIGSIZE)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], c=robot_positions_un[:, 2], cmap='viridis', marker='o', s=1)

ax1.scatter(line[:, 0], line[:, 1], line[:, 2], c='r', marker='o', s=20)
ax1.plot3D(line[:, 0], line[:, 1], line[:, 2], 'r.-')

ax1.scatter(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], c='k', marker='o', s=20)
ax1.plot3D(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 'k.-')
ax1.set_xlabel('X - axis')
ax1.set_ylabel('Y - axis')
ax1.set_zlabel('Z - axis')
plt.show()

print("Successful script...")

