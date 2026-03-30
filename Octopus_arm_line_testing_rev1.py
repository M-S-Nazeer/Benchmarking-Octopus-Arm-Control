import numpy as np, torch
from project_utilities import (create_3d_straight_line, Octopus_tentacle_actuation,
                               normalize_it, un_normalize_it, VAE_model, prepare_overlapping_sequences)
from scipy.io import loadmat, savemat
from project_utilities import extract_aurora_data
from sksurgerynditracker.nditracker import NDITracker
from time import sleep
import matplotlib.pyplot as plt, time


data_directory = "Data_collection_rev1/"
chosen_shape = "spherical"
model_freq = 1

data_file = data_directory + "positions_" + chosen_shape + "_freq" + str(model_freq) + ".mat"
overall_data = loadmat(data_file)
robot_positions_un = np.squeeze(overall_data[chosen_shape + "pos"])

model_directory = data_directory + chosen_shape + "_model/"
file_name = model_directory + "normalization_parameters_dict_" + str(model_freq) + "Hz.mat"
loaded_matrix = loadmat(file_name)
total_p_min = np.squeeze(loaded_matrix['positions_min'])
total_p_max = np.squeeze(loaded_matrix['positions_max'])
total_act_min = np.squeeze(loaded_matrix['actuations_min'])
total_act_max = np.squeeze(loaded_matrix['actuations_max'])
scaling_factor = np.squeeze(loaded_matrix['scaling_factor'])


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
octopus_operation.reset_dxl = np.asarray([2200, 3900, 2100], dtype=int)
octopus_operation.reset_dxl_positions()
print('Robot class ready...')

robot_frequency = model_freq
motor_frequency = robot_frequency  * 5      # Hz
custom_velocity = np.array(motor_frequency/VEL_UNIT*np.ones(N_motors), dtype='int32')
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
for _ in range(20):
    init_frame = tracker.get_frame()
    position = extract_aurora_data(init_frame)
    sleep(0.02)

position = np.squeeze(position)
print("Robot initial position is: ", position)
position_n = normalize_it(input_data=position, total_p_max=total_p_max, total_p_min=total_p_min, scaling_factor=scaling_factor)
position_n = np.squeeze(position_n)
# --------------------------------------------------------------------------------- #
# ----------------------------- Create the 3D line -------------------------------- #
# --------------------------------------------------------------------------------- #
# Example usage

# step_size = 5
# line = create_3d_straight_line(axis='x', min_val=-50.0, max_val=100.0, step=step_size, const_vals=(position[1], -275.0))
test_shape = "testhelix"
line_ = loadmat(data_directory + test_shape + ".mat")
line = line_[test_shape]
new_data = data_directory + chosen_shape + "_model/" + test_shape + "_freq" + str(robot_frequency) + ".mat"
print("shape of the line: ", line.shape)
line = line[::-1]
FIGSIZE = (8, 7)
fig1 = plt.figure(figsize=FIGSIZE)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], c=robot_positions_un[:, 2], cmap='viridis', marker='o', s=1)
ax1.scatter(line[:, 0], line[:, 1], line[:, 2], c='r', marker='o', s=20)
ax1.plot3D(line[:, 0], line[:, 1], line[:, 2], 'r.-')
ax1.scatter(line[0, 0], line[0, 1], line[0, 2], c='m', s=200, marker="o")
ax1.scatter(line[-1, 0], line[-1, 1], line[-1, 2], c='m', s=200, marker="*")
ax1.set_xlabel('X - axis')
ax1.set_ylabel('Y - axis')
ax1.set_zlabel('Z - axis')
plt.show()


normalized_line = normalize_it(input_data=line, total_p_min=total_p_min, total_p_max=total_p_max, scaling_factor=scaling_factor)


# --------------------------------------------------------------------------------- #
# ------------------------------- Load the model ---------------------------------- #
# --------------------------------------------------------------------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS = 2
SEQ_LEN = 1
STEP_SIZE = 1
LATENT_DIM = 20
HIDDEN_DIM = 128

INPUT_DIM = 3
OUTPUT_DIM = 3

file_w_weights = model_directory + "INV_VAE_model_weights_" + str(model_freq) + "Hz.pth"
VAE_trained_model = VAE_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                              hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
                              seq_len=SEQ_LEN, num_layers=NUM_LAYERS).to(device)
VAE_trained_model.load_state_dict(torch.load(file_w_weights))
VAE_trained_model.eval()

past_past_inst_pos = position_n.copy()
past_inst_pos = position_n.copy()
current_inst_pos = position_n.copy()
next_inst_pos = normalized_line[0, :]

init_motor_position = octopus_operation.read_dxl_position()
current_inst_tau = normalize_it(init_motor_position, total_p_min=total_act_min,
                                total_p_max=total_act_max, scaling_factor=scaling_factor)
time_sample = 1 / robot_frequency

actual_positions, actual_motors = [], []
actual_positions.append(position)
actual_motors.append(init_motor_position)
time_record = time.time()
for l_n in range(1, normalized_line.shape[0]):
    # input_features_CL = np.hstack((past_inst_pos, current_inst_pos, next_inst_pos))
    input_seq = torch.tensor(np.array(next_inst_pos.reshape(1,1,INPUT_DIM)), dtype=torch.float32).to(device)

    with torch.no_grad():
        testing_pred_seq, predicted_z_mean, predicted_z_var = VAE_trained_model(input_seq)

    predicted_action_n = np.squeeze(testing_pred_seq.cpu().numpy())
    predicted_action = un_normalize_it(input_data=predicted_action_n, total_p_max=total_act_max,
                                       total_p_min=total_act_min, scaling_factor=scaling_factor)

    predicted_action = predicted_action.astype(int)
    predicted_action = np.squeeze(predicted_action)
    octopus_operation.write_dxl_goal_velocity(vel=custom_velocity, pos=predicted_action)

    time_to_sleep = time_sample - (time.time() - time_record)
    if np.sign(time_to_sleep) != -1:
        sleep(time_to_sleep)

    time_record = time.time()
    frame_instant = tracker.get_frame()
    position = extract_aurora_data(frame_instant)
    position_n = normalize_it(input_data=position, total_p_max=total_p_max,
                              total_p_min=total_p_min, scaling_factor=scaling_factor)
    position_n = np.squeeze(position_n)
    read_motor = octopus_operation.read_dxl_position()

    position = np.squeeze(position)
    actual_positions.append(position)
    actual_motors.append(read_motor)

    past_past_inst_pos = past_inst_pos.copy()
    past_inst_pos = current_inst_pos.copy()
    current_inst_pos = position_n.copy()
    next_inst_pos = normalized_line[l_n, :]

actual_positions = np.array(actual_positions)
actual_motors = np.array(actual_motors)

# Here reset the robot
octopus_operation.reset_dxl_positions()

# SAve the data
collected_data_dict = {"actual_positions": actual_positions, "actual_motors": actual_motors, "desired_line": line}
savemat(file_name=new_data, mdict=collected_data_dict)
print("File is saved. Going to sleep now...")
sleep(2.0)

# close the tracker now
tracker.stop_tracking()
tracker.close()
print("Tracking device is successfully closed...")

octopus_operation.reset_dxl_positions()
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

MAE_data = []
for i in range(actual_positions.shape[0]):
    MAE_data.append([np.around(np.linalg.norm(np.squeeze(actual_positions[i, :]) - line[i, :]), 3)])

MAE_data = np.array(MAE_data)
plt.figure()
plt.plot(MAE_data, 'ro-')
plt.show()

print("Successful script...")

