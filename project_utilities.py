import numpy as np, random
import matplotlib.pyplot as plt
import sys
file_dir = sys.stdin.fileno()
from dynamixel_files import *


def normalize_it(input_data, total_p_min, total_p_max, scaling_factor):
    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_array[0, ind] = 1 + ((2 / (total_p_max[ind] - total_p_min[ind])) * (input_data[ind] - total_p_max[ind]))
            input_array[0, ind] = input_array[0, ind] * scaling_factor

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            input_array[:, ind] = 1 + ((2 / (total_p_max[ind] - total_p_min[ind])) * (input_data[:, ind] - total_p_max[ind]))
            input_array[:, ind] = input_array[:, ind] * scaling_factor

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array


def un_normalize_it(input_data, total_p_min, total_p_max, scaling_factor):
    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_data[ind] = input_data[ind] / scaling_factor
            input_array[0, ind] = total_p_max[ind] + ((input_data[ind] - 1) * (total_p_max[ind] - total_p_min[ind]) / 2)

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            input_data[:, ind] = input_data[:, ind] / scaling_factor
            input_array[:, ind] = total_p_max[ind] + ((input_data[:, ind] - 1) * (total_p_max[ind] - total_p_min[ind]) / 2)

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array


def inverse_time_series_conversion(Positions, Actions, Baseline_variant):
    input_features, output_features = [], []
    next_inst_pos = Positions[2, :]
    current_inst_pos = Positions[1, :]
    past_inst_pos = Positions[0, :]
    past_past_inst_pos = Positions[0, :]

    current_inst_tau = Actions[1, :]
    past_inst_tau = Actions[0, :]
    past_past_inst_tau = Actions[0, :]
    past_past_past_inst_tau = Actions[0, :]
    for i in range(2, Positions.shape[0] - 1):
        if Baseline_variant == "B1":   # ---> B^(1)
            temp_inp_features = np.hstack((past_inst_pos, current_inst_pos, next_inst_pos))
            multiplier = 3
        elif Baseline_variant == "B2":   # ---> B^(2)
            temp_inp_features = np.hstack((past_inst_tau, past_inst_pos, current_inst_pos, next_inst_pos))
            multiplier = 4
        elif Baseline_variant == "B3":   # ---> B^(3)
            temp_inp_features = np.hstack((past_inst_tau, next_inst_pos))
            multiplier = 2
        elif Baseline_variant == "mainPolicy":   # ---> Main Policy Results
            temp_inp_features = np.hstack((past_past_past_inst_tau, past_past_inst_tau, past_inst_tau, next_inst_pos))
            multiplier = 4
        else:
            print("case undefined....")
            multiplier = None
            break
        input_features.append([temp_inp_features])  # inputs
        output_features.append([current_inst_tau])  # to predict

        past_past_past_inst_tau = past_past_inst_tau.copy()
        past_past_inst_tau = past_inst_tau.copy()
        past_inst_tau = Actions[i - 1, :]
        current_inst_tau = Actions[i, :]

        past_past_inst_pos = Positions[i - 2, :]
        past_inst_pos = Positions[i - 1, :]
        current_inst_pos = Positions[i, :]
        next_inst_pos = Positions[i + 1, :]

    input_features = (np.array(input_features)).reshape(len(input_features), Positions.shape[1] * multiplier)
    output_features = (np.array(output_features)).reshape(len(output_features), Actions.shape[1])
    return input_features, output_features




class Octopus_tentacle_actuation:
    def __init__(self, N_MOTORS, DEVICENAME):
        # ********* DYNAMIXEL Model definition *********
        # Addresses on the control table of the Dynamixel, it is specific to the series we are using
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_MAX_POS_LIM = 48
        self.ADDR_MIN_POS_LIM = 52
        self.ADDR_PRO_TORQUE_ENABLE = 64
        self.ADDR_ERROR_STATUS = 70
        self.ADDR_PRO_PROFILE_VELOCITY = 112
        self.ADDR_PRO_GOAL_POSITION = 116
        self.ADDR_PRO_PRESENT_CURRENT = 126
        self.ADDR_PRO_PRESENT_POSITION = 132

        # Data Byte Length, coming from the same table
        self.LEN_ERROR_STATUS = 1
        self.LEN_MAX_POS_LIM = 4
        self.LEN_MIN_POS_LIM = 4
        self.LEN_PRO_GOAL_POSITION = 4
        self.LEN_PRO_PRESENT_POSITION = 4
        self.LEN_PRO_PRESENT_CURRENT = 2
        self.LEN_PRO_VEL_AND_POS = 8

        # Protocol version
        self.PROTOCOL_VERSION = 2.0

        # Default setting
        self.BAUDRATE = 115200  # 57600
        self.DEVICENAME = DEVICENAME  # Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.EXT_POSITION_CONTROL_MODE = 4  # Value for extended position control mode (operating mode)
        self.DXL_MOVING_STATUS_THRESHOLD = 20

        self.MIN_BOUND = -1048575
        self.MAX_BOUND = 1048575

        self.N_MOTORS = N_MOTORS
        # Make sure that each DYNAMIXEL ID should have unique ID - List of IDs assigned ot motors
        self.DXL_IDs = [i + 1 for i in range(self.N_MOTORS)]
        self.reset_dxl = np.asarray([1339, 836, 3198], dtype=int)

        # Initialize PortHandler instance
        self.portHandler = PortHandler(self.DEVICENAME)
        # Initialize PacketHandler instance
        self.packetHandler = Protocol2PacketHandler()
        # Initialize groupsyncwrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_PROFILE_VELOCITY,
                                             self.LEN_PRO_VEL_AND_POS)
        # Initialize groupsyncread instance
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_POSITION,
                                           self.LEN_PRO_PRESENT_POSITION)
        # Initialize groupsyncread instance for feedback
        self.groupSyncReadCurrent = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_CURRENT,
                                                  self.LEN_PRO_PRESENT_CURRENT)

        # ########################### Open port ########################### #
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # ########################### set baud rate ########################### #
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            print("Failed to change the baudrate")
            quit()

        # ########################### set operating mode ########################### #
        for id in range(len(self.DXL_IDs)):
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_IDs[id],
                                                                           self.ADDR_OPERATING_MODE,
                                                                           self.EXT_POSITION_CONTROL_MODE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        self.MINPOS, self.MAXPOS = self.get_bounds()

        # ########################### Add parameter storage for present position value ########################### #
        for i in self.DXL_IDs:
            dxl_addparam_result = self.groupSyncRead.addParam(i)
            dxl_addparam_result_cur = self.groupSyncReadCurrent.addParam(i)
            if dxl_addparam_result != True or dxl_addparam_result_cur != True:
                print("[ID:%03d] groupSyncRead addparam failed" % i)
                quit()

    def __del__(self):
        # Destructor: clear syncread parameter storage, disable torque and close port
        # Clear syncread parameter storage
        self.groupSyncRead.clearParam()
        self.torque_disable()
        # Close port
        self.portHandler.closePort()

    def write_dxl_goal_velocity(self, vel, pos):
        # Allocate velocity and position for all motors with relative Id specified in relId
        # handles scalar case
        if type(self.DXL_IDs) == np.int32:
            ids = np.array([self.DXL_IDs], dtype="int")
            vel = np.array([vel], dtype="int")
            pos = np.array([pos], dtype="int")
        else:
            ids = self.DXL_IDs.copy()
        # if goal position command is outside of the limits, go to the limit
        for i in range(len(self.DXL_IDs)):
            if pos[i] > self.MAX_BOUND:
                pos[i] = self.MAX_BOUND
            if pos[i] < self.MIN_BOUND:
                pos[i] = self.MIN_BOUND

            if pos[i] < 0:
                pos[i] += 4294967296

            # Allocate goal position value into byte array
            param_goal = [DXL_LOBYTE(DXL_LOWORD(vel[i])), DXL_HIBYTE(DXL_LOWORD(vel[i])),
                          DXL_LOBYTE(DXL_HIWORD(vel[i])), DXL_HIBYTE(DXL_HIWORD(vel[i])),
                          DXL_LOBYTE(DXL_LOWORD(pos[i])), DXL_HIBYTE(DXL_LOWORD(pos[i])),
                          DXL_LOBYTE(DXL_HIWORD(pos[i])), DXL_HIBYTE(DXL_HIWORD(pos[i]))]
            # Add goal value to the Syncwrite parameter storage
            dxl_addparam_result = self.groupSyncWrite.addParam(ids[i], param_goal)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % ids[i])
                quit()

        # Send packets in groupSyncWrite and clears it afterward
        dxl_comm_result = self.groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # Clear syncwrite parameter storage
        self.groupSyncWrite.clearParam()

    def reset_dxl_positions(self):
        pos_temp = self.read_dxl_position()
        for i in range(len(self.DXL_IDs)):
            # First, bring the motor close to reset pose slowly
            while not (np.isclose(pos_temp[i], self.reset_dxl[i], atol=150)):
                pos_temp = self.read_dxl_position()
                if pos_temp[i] > self.reset_dxl[i]:
                    dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                        self.portHandler, self.DXL_IDs[i], self.ADDR_PRO_GOAL_POSITION, pos_temp[i] - 150
                    )
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                elif pos_temp[i] <self.reset_dxl[i]:
                    dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                        self.portHandler, self.DXL_IDs[i], self.ADDR_PRO_GOAL_POSITION, pos_temp[i] + 150
                    )
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % self.packetHandler.getRxPacketError(dxl_error))

            # Now close the gap to the reset pose fast
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, self.DXL_IDs[i], self.ADDR_PRO_GOAL_POSITION, self.reset_dxl[i]
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def read_dxl_position(self):
        # Uses syncread to get current position of all motors.
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        current_pos = np.zeros(len(self.DXL_IDs), dtype='int')

        for i in range(len(self.DXL_IDs)):
            # Check if groupsyncread data is available
            dxl_getdata_result = self.groupSyncRead.isAvailable(self.DXL_IDs[i], self.ADDR_PRO_PRESENT_POSITION,
                                                                self.LEN_PRO_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.DXL_IDs[i])
                continue
            # Get present position value
            temp_read = self.groupSyncRead.getData(self.DXL_IDs[i], self.ADDR_PRO_PRESENT_POSITION,
                                                   self.LEN_PRO_PRESENT_POSITION)
            if temp_read > self.MAX_BOUND:
                temp_read -= 4294967296
            current_pos[i] = temp_read

        return current_pos

    def read_dxl_torque(self):
        # Uses syncread to get current torque of all motors.
        dxl_comm_result = self.groupSyncReadCurrent.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        current_torque = np.zeros(len(self.DXL_IDs), dtype='int')  # [0 for i in ids]

        for i in range(len(self.DXL_IDs)):
            # Check if groupsyncread data is available
            dxl_getdata_result = self.groupSyncReadCurrent.isAvailable(self.DXL_IDs[i], self.ADDR_PRO_PRESENT_CURRENT,
                                                                       self.LEN_PRO_PRESENT_CURRENT)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.DXL_IDs[i])
                continue
            # Get present position value
            temp_read = self.groupSyncReadCurrent.getData(self.DXL_IDs[i], self.ADDR_PRO_PRESENT_CURRENT,
                                                          self.LEN_PRO_PRESENT_CURRENT)
            if temp_read > 32768:
                temp_read = temp_read - 65536
            current_torque[i] = temp_read

        return current_torque

    def torque_enable(self):
        if type(self.DXL_IDs) == np.int32:
            ids = np.array([self.DXL_IDs], dtype="int")
        else:
            ids = self.DXL_IDs.copy()
        for i in ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, i,
                                                                           self.ADDR_PRO_TORQUE_ENABLE,
                                                                           self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("[ID:%03d] Torque Enabled" % i)

    def torque_disable(self):
        # Disable torque for relId motors. relId is an array with 1,2 or 3 elements
        if type(self.DXL_IDs) == np.int32:
            ids = np.array([self.DXL_IDs], dtype="int")
        else:
            ids = self.DXL_IDs.copy()

        for i in ids:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, i,
                                                                           self.ADDR_PRO_TORQUE_ENABLE,
                                                                           self.TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("[ID:%03d] Torque Disabled" % i)

    def set_bounds(self, minConfLim, maxConfLim):
        # Set conf limits for motor with ids
        # confLim registers are in the EEPROM, so the code first disable torque, then applies the change and finally
        # enable the torque again.
        # Note that if goal position is outside of the limits the motor will not move and you will get a dxl_error
        self.torque_disable()

        for i in range(len(self.DXL_IDs)):
            # Set minConfLim
            if minConfLim[i] < 0:
                minConfLim[i] = ~minConfLim[i]
            print(minConfLim[i])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_IDs[i],
                                                                           self.ADDR_MIN_POS_LIM, minConfLim[i])
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

            # Set maxConfLim
            if maxConfLim[i] < 0:
                maxConfLim[i] = ~maxConfLim[i]
            print(maxConfLim[i])
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_IDs[i],
                                                                           self.ADDR_MAX_POS_LIM, maxConfLim[i])
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        self.MINPOS, self.MAXPOS = self.get_bounds()  # update minpos and maxpos variables
        self.torque_enable()

    def get_bounds(self):
        # Get conf limits for motor with ids
        dxl_min_pos = np.zeros(len(self.DXL_IDs), dtype='int')  # [0 for i in ids]
        dxl_max_pos = np.zeros(len(self.DXL_IDs), dtype='int')  # [0 for i in ids]
        for i in range(len(self.DXL_IDs)):
            # get minPosLim
            dxl_min_pos[i], dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler,
                                                                                          self.DXL_IDs[i],
                                                                                          self.ADDR_MIN_POS_LIM)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

            # get maxPosLim
            dxl_max_pos[i], dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler,
                                                                                          self.DXL_IDs[i],
                                                                                          self.ADDR_MAX_POS_LIM)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_min_pos, dxl_max_pos

    def get_status(self):
        # returns status for all motors with relId
        # if dxl_status[i] is 32, motor i is in overload
        dxl_status = np.zeros(len(self.DXL_IDs), dtype='int')
        for i in range(len(self.DXL_IDs)):
            # get status
            dxl_status[i], dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler,
                                                                                         self.DXL_IDs[i],
                                                                                         self.ADDR_ERROR_STATUS)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        return dxl_status

    def close_port(self):
        # Close port
        self.portHandler.closePort()


def move_motor_to_target(current_position, target_position, min_step, max_step):
    trajectory = [current_position]
    current_pos = current_position

    if current_pos == target_position:
        return trajectory, current_pos # No movement needed

    direction = 1 if target_position > current_pos else -1

    while current_pos != target_position:
        remaining_distance = abs(target_position - current_pos)

        # Determine a random step size
        # Ensure step doesn't overshoot and respects min/max
        possible_step = random.randint(min_step, max_step)
        step = min(possible_step, remaining_distance)

        current_pos += direction * step
        trajectory.append(current_pos)

    return trajectory, current_pos


def get_motor_saturation(NUM_MOTORS, SAMPLES_PER_MOTOR):

    # --- Parameters ---
    MIN_POSITION = -1000
    MAX_POSITION = 1000
    MIN_STEPSIZE = 150
    MAX_STEPSIZE = 200

    # --- Simulation ---
    all_motor_trajectories = []  # List to store trajectories for each motor
    final_positions_log = []  # List to store final positions for each move

    print(f"Simulating {NUM_MOTORS} motors, each moving to {SAMPLES_PER_MOTOR} random target positions.")
    print(f"Position range: [{MIN_POSITION}, {MAX_POSITION}]")
    print(f"Step size range: [{MIN_STEPSIZE}, {MAX_STEPSIZE}]")

    for motor_id in range(NUM_MOTORS):
        print(f"\n--- Motor {motor_id + 1} ---")
        motor_trajectory_full = []
        current_motor_pos = 0  # Start each motor from 0
        motor_trajectory_full.append(current_motor_pos)  # Log initial position

        for sample_num in range(SAMPLES_PER_MOTOR):
            # Choose a new random target position
            target_pos = random.randint(MIN_POSITION, MAX_POSITION)

            # Move the motor
            # print(f"  Sample {sample_num + 1}: Moving from {current_motor_pos} to {target_pos}")
            segment_trajectory, final_pos_segment = move_motor_to_target(
                current_motor_pos,
                target_pos,
                MIN_STEPSIZE,
                MAX_STEPSIZE
            )

            # Append the new segment (excluding the first point as it's the previous end)
            motor_trajectory_full.extend(segment_trajectory[1:])
            current_motor_pos = final_pos_segment  # Update current position for next move
            final_positions_log.append(
                {'motor': motor_id + 1, 'sample': sample_num + 1, 'target': target_pos, 'reached': final_pos_segment})

        all_motor_trajectories.append(motor_trajectory_full)
        print(f"Motor {motor_id + 1} simulation complete. Total points: {len(motor_trajectory_full)}")

    print("\nSimulation finished.")

    # --- Optional: Print some stats or save data ---
    returned_positions = np.zeros((SAMPLES_PER_MOTOR, NUM_MOTORS), dtype=int)
    for i, traj in enumerate(all_motor_trajectories):
        print(f"Motor {i + 1} total trajectory length: {len(traj)} points")
        returned_positions[:, i] = traj[:SAMPLES_PER_MOTOR]

    # --- Plotting (Optional) ---
    # Plot the full trajectory for each motor separately
    plt.figure(figsize=(15, NUM_MOTORS * 4))
    for i in range(NUM_MOTORS):
        plt.subplot(NUM_MOTORS, 1, i + 1)
        if all_motor_trajectories[i]:
            plt.plot(returned_positions[:, i], marker='*', linestyle='-', markersize=10)
            plt.title(f"Motor {i + 1} Full Trajectory")
            plt.xlabel("Step Number (Cumulative)")
            plt.ylabel("Position")
            plt.grid(True)
        else:
            plt.title(f"Motor {i + 1} (No data)")

    plt.tight_layout()
    plt.show()

    return returned_positions


def extract_aurora_data(frame):
    Transformation_matrix = frame[3][0]
    translation_list = np.zeros(3)
    translation_list[0] = Transformation_matrix[0][3]
    translation_list[1] = Transformation_matrix[1][3]
    translation_list[2] = Transformation_matrix[2][3]

    translation_list = np.array(translation_list).reshape(1, 3)
    return translation_list


def extract_aurora_data_m2(frame):

    Transoformation_matrix1 = frame[3][0]
    Transoformation_matrix2 = frame[3][1]

    # print("Whole frame: ", frame)

    translation_list = np.zeros(6)
    # print("Transformer 01: ", Transoformation_matrix1)
    translation_list[0] = Transoformation_matrix1[0][3]
    translation_list[1] = Transoformation_matrix1[1][3]
    translation_list[2] = Transoformation_matrix1[2][3]

    # print("Transforer 02: ", Transoformation_matrix2)
    translation_list[3] = Transoformation_matrix2[0][3]
    translation_list[4] = Transoformation_matrix2[1][3]
    translation_list[5] = Transoformation_matrix2[2][3]

    translation_list = np.squeeze(np.array(translation_list).reshape(1, 6))
    return translation_list


def create_3d_straight_line(axis='x', min_val=0, max_val=10, step=1, const_vals=(0, 0)):
    values = np.arange(min_val, max_val + step, step)

    if axis == 'x':
        x = values
        y = np.full_like(values, const_vals[0])
        z = np.full_like(values, const_vals[1])
    elif axis == 'y':
        x = np.full_like(values, const_vals[0])
        y = values
        z = np.full_like(values, const_vals[1])
    elif axis == 'z':
        x = np.full_like(values, const_vals[0])
        y = np.full_like(values, const_vals[1])
        z = values
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return np.column_stack((x, y, z))



