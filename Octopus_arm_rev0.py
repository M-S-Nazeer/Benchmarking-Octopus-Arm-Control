#!/usr/bin/env python
# ----------------------------------------------------------------------------------------------- #
# ------------------- This file will group write the motors synchronously ----------------------- #
# ----------------------------------------------------------------------------------------------- #
from itertools import count
import time, os, sys
from time import sleep
import serial.tools.list_ports
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
file_dir = sys.stdin.fileno()
from dynamixel_files import *


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
# Physical units
VEL_UNIT = 0.229                        # (RPM/unit) A command of 1 in profile velocity is 0.229 rpm

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
octopus_operation.reset_dxl = np.asarray([0, 0, 0], dtype=int)
octopus_operation.reset_dxl_positions()
print('Robot class ready...')

# Compute velocity for the motors movement
motor_frequency = 5      # Hz
time_sample = 1 / 50
custom_velocity = np.array(motor_frequency/VEL_UNIT*np.ones(N_motors), dtype='int32')
print("Desired velocity: ", custom_velocity)

time_record = time.time()
for iter in range(for_plot.shape[0]):
    octopus_operation.write_dxl_goal_velocity(vel=custom_velocity, pos=for_plot[iter, :])

    time_to_sleep = time_sample - (time.time() - time_record)
    if np.sign(time_to_sleep) != -1:
        sleep(time_to_sleep)
        print("slept: ", round(time_to_sleep, 5), ", position: ", octopus_operation.read_dxl_position(),
              ", and velocity: ", custom_velocity)

    time_record = time.time()

print('Now resetting the positions of the motors...')
octopus_operation.reset_dxl_positions()
octopus_operation.torque_disable()
octopus_operation.close_port()


