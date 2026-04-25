import numpy as np
import matplotlib.pyplot as plt

import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat, savemat

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
# import keras_tuner
from tensorflow.keras import Model, initializers
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from project_utilities import normalize_it, un_normalize_it, inverse_time_series_conversion
from tensorflow.python.platform import gfile


FIGSIZE = (8, 7)
data_directory = "Data_collection_rev1/"
chosen_shape = "randdisturbed"     # circle_compl_rand, spiral_compl_rand, logspiral, logcircle, randdisturbed
frequency = 2
data_file_pos = data_directory + "data_" + chosen_shape + "_positions_" + str(frequency) + "Hz.mat"
# data_file_pos = data_directory + "positions_" + chosen_shape + "_freq" + str(frequency) + "_OutsideWater.mat"
overall_data_pos = loadmat(data_file_pos)
robot_positions_un_ = np.squeeze(overall_data_pos["data_" + chosen_shape + "_positions"])
# robot_positions_un_ = np.squeeze(overall_data_pos["collected_positions"])


data_file_act = data_directory + "data_" + chosen_shape + "_motor_" + str(frequency) + "Hz.mat"
overall_data_acts = loadmat(data_file_act)
tendon_actions_un_ = np.squeeze(overall_data_acts["data_" + chosen_shape + "_motor"])
# tendon_actions_un_ = np.squeeze(overall_data_pos["read_motors"])

loaded_matrix_ = loadmat(data_directory + "data_" + chosen_shape + ".mat")
# print("Matrix keys: ", loaded_matrix_.keys())
# for_plot = np.squeeze(loaded_matrix_["data_" + chosen_shape])
# for_plot = np.squeeze(loaded_matrix_["data_" + chosen_shape + "_compl"])
# loaded_matrix_ = loadmat(data_directory + "dataset_" + chosen_shape + ".mat")
for_plot = np.squeeze(loaded_matrix_["store"])
for_plot = for_plot.astype(int)
for_plot = for_plot[:4200, :]

# robot_positions_un_ = robot_positions_un_[42:, :]
# tendon_actions_un_ = tendon_actions_un_[42:, :]
# for_plot = for_plot[42:, :]

print("Overall Robot positions shape: ", robot_positions_un_.shape)
print("Overall Robot actuations shape: ", tendon_actions_un_.shape)
nan_indices = np.any(np.isnan(robot_positions_un_), axis=1)
print("Found nan indices: ", nan_indices)

robot_positions_un = robot_positions_un_[~nan_indices]
tendon_actions_un = tendon_actions_un_[~nan_indices]

print("New Robot positions shape: ", robot_positions_un.shape)
print("New Robot actuations shape: ", tendon_actions_un.shape)

mesh_x, mesh_y = np.meshgrid(robot_positions_un[:, 0], robot_positions_un[:, 1])

fig1 = plt.figure(figsize=FIGSIZE)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], c=robot_positions_un[:, 2], cmap='viridis', marker='o', s=5)
# ax1.plot_trisurf(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], cmap='viridis')
# ax1.plot3D(robot_positions_un[:, 0], robot_positions_un[:, 1], robot_positions_un[:, 2], 'r.-')
ax1.set_xlabel('X - axis')
ax1.set_ylabel('Y - axis')
ax1.set_zlabel('Z - axis')

fig2, ax2 = plt.subplots(3, 1, figsize=FIGSIZE)
ax2[0].plot(tendon_actions_un[:, 0], 'r.-')
ax2[1].plot(tendon_actions_un[:, 1], 'r.-')
ax2[2].plot(tendon_actions_un[:, 2], 'r.-')

ax2[0].plot(for_plot[:, 0], 'b.-')
ax2[1].plot(for_plot[:, 1], 'b.-')
ax2[2].plot(for_plot[:, 2], 'b.-')

EXTREMA_AVAILABLE = True
model_directory = data_directory + chosen_shape + "_model/"
file_name = model_directory + "normalization_parameters_dict_" + str(frequency) + "Hz.mat"
if EXTREMA_AVAILABLE:
    loaded_matrix = loadmat(file_name)
    total_p_min = np.squeeze(loaded_matrix['positions_min'])
    total_p_max = np.squeeze(loaded_matrix['positions_max'])
    total_act_min = np.squeeze(loaded_matrix['actuations_min'])
    total_act_max = np.squeeze(loaded_matrix['actuations_max'])
    scaling_factor = np.squeeze(loaded_matrix['scaling_factor'])
else:
    total_p_min = np.zeros(robot_positions_un.shape[1])
    total_p_max = np.zeros(robot_positions_un.shape[1])
    for i in range(robot_positions_un.shape[1]):
        total_p_min[i] = round(min(robot_positions_un[:, i]), 5)
        total_p_max[i] = round(max(robot_positions_un[:, i]), 5)

    total_p_min, total_p_max = np.array(total_p_min), np.array(total_p_max)

    total_act_min = np.zeros(tendon_actions_un.shape[1])
    total_act_max = np.zeros(tendon_actions_un.shape[1])
    for i in range(tendon_actions_un.shape[1]):
        total_act_min[i] = round(min(tendon_actions_un[:, i]), 5)
        total_act_max[i] = round(max(tendon_actions_un[:, i]), 5)

    scaling_factor = 1.0

    savemat(file_name, {'actuations_min': total_act_min, "actuations_max": total_act_max,
                        "positions_min": total_p_min, "positions_max": total_p_max,
                        "scaling_factor": scaling_factor})

print("Total px min: ", total_p_min)
print("Total px max: ", total_p_max)
print("Total act min: ", total_act_min)
print("Total act max: ", total_act_max)

robot_positions_n = normalize_it(robot_positions_un, total_p_min=total_p_min, total_p_max=total_p_max,
                                 scaling_factor=scaling_factor)
print("Normalized Rod positions shape: ", robot_positions_n.shape)
tendon_actions_n = normalize_it(tendon_actions_un, total_p_min=total_act_min, total_p_max=total_act_max,
                                   scaling_factor=scaling_factor)
print("Normalized pneumatic actions shape: ", tendon_actions_n.shape)

fig3 = plt.figure(figsize=FIGSIZE)
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(robot_positions_n[:, 0], robot_positions_n[:, 1], robot_positions_n[:, 2], c=robot_positions_un[:, 2], cmap='viridis', marker='o', s=1)
ax3.plot3D(robot_positions_n[:, 0], robot_positions_n[:, 1], robot_positions_n[:, 2], 'r.-')
ax3.set_xlabel('X - axis')
ax3.set_ylabel('Y - axis')
ax3.set_zlabel('Z - axis')

fig4, ax4 = plt.subplots(3, 1, figsize=FIGSIZE)
ax4[0].plot(tendon_actions_n[:, 0])
ax4[1].plot(tendon_actions_n[:, 1])
ax4[2].plot(tendon_actions_n[:, 2])
plt.show()


TRAINING_FLAG = True
OL_TESTING_FLAG = True
CL_TESTING_FLAG = True
FREEZE_MODEL = True
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

ANN_file_name = model_directory + "INV_LSTM_model_weights_" + str(frequency) + "Hz_rev" + str(Baseline_variant) + ".keras"
ANN_gen_file = model_directory + "INV_LSTM_model_weights_" + str(frequency) + "Hz_rev" + str(Baseline_variant) + ".keras"
file_model = "INV_LSTM_model_rev" + str(Baseline_variant)

if OL_TESTING_FLAG:
    # ------------------------------- DATA PREPARATION FOR TRAINING ----------------------------------- #

    input_features, output_features = inverse_time_series_conversion(Positions=robot_positions_n,
                                                                     Actions=tendon_actions_n, Baseline_variant=Baseline_variant)

    # Split the data into training and testing dataset
    train_data, test_data, train_label, test_label = train_test_split(input_features, output_features, test_size=0.3,
                                                                      shuffle=True)
    print('Data is successfully split into training and testing data set.')

    # Reshape the data into None x 1 x n_inputs and None x 1 X n_outputs
    train_data = np.asarray(train_data).reshape(train_data.shape[0], 1, input_features.shape[1])
    train_label = np.asarray(train_label).reshape(train_label.shape[0], 1, output_features.shape[1])
    test_data = np.asarray(test_data).reshape(test_data.shape[0], 1, input_features.shape[1])
    test_label = np.asarray(test_label).reshape(test_label.shape[0], 1, output_features.shape[1])
    print('Data is reshaped.')

    if TRAINING_FLAG:

        def build_model():

            model = Sequential()

            model.add(LSTM(units=256, input_shape=(None, train_data.shape[2]),
                           activation="tanh", recurrent_activation=None, return_sequences=True))
            model.add(LSTM(units=256, activation="tanh", recurrent_activation=None, return_sequences=True))
            # model.add(Dense(units=256, activation="softsign"))
            model.add(layers.Dropout(rate=0.25))
            model.add(Dense(test_label.shape[2], activation="tanh"))

            # Define the optimizer learning rate as a hyperparameter.
            learning_rate = 0.001

            # Now compile the model
            model.compile(Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=["mean_absolute_error"])
            return model


        # build_model(keras_tuner.HyperParameters())
        # Hyper-parameters for the training model
        BATCH_SIZE = 64
        # LEARNING_RATE = 0.0001
        # DECAY_RATE = 1e-6
        EPOCHS = 100

        # tuner = keras_tuner.RandomSearch(hypermodel=build_model, objective='val_loss', max_trials=5,
        #                                  executions_per_trial=3, overwrite=True,
        #                                  directory=global_path + 'IDM_dir', project_name="IDM_ver0")
        print('Started tuning hyperparameters...')
        # tuner = keras_tuner.BayesianOptimization(hypermodel=build_model, objective='val_loss', max_trials=5,
        #                                          directory=global_path + 'IDM_dir', project_name="IDM_ver4")
        # tuner.search(train_data, train_label, epochs=EPOCHS, validation_split=0.2)

        # Get the top 2 models.
        # models = tuner.get_best_models(num_models=1)
        # best_model = models[0]

        # best_model.build(input_shape=(None, train_data.shape[2]))
        # print('Best model summary: ', best_model.summary())
        # print('Tuner summary: ', tuner.results_summary())

        # Get the top 2 hyperparameters.
        # best_hps = tuner.get_best_hyperparameters(5)
        # Build the model with the best hp.
        model = build_model()

        my_callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_directory + 'IDM_dir_' + str(frequency) + 'Hz/TD_robot_{epoch:02d}_IDM_norm_rev' + str(Baseline_variant) + '.keras',
                                               save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=model_directory + 'IDM_dir_' + str(frequency) + 'Hz/logs')
        ]

        history = model.fit(train_data, train_label, batch_size=BATCH_SIZE, validation_split=0.2,
                            epochs=EPOCHS, shuffle=True, verbose=2, callbacks=my_callbacks)

        # Saving the trained model
        model.save(ANN_file_name)
        print('Training is done and model is saved...')

        # Plots 'history'
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        plt.figure(1)
        plt.plot(loss_values, 'bo', label='training loss')
        plt.plot(val_loss_values, 'r', label='training loss val')
        plt.show()

    if TRAINING_FLAG:
        # For open loop testing
        model = tf.keras.models.load_model(ANN_file_name)
        print('model is loaded for OL testing...')
    else:
        # For open loop testing
        model = tf.keras.models.load_model(ANN_gen_file)
        print('model is loaded for OL testing...')
    # train_label_pred = model.predict(train_data, batch_size=30, verbose=1)
    test_label_pred = model.predict(test_data, verbose=1)

    # Inverse transform of the data
    test_label = test_label.reshape(test_label.shape[0], int(test_label.shape[2]))
    test_label_pred = test_label_pred.reshape(test_label.shape[0], int(test_label.shape[1]))

    # ---------------------- UN-NORMALIZATION ORIGINAL and PREDICTED LABELS ------------------------- #
    test_label = un_normalize_it(input_data=test_label, total_p_min=total_act_min, total_p_max=total_act_max, scaling_factor=scaling_factor)
    test_label_pred = un_normalize_it(input_data=test_label_pred, total_p_min=total_act_min, total_p_max=total_act_max, scaling_factor=scaling_factor)

    # Calculations and prints r2 score of traing and testing data
    print('R2 score d1 is:\t{}'.format(r2_score(test_label[:, 0], test_label_pred[:, 0])))
    print('R2 score d2 is:\t{}'.format(r2_score(test_label[:, 1], test_label_pred[:, 1])))
    print('R2 score d3 is:\t{}'.format(r2_score(test_label[:, 2], test_label_pred[:, 2])))


    fig_1, axs = plt.subplots(3, 1)
    axs[0].plot(test_label[:, 0], 'r')
    axs[1].plot(test_label[:, 1], 'r')
    axs[2].plot(test_label[:, 2], 'r')

    axs[0].plot(test_label_pred[:, 0], 'k')
    axs[1].plot(test_label_pred[:, 1], 'k')
    axs[2].plot(test_label_pred[:, 2], 'k')


    print('The overall R2 score of test data:\t{}'.format(r2_score(test_label, test_label_pred)))

    print('mse of d1 is:\t{}'.format(mean_squared_error(test_label[:, 0], test_label_pred[:, 0])))
    print('mse of d2 is:\t{}'.format(mean_squared_error(test_label[:, 1], test_label_pred[:, 1])))
    print('mse of d3 is:\t{}'.format(mean_squared_error(test_label[:, 2], test_label_pred[:, 2])))

    print('mae of d1 is:\t{}'.format(mean_absolute_error(test_label[:, 0], test_label_pred[:, 0])))
    print('mae of d2 is:\t{}'.format(mean_absolute_error(test_label[:, 1], test_label_pred[:, 1])))
    print('mae of d3 is:\t{}'.format(mean_absolute_error(test_label[:, 2], test_label_pred[:, 2])))

    print('The overall mae of test set_x is:\t{}'.format(mean_absolute_error(test_label, test_label_pred)))
    print('The overall mse of test set_y is:\t{}'.format(mean_squared_error(test_label, test_label_pred)))

    plt.show()


if FREEZE_MODEL:
    from tensorflow import keras
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    if TRAINING_FLAG:
        # For open loop testing
        model = tf.keras.models.load_model(ANN_file_name)
        print('model is loaded to generate actions for desired trajectory...')
    else:
        # For open loop testing
        model = tf.keras.models.load_model(ANN_gen_file)
        print('model is loaded to generate actions for desired trajectory...')

    print('outputs: ', model.outputs)
    print('inputs: ', model.inputs)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_directory + 'IDM_dir_' + str(frequency) + 'Hz/',
                      name=f"{file_model}.pb",
                      as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_directory + 'IDM_dir_' + str(frequency) + 'Hz/',
                      name=f"{file_model}.pbtxt",
                      as_text=True)


# To do closed loop testing now
if CL_TESTING_FLAG:

    # Load the frozen trained dynamics model for faster predictions
    with gfile.GFile(model_directory + "IDM_dir_" + str(frequency) + "Hz/INV_LSTM_model_" + str(Baseline_variant) + ".pb",
                     'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read(-1))

    with tf.Graph().as_default() as graph1:
        tf.import_graph_def(graph_def)

    sess1 = tf.compat.v1.Session(graph=graph1)
    IDM_op_tensor = sess1.graph.get_tensor_by_name('import/Identity:0')

    combined_positions = robot_positions_n[10:2000, :]
    tendon_inps = tendon_actions_n[10:2000, :]

    original_tau, predicted_tau = [], []

    current_inst_pos = combined_positions[0, :]
    past_inst_pos = combined_positions[0, :]
    next_inst_pos = combined_positions[1, :]
    current_inst_tau = tendon_inps[0, :]     # for comparison
    past_inst_tau = tendon_inps[0, :]
    past_past_inst_tau = tendon_inps[0, :]
    past_past_past_inst_tau = tendon_inps[0, :]
    for i in range(1, combined_positions.shape[0] - 1):
        if Baseline_variant == 2:   # ---> Results not shown
            tau_pos_inp = (np.hstack([past_past_inst_tau, past_inst_tau, past_inst_pos, current_inst_pos, next_inst_pos])).reshape(1, 1, 15)
        elif Baseline_variant == 1:   # ---> B^(2)
            tau_pos_inp = (np.hstack([past_inst_tau, past_inst_pos, current_inst_pos, next_inst_pos])).reshape(1, 1, 12)
        elif Baseline_variant == 0:   # ---> Results not shown
            tau_pos_inp = (np.hstack([past_inst_tau, current_inst_pos, next_inst_pos])).reshape(1, 1, 9)
        elif Baseline_variant == 3:   # ---> Results not shown
            tau_pos_inp = (np.hstack((past_past_inst_tau, past_inst_tau, current_inst_pos, next_inst_pos))).reshape(1, 1, 12)
        elif Baseline_variant == 4:   # ---> Results not shown
            tau_pos_inp = (np.hstack((past_past_inst_tau, past_inst_tau, next_inst_pos))).reshape(1, 1, 9)
        elif Baseline_variant == 5:   # ---> B^(3)
            tau_pos_inp = (np.hstack((past_inst_tau, next_inst_pos))).reshape(1, 1, 6)
        elif Baseline_variant == 6:   # ---> Results not shown
            tau_pos_inp = (np.hstack((current_inst_pos, next_inst_pos))).reshape(1, 1, 6)
        elif Baseline_variant == 7:   # ---> B^(1)
            tau_pos_inp = (np.hstack((past_inst_pos, current_inst_pos, next_inst_pos))).reshape(1, 1, 9)
        elif Baseline_variant == 8:   # ---> Main Policy Results
            tau_pos_inp = (np.hstack((past_past_past_inst_tau, past_past_inst_tau, past_inst_tau, next_inst_pos))).reshape(1, 1, 12)
        else:
            print("Case not implemented.....")
        # tau_pos_inp = next_inst_pos.reshape(1, 1, 3)
        # original pressure
        original_tau.append([current_inst_tau])
        # predicted pressure
        # pred_temp = np.squeeze((model.predict(tau_pos_inp)).reshape(1, 3))
        pred_temp = sess1.run(IDM_op_tensor, {'import/x:0': tau_pos_inp})
        pred_temp = np.squeeze((pred_temp).reshape(1, 3))

        predicted_tau.append([pred_temp])
        current_inst_tau = tendon_inps[i, :]           # current tau from original data
        past_past_past_inst_tau = past_past_inst_tau.copy()
        past_past_inst_tau = past_inst_tau.copy()
        past_inst_tau = pred_temp.copy()

        past_inst_pos = current_inst_pos.copy()
        current_inst_pos = next_inst_pos.copy()
        next_inst_pos = combined_positions[i + 1, :]

    original_tau = np.array(original_tau).reshape(len(original_tau), 3)
    predicted_tau = np.array(predicted_tau).reshape(len(predicted_tau), 3)

    # ---------------------- UN-NORMALIZATION ORIGINAL AND PREDICTED LABELS ------------------------- #
    original_tau = un_normalize_it(input_data=original_tau, total_p_max=total_act_max, total_p_min=total_act_min, scaling_factor=scaling_factor)
    predicted_tau = un_normalize_it(input_data=predicted_tau, total_p_max=total_act_max, total_p_min=total_act_min, scaling_factor=scaling_factor)

    # Calculations and prints r2 score of traing and testing data
    print('R2 score d1 is:\t{}'.format(r2_score(original_tau[:, 0], predicted_tau[:, 0])))
    print('R2 score d2 is:\t{}'.format(r2_score(original_tau[:, 1], predicted_tau[:, 1])))
    print('R2 score d3 is:\t{}'.format(r2_score(original_tau[:, 2], predicted_tau[:, 2])))

    fig_1, axs = plt.subplots(3, 1)
    axs[0].plot(original_tau[:, 0], 'r')
    axs[1].plot(original_tau[:, 1], 'r')
    axs[2].plot(original_tau[:, 2], 'r')

    axs[0].plot(predicted_tau[:, 0], 'k')
    axs[1].plot(predicted_tau[:, 1], 'k')
    axs[2].plot(predicted_tau[:, 2], 'k')

    print('The overall R2 score of test data:\t{}'.format(r2_score(original_tau, predicted_tau)))

    print('mse of d1 is:\t{}'.format(mean_squared_error(original_tau[:, 0], predicted_tau[:, 0])))
    print('mse of d2 is:\t{}'.format(mean_squared_error(original_tau[:, 1], predicted_tau[:, 1])))
    print('mse of d3 is:\t{}'.format(mean_squared_error(original_tau[:, 2], predicted_tau[:, 2])))

    print('mae of d1 is:\t{}'.format(mean_absolute_error(original_tau[:, 0], predicted_tau[:, 0])))
    print('mae of d2 is:\t{}'.format(mean_absolute_error(original_tau[:, 1], predicted_tau[:, 1])))
    print('mae of d3 is:\t{}'.format(mean_absolute_error(original_tau[:, 2], predicted_tau[:, 2])))

    print('The overall mae of test set_x is:\t{}'.format(mean_absolute_error(original_tau, predicted_tau)))
    print('The overall mse of test set_y is:\t{}'.format(mean_squared_error(original_tau, predicted_tau)))

    plt.show()





