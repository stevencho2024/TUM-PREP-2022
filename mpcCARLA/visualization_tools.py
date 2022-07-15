""" This module contains toosl to visualize the performance of MPC controllers. """

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_log_informations(filename: str):
    """
    Reading the h5-logfile.
    :param filename: h5-filename
    :return: panda.Dataframe
    """
    return pd.read_hdf(filename, 'frenet_log')


def plot_frenet_states(filename: str, time=None):
    """
    Plotting the frenet states.
    :param filename: h5-filename
    :param time: timestamp for zoom-in.
    :return:
    """
    plt.rcParams['figure.figsize'] = [20, 30]

    # Reading the h5-file to retrieve the panda dataframe with the log-information
    data = read_log_informations(filename)

    # Coping the dataframe and convert Velocity entry from (m/s) to (km/h)
    data_copz = data.copy()
    data_copz['Velocity'] *= 3.6

    # Plotting the Frenet state [velocity, ETA, THETA] + curvature information
    plots = data_copz.plot.line(y=['Velocity', 'Eta', 'Theta', "kappa"], lw=5, subplots=True)

    # Set plot styles
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('time t (s)', size=40)
    plt.tick_params(axis='x', pad=2)

    # Labeling the axes + adding balck line at zero
    plt.title('Frenet state MPC', size=50)
    y_labels = ['Velocity v(m/s)', 'Lateral offset η (m)', 'Orientation offset φ (rad)', 'Curvature κ (1/m)']
    plots[0].plot(data.index, data['target_velocity'], color='green', lw=5)
    [plots[i].hlines(0, 0, 300, lw=4) for i in range(len(plots))]
    [plots[i].legend(loc='upper right', ) for i in range(len(plots))]
    [plots[i].set_ylabel(y_labels[i], size=40) for i in range(len(plots))]

    # Tighten up the plot layout
    plt.tight_layout()

    # Zoom in at a specific timestamp
    if time:
        plt.xlim([time - 4, time + 4])
        [plots[i].axvline(time, color='black') for i in range(4)]

    # Saving the plot as pdf
    plot_filename = filename.split(".")
    plot_filename = plot_filename[0].split("/")
    plt.savefig('plots/'+plot_filename[-1]+'_frenet_state_plot.pdf')


def compare_velocity_errors(data1, data2, txt: str, filename: str= None, timestamp=0):
    plt.rcParams['figure.figsize'] = [20, 15]

    states_labels = ["Velocity", 'velocity_error', ]

    # Copying the dataframe and convert every velocity related entry from (m/s) to (km/h)
    data1_copy = data1.copy()
    data1_copy[states_labels] = data1[states_labels].multiply(3.6)
    # Plotting the first dataframe + target speed
    plots = data1_copy.plot.line(y=states_labels, subplots=True, lw=4, color='blue')
    plots[0].plot(data1_copy.index, data1_copy['target_velocity'], color='green', lw=5)

    # Adding the plots of the second dataframe to the plots
    for i, label in enumerate(states_labels):
        plots[i].plot(data2.index, data2[label] * 3.6, "--", color='red', label=str(txt), lw=4)
        plots[i].legend()

    # Set plot styles
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('time t (s)', size=40)
    plt.tick_params(axis='x', pad=15)

    # Labeling the axes + adding balck line at zero
    y_labels = ['Velocity v (km/h)', 'Velocity v (km/h)', '[rad]', '[1/m]']
    titles = ['Total velocity of EV', 'Velocity offset to target speed']
    [plots[i].hlines(0, 0, 300, lw=3) for i in range(len(states_labels))]
    [plots[i].legend(loc='lower right', ) for i in range(len(states_labels))]
    [plots[i].set_ylabel(y_labels[i], size=40) for i in range(len(states_labels))]
    [plots[i].set_title(titles[i]) for i in range(len(states_labels))]

    # Zoom in at a specific timestamp
    if timestamp > 0:
        plt.xlim([timestamp - 4, timestamp + 4])
        [plots[i].axvline(timestamp, color='black') for i in range(7)]

    # Saving the plot to pdf
    #plt.tight_layout()
    if filename is None:
        filename = 'plots/comparison_MPC_with_integral_velocity.pdf'
    plot_filename = filename.split(".")
    plot_filename = plot_filename[0].split("/")
    plt.savefig('plots/' + plot_filename[-1] +'_comparison.pdf')


def compare_errors(file1, file2, filename:str = None, txt: str = None, timestamp=0):
    """
    Plotting function to compare the control performance of two dataframes.
    :param file1: h5-file of first measurement
    :param file2: h5-file of second measurement
    :param txt: Text displayed in legend for 2. dataframe (e.g. 'PID')
    :param timestamp:
    :return:
    """

    # Loading the Dataframes
    data1 = read_log_informations(file1)
    data2 = read_log_informations(file2)

    if filename is None:
        filename = 'comparison_frenet_states_with_pid_plot.pdf'
    plt.rcParams['figure.figsize'] = [20, 30]

    states_labels = ['velocity_error', 'Eta', 'Theta', "kappa"]
    plots = data1.plot.line(y=states_labels, subplots=True, lw=5, color='blue')

    for i, label in enumerate(states_labels):
        color = plots[i].get_lines()[-1].get_c()
        plots[i].plot(data2.index, data2[label], "--", color='red', label=str(txt), lw=5)
        plots[i].legend()

    # Set plot styles
    # Set plot styles
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('time t (s)', size=40)
    plt.tick_params(axis='x', pad=15)

    y_labels = ['Velocity v(m/s)', 'Lateral offset η (m)', 'Orientation offset φ (rad)', 'Curvature κ (1/m)']
    [plots[i].hlines(0, 0, 155, lw=4) for i in range(4)]
    # [plots[i].legend(loc='upper right',) for i in range(4)]
    [plots[i].set_ylabel(y_labels[i], size=40) for i in range(4)]

    # Zoom in at a specific timestamp
    if timestamp > 0:
        plt.xlim([timestamp - 4, timestamp + 4])
        [plots[i].axvline(timestamp, color='black') for i in range(7)]

    plt.tight_layout()
    plot_filename = filename.split(".")
    plot_filename = plot_filename[0].split("/")
    plt.savefig('plots/' + plot_filename[-1] + '_comparison.pdf')



def plot_prediction_at_time(filename: str, time: float):


    data = read_log_informations(filename)

    plt.rcParams['figure.figsize'] = [40, 20]

    panda_labels = ['u_acceleration', 'u_steering_angle', 'PSI', 'Theta', 'Eta', 'kappa', ]
    plots = data.plot.line(y=panda_labels, subplots=True, layout=(3,2), lw=4)
    plt.suptitle("Prediction at timestamp:" + str(time), fontsize=16)
    prediction_u = data.loc[time]['pred_control']
    prediction_states = data.loc[time]['pred_states']


    # Set plot styles
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.rcParams.update({'font.size': 30})
    plt.xlabel('time t (s)', size=40)
    plt.tick_params(axis='x', pad=15)

    # Prediction plots
    pred_time = np.arange(time, time + .2 * 10, 0.2)

    plots[0][0].plot(pred_time, prediction_u[0], "r--", label='prediction', lw=5)
    plots[0][1].plot(pred_time, prediction_u[1], "r--", label='prediction', lw=5)
    plots[1][0].plot(pred_time, prediction_states[:, 2], "r--", label='prediction',lw=5)
    plots[2][0].plot(pred_time, prediction_states[:, 5], "r--", label='prediction', lw=5)
    plots[1][1].plot(pred_time, prediction_states[:, 6], "r--", label='prediction', lw=5)
    plots[2][1].plot(pred_time, prediction_states[:, 9], "r--", label='prediction', lw=5)


    # Zooming to prediction timestamp
    plt.xlim([time - 0.4, time + 2.4])
    # Update legend and axis in every plot
    y_labels = ['Acceleration a(m2/s)', 'Orientation offset φ (rad)', 'Lateral offset η (m)', 'Steering (rad)','Orientation (rad)', 'Curvature κ (1/m)']
    [plots[i%3][i%2].hlines(0, time - 5, time + 5) for i in range(len(y_labels))]
    [plots[i%3][i%2].legend() for i in range(len(y_labels))]
    [plots[i%3][i%2].set_ylabel(y_labels[i], size=30) for i in range(len(y_labels))]


    plot_filename = filename.split(".")
    plot_filename = plot_filename[0].split("/")
    plt.savefig('plots/' + plot_filename[-1] + '_prediction.pdf')
    

