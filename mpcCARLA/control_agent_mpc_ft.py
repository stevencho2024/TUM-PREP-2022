#!/usr/bin/env python3

#
# authors: Michael Seegerer (michael.seegerer@tum.de)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a model predictive controller to perform low-level waypoint following. """
import glob
import os
import sys
import time

from enum import Enum
from collections import deque
import random
import numpy as np
from termcolor import colored
import mpctools as mpc
from scipy.optimize import curve_fit

try:
    # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #     sys.version_info.major,
    #     sys.version_info.minor,
    #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla

    # from agents.navigation.controller import VehiclePIDController
    from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed, is_within_distance_ahead
except ImportError:
    print("Carla libary is not installed !!")

from mpcCARLA.road_aligned_mpc import CurvMPCController, converting_mpc_u_to_control, wrap2pi
from mpcCARLA.waypoint_utilities import *
from mpcCARLA.ft_mpc import FTController



class MpcFTControl(object):
    """
    MPC + FT controller is based model predictive controller to perform the low level control of
    a vehicle from client side.
    Here a MPC framework will used to obtain efficient trajectories, if this method reaches no feasible
    control for current state
    """

    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, tvs=dict(), opt_dict=None):
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._last_control = np.array([0, 0])
        self._opt_dict = opt_dict
        self.curv_args = np.array([0, 0, 0, 0])
        self.curv_x0 = 0
        self.desired_lane_id = 0
        self.ft_counter = 0

        # Option for the MPC controller
        self.manual_control_on = False

        # list of other target vehicles in the secenario
        self._tvs = tvs


        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=30)
        self._buffer_size = 21
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._passed_waypoints = deque(maxlen= 15)
        self._waypoint_buffer_mpc = None

        # Filling waypoint buffer
        self.sample_waypoints()

        # define size of states, inputs, etc., for Casadi
        self.Nx = 12 # Number of states
        self.Nu = 2  # Number of Inputs
        self.Nt = 10  # Number of Steps

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def sample_waypoints(self):
        """
        Filling the waypoint buffer with sample waypoint in front of teh vehicle and behind.
        :return:
        """
        self._sampling_radius = calculate_step_distance(get_speed(self._vehicle), 0.2, factor=1)
        if self._sampling_radius < 2:
            self._sampling_radius = 3

        # Getting future waypoints
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=15,
                                                     stay_on_lane=True, active_lane_change=self.changing_lane)
        # Getting waypoint history --> history somehow starts at last wp of future wp
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._previous_wp_queue = compute_previous_waypoints(self._current_waypoint, d=self._sampling_radius, k=5,
                                                             stay_on_lane=True, active_lane_change=self.changing_lane)

        # concentrate history, current waypoint, future
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._waypoint_buffer.extendleft(self._previous_wp_queue)
        self._waypoint_buffer.append((self._map.get_waypoint(self._vehicle.get_location()), RoadOption.LANEFOLLOW))
        self._waypoint_buffer.extend(self._next_wp_queue)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        #self._dt = 1.0 / 30.0
        self._dt = 0.2
        self._target_speed = 80.0  # Km/h
        self._sampling_radius = calculate_step_distance(self._target_speed, self._dt,
                                                        factor=1)  # factor 11 --> prediction horizon 10 steps
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        self.data_log = dict()

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = calculate_step_distance(self._target_speed, opt_dict['sampling_radius'],
                                                                factor=5)  # factor 11 --> prediction horizon 10 steps

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.desired_lane_id = self._current_waypoint.lane_id

        # fill waypoint trajectory queue
        self._waypoints_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=200)

        # Set Vehicle controller
        state_dim = {'Nx': self.Nx, 'Nu': self.Nu, 'Nt': self.Nt}
        self._vehicle_controller = CurvMPCController(vehicle=self._vehicle, dt=self._dt, args_state_dimension=state_dim)
        # failsafe controller
        self._vehicle_ft_controller = FTController(state_0=xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius))

    def set_speed(self, speed=None):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        if speed is None:
            speed = self.current_traffic_speed_limit
        self._target_speed = speed

    @property
    def current_traffic_speed_limit(self):
        return self._vehicle.get_speed_limit()

    @property
    def changing_lane(self):
            return self._current_waypoint.lane_id - self.desired_lane_id

    def get_state(self):
        return self._vehicle_controller.state

    def set_lane_change(self, lane_change: int):
        """
        Trigger a lane change of the ego vehicle. Therefore, the desired lane id of the controller will be increased
        or decreased by the integer number in lane_change.
        If desired_lane_id and current lane id of ego vehicle dont match, controller will trigger the future waypoints
        on the desired lane id.

        :param lane_change: integer number between [-1, 1]
        :return:
        """
        self.desired_lane_id = self.desired_lane_id - lane_change



    def run_step(self,timestep:int,  debug=True, log=False):
        """
        Execute one step of classic mpc controller which follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        start_time = time.time()

        # Update target velocity to current speed limit
        self.set_speed()

        # Trigger a lane change
        # if timestep / 30 == 12:
        #     self.set_lane_change(-1)

        # ----------------------------------------------------------------
        # Updating reference wp line --> only needed if nmpc is solved
        # ----------------------------------------------------------------
        if timestep % 6 == 0 or True:
            # not enough waypoints in the horizon? => Sample new ones
            self._sampling_radius = calculate_step_distance(get_speed(self._vehicle), 0.2, factor=1)
            if self._sampling_radius < 2:
                self._sampling_radius = 3


            # Getting future waypoints
            self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=15, stay_on_lane=True, active_lane_change=self.changing_lane)
            # Getting waypoint history --> history somehow starts at last wp of future wp
            self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            self._previous_wp_queue = compute_previous_waypoints(self._current_waypoint, d=self._sampling_radius, k=5, stay_on_lane=True, active_lane_change=self.changing_lane)

            # concentrate history, current waypoint, future
            self._waypoint_buffer = deque(maxlen=self._buffer_size)
            self._waypoint_buffer.extendleft(self._previous_wp_queue)
            self._waypoint_buffer.append((self._map.get_waypoint(self._vehicle.get_location()), RoadOption.LANEFOLLOW))
            self._waypoint_buffer.extend(self._next_wp_queue)

            self._waypoints_queue = self._next_wp_queue

            # target waypoint for Frenet calculation
            self.wp1 = self._map.get_waypoint(self._vehicle.get_location())
            self.wp2, _ = self._next_wp_queue[0]

            # Flipping x-axis of wp1 for calculation
            # self.wp1.transform.location.x = -1 * self.wp1.transform.location.x

            # calculation of kappa --> only needed if nmpc problem is solved.
            self.kappa_log = dict()
            self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            self.curv_args, self.kappa_log = self.calculate_curvature_func_args(log=True)
            self.curv_x0 = func_kappa(0, self.curv_args)

            # input for MPC controller --> wp_current, wp_next, kappa
            target_wps = [self.wp1, self.wp2, self.curv_x0]
            target_wps2 = [self.wp1, self.wp2, self.curv_args]

        # If no more waypoints in queue, returning emergency braking control
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            print("Applied emergency control !!!")

            return control

        # Update the constraints for the current scene
        if self._tvs:
            self._vehicle_controller.set_constraints(self.get_constraints())

        # -------------------------------------------------------------------------
        # Getting control from MPC controller, define manual control if desired
        # ------------------------------------------------------------------------
        # OCP gets only solved every 6. timestep. --> every 0.2 seconds by a 30 fps
        if timestep % 6 == 0:
            status, control, state, u, u_log, x_log, _ = self._vehicle_controller.mpc_control(target_wps2, self._target_speed, solve_nmpc=True, log=log)
            # Updating logging information of the logger
            self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                             'Eta': state[5],
                             'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                             'pred_states': [x_log],
                             'pred_control': [u_log], 'computation_time': time.time() - start_time, "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}

        else:
            status = True
            control, state, prediction, u = self._vehicle_controller.mpc_control(target_wps,self._target_speed, solve_nmpc=False, log=log)
            # Updating logging information of the logger
            self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                             'Eta': state[5],
                             'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                             'pred_states': [prediction],
                             'pred_control': self.data_log.get("pred_control"), 'computation_time': time.time() - start_time,
                             "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}
        self.data_log["velocity_error"] = state[3] - self._target_speed / 3.6
        # self.data_log['kappa_state'] = state[9]
        self.data_log['target_velocity'] = self._target_speed


        # if SMPC solving was successful update the failsafe trajectory based
        if status and timestep % 6 == 0:
            # predicted state of the vehicle after applying the first derived control
            x_mpc_pred_1 = predict_frenet_kinVehMod(xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius), u, self.curv_x0)
            self._vehicle_ft_controller.mpc_control(x_mpc_pred_1, self.curv_args, self._target_speed, solve_nmpc=True, log=log)
            self.ft_counter = 0
        elif status == False:
            # Getting ft control based on position of ft counter
            control = converting_mpc_u_to_control(u=self._vehicle_ft_controller.failsafe_trajectory[self.ft_counter,:])
            self.ft_counter += 1

        # Print State information on TVs in realtion to EV
        for tv_name, tv in self._tvs.items():
            print(colored(tv_name+' current state:'+
                          str(xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)) +
                          '   euclidean distance: '+
                          str(distance_vehicle(self.wp1, tv.get_transform())), 'white', 'on_red'))

        if debug and timestep % 6 == 0:
            prediction_location = []
            current_location = self._vehicle.get_location()
            for i in range(self.Nt):
                loc = carla.Location(x=-1 * x_log[i, 0], y=x_log[i, 1], z=current_location.z + 0.5)
                prediction_location.append(loc)

            draw_prediction_trajectory(self._vehicle.get_world(), prediction_location)
            last_wp, _ = self._waypoint_buffer[len(self._waypoint_buffer)-1]
            draw_waypoints_debug(self._vehicle.get_world(), [
                self.wp1, self.wp2], self._vehicle.get_location().z + 1.0)
            draw_waypoints_debug(self._vehicle.get_world(), [
                last_wp], self._vehicle.get_location().z + 1.0, color=(0, 255, 0))
            self.visualizeKappa2Carla(self.curv_args)


        return control, self.data_log, self.kappa_log if log else control


    def calculate_curvature_func_args(self, log=False):
        """
        Function to calculate the curvature function arguments.
        This is done by fitting all reference waypoint from the waypoint buffer to 3th-polynomial function.
        :return: [p0, p1, p2, p3] of the 3th polynomial.
        """
        kappa_log = dict()
        # Current waypoint as reference point to center all waypoints
        ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])

        # Rotation matrix to align current waypoint to x-axis
        angle_wp = get_wp_angle(self.wp1, self.wp2)
        R = rotmat(angle_wp)

        # Getting reduced waypoint matrix
        wp_mat = np.zeros([self._buffer_size, 2])
        wp_mat_0 = np.zeros([self._buffer_size, 2])  # saving original locations of wps for debug

        counter_mat = 0
        for i in range(self._buffer_size):
            wp = self._waypoint_buffer[i][0]
            loc = get_localization_from_waypoint(wp)
            point = np.array([loc.x,  loc.y])  # init point
            point = point.T - ref_point.T  # center traj to origin
            point = mpc.mtimes(R, point)  # rotation of point ot allign with x-axis

            wp_mat_0[counter_mat] = np.array([loc.x, loc.y])
            wp_mat[counter_mat] = point
            counter_mat += 1

        # Getting optimal parameters for 3th polynomial by fitting the a curve
        # Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        # p_opt, _ = curve_fit(polynomial3, wp_mat[:, 0], wp_mat[:, 1])
        p_opt = np.polynomial.polynomial.polyfit(wp_mat[:, 0], wp_mat[:, 1], deg=3)
        # Logging all necessary information
        if log:
            kappa_log = {
                'wp_mat': [wp_mat],
                'wp_mat_0': [wp_mat_0],
                'refernce_point': [ref_point],
                'rotations_mat': [R],
                'angle_wp': [angle_wp],
                'p_opt': [p_opt]
            }

        return p_opt, kappa_log


    def visualizeKappa2Carla(self, p_opt):

        # Rotation matrix to align current waypoint to x-axis
        angle_wp = get_wp_angle(self.wp1, self.wp2)
        R_inv = np.linalg.inv(rotmat(angle_wp))
        R_inv = inv_rotmat(angle_wp)

        # Current waypoint as reference point to center all waypoints
        ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])


        # Define polynomial function
        p = np.poly1d(p_opt[-1::])
        p = np.poly1d(p_opt)

        # Sammpling points on refernce curve
        number_sample = 100
        number_hist_samples = 10
        ref_curve_points = np.zeros([number_sample + number_hist_samples,2])
        for k in range(number_hist_samples):
            x= -2*(number_hist_samples - k)
            y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
            point = np.array([x, y])


            # Roation
            point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
            # Translation
            point = point.T + ref_point.T  # center traj to origin
            # Converting Point into Carla coordination system
            ref_curve_points[k] = np.array([-1*point[0], point[1]])


        for k in range(number_sample):
            x= 2*k
            y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
            point = np.array([x, y])


            # Roation
            point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
            # Translation
            point = point.T + ref_point.T  # center traj to origin
            # Converting Point into Carla coordination system
            ref_curve_points[k+number_hist_samples] = np.array([-1*point[0], point[1]])

        # Convert to a list carla location objects
        ref_curve_location = []
        current_location = self._vehicle.get_location()
        for i in range(number_sample):
            loc = carla.Location(x= ref_curve_points[i, 0], y=ref_curve_points[i, 1], z=current_location.z + 0.5)
            ref_curve_location.append(loc)

        draw_prediction_trajectory(self._vehicle.get_world(), ref_curve_location, color=[0, 255, 0], thickness=0.1) # Draw green line


    def get_constraints(self):
        """
        Returning constraints inequality function based on current scene.
        :return: constraints function
        """

        # define inequality constraint for disctance between TV and EV


        # vehicle size
        car_dim = {'width': 2, 'length': 6}


        # Case 210: one TV vehicle in front is consider for constraints --> vertical xi constraint
        # Case 222: case 210 + one TV on right lane of EV consider in constraints
        # case 21121: case 210 but without vertical linear constraint, instead linear inclined constraint to overtake TV is used
        case = 21121



        min_dist = 0.5 * get_speed(self._vehicle)
        Delta = 0.2

        if self._tvs:

            tv1_state = xy2frenet_wp(self._tvs['TV1'], self._map, self._waypoint_buffer, self._sampling_radius)
            tv2_state = xy2frenet_wp(self._tvs['TV2'], self._map, self._waypoint_buffer, self._sampling_radius)
            ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)

            wp_tv1 = self._map.get_waypoint(self._tvs['TV1'].get_location(), project_to_road=True,
                                           lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            wp_tv2 = self._map.get_waypoint(self._tvs['TV2'].get_location(), project_to_road=True,
                                           lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))


            ub_cond = dict()
            lb_cond = dict()
            lb_cond['eta'] = lambda t: mpc.vcat([t * 0 - np.inf])
            ub_cond['xi'] = lambda t: mpc.vcat([t * 0  + np.inf])

            take_over_counter = 0


            if case == 210:
                dim_lin_cond = 1

                ub_cond['xi'] = lambda t: mpc.vcat([tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
                lcon = lambda x, s: mpc.vcat([x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s])

            if case == 220:
                dim_lin_cond = 2
                ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
                lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[1]) * (car_dim['width'] + 1)])
                lcon = lambda x, s: mpc.vcat([  x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
                                                - x[5] + (0.5 * tv2_state[5]) -s[1]])

            if case == 21121:
                dim_lin_cond = 2
                #ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
                #lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[5]) * (car_dim['width'] + 1)])
                lcon = lambda x, s: mpc.vcat([  #x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
                                                - x[5] + (0.75 * tv2_state[5]) -s[0],
                                                - x[5] + ((2.5* car_dim['width'] + tv1_state[5] - ev_state[5]) / (tv1_state[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1]
                                                ])

                if abs(wp_tv1.lane_id) > abs(self.wp1.lane_id):
                    take_over_counter += 1
                    lcon = lambda x, s: mpc.vcat(
                        [  # x[4] - (tv1_state[0] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
                            - x[5] + (0.5 * tv2_state[5]) - s[0],
                            #- x[5] + (car_dim['width']) - s[0],
                            #- x[5] + (0.5 * car_dim['width']) - s[1],
                            s[1]
                        ])

            cond = {'x_bounds': ub_cond, 'linear_cond': lcon, 'x_bounds_low': lb_cond, 'dim_lin_cond': dim_lin_cond}

        return cond

    def get_constraints2(self):
        """
        Returning constraints inequality function based on current scene.
        :return: constraints function
        """

        # define inequality constraint for disctance between TV and EV

        # vehicle size
        car_dim = {'width': 2, 'length': 6}

        min_dist = 0.5 * get_speed(self._vehicle)
        Delta = 0.2

        self._tv_states = {}

        if self._tvs:
            for tv_name, tv in self._tvs.items():
                if "TV1" in tv_name:
                    self._tv_states[tv_name] = xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)

                    wp_tv = self._map.get_waypoint(tv.get_location(), project_to_road=True,
                                                   lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

                    # Case B
                    if self.wp1.lane_id == wp_tv.lane_id and self._tv_states[tv_name][0] > 0:
                        ub_cond = lambda t: mpc.vcat([self._tv_states[tv_name][0] + t * Delta *
                                                      self._tv_states[tv_name][3] - car_dim['length'] - 2])
                        lcon = lambda x, s: mpc.vcat([x[4] - (
                                    self._tv_states[tv_name][0] + x[11] * Delta * self._tv_states[tv_name][
                                3] - min_dist - car_dim['length']) - s])

                    # if self.wp1.lane_id != wp_tv.lane_id:
                    #     diff_lane = abs(self.wp1.lane_id) - abs(wp_tv.lane_id)
                    #
                    #     if

            cond = {'x_bounds': ub_cond, 'linear_cond': lcon}

        return cond