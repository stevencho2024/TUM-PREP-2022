""" This module contains a model predictive controller to perform low-level waypoint following. """
#performs MPC step calc, polynomial fit, and condition creation/application
import glob
import math
import os
from signal import SIG_DFL
import sys
import time
import math

from enum import Enum
from collections import deque
import random
import numpy as np
from termcolor import colored
import mpctools as mpc
from scipy.optimize import curve_fit
import pdb
import cmath
from scipy import optimize as opt
from scipy.linalg import block_diag
from collections import deque
#from sklearn.cluster import DBSCAN
np.set_printoptions(precision=4, suppress=True)

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

from mpcCARLA.road_aligned_mpc import CurvMPCController, converting_mpc_u_to_control, wrap2pi, MPCController
from mpcCARLA.waypoint_utilities import *


class VehicleCurvMPC(object):
    """
    Model Predictive Controller implements the basic behavior of following a trajectory of waypoints that is generated
     on-the-fly.
    When multiple paths are available (intersections) this controller makes a random choice.
    """

    MIN_DISTANCE_PERCENTAGE = 0.9
    #define all paramters in the class Vehicle Cruve MPC
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

        self.T = 0.2 #sampling time
        self.K = np.array([[0, -0.55, 0, 0],[0, 0, -0.63, -1.15]]) #from the paper

        # Option for the MPC controller
        self.manual_control_on = False

        # list of other target vehicles in the secenario
        self._tvs = tvs
        self._tvs = []

        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=30)
        self._buffer_size = 21
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._passed_waypoints = deque(maxlen= 15)
        self._waypoint_buffer_mpc = None

        # define size of states, inputs, etc., for Casadi
        self.Nx = 12 # Number of states
        self.Nu = 2  # Number of Inputs (steer and acceleration)
        self.Nt = 10  # Number of Steps

        # initializing controller
        self._init_controller(opt_dict)

        self._car_width = 2
        self._car_length = 6
        self._lane_width = 3.5
        self._lidar_height = 1.6
        self._last_frenet_tvs = {'front': 0, 'right': 0, 'left':0}
        self._TV_locs = []
        self.A = np.array([[1, self.T, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.T],
            [0, 0, 0, 1] 
            ])

        self.B = np.array([[0.5*(self.T ** 2), 0],
            [self.T, 0],
            [0, 0.5*(self.T ** 2)],
            [0, self.T]])
        self.timesteps = 5 #within one MPC cycle
        self.wp_ev = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True,
                                        lane_type=(carla.LaneType.Driving))

    def stoch_bubble(self):
        #propagates the array uncertainty of tv_state variables (xi, v_xi, eta, V_eta)of the lidar data. 
        #returns a 2D diag array
        #intended for car_dim modification 
        var_x0_tv = np.diag((0.05, 0.01, 0.05, 0.01))
        var_omega = np.diag((0.1, 0.05, 0.1, 0.05))
        vars_tv = [var_x0_tv]
        Bk = np.matmul(self.B, self.K)
        coeff = np.add(self.A, Bk)
        second_term = np.matmul(self.B, np.matmul(var_omega, np.transpose(self.B)))
        for n in range(self.timesteps):
            first_term = np.matmul(coeff, np.matmul(vars_tv[n],np.transpose(coeff)))
            next_var_tv = first_term + second_term
            vars_tv.append(next_var_tv)
        return vars_tv

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

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
        self.desired_lane_id = self._current_waypoint.lane_id #stores initial lane

        # fill waypoint trajectory queue
        self._waypoints_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=200)

        # Set Vehicle controller
        state_dim = {'Nx': self.Nx, 'Nu': self.Nu, 'Nt': self.Nt}
        self._vehicle_controller = CurvMPCController(vehicle=self._vehicle, dt=self._dt, args_state_dimension=state_dim)

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



    def run_step(self, data, timestep:int,  debug=True, log=False, printer=True):
        """
        Execute one step of classic mpc controller which follow the waypoints trajectory.
        :param debug: boolean flag to activate waypoints debugging
        :return:
        """
        
        start_time = time.time()

        # Update target velocity to current speed limit
        self.set_speed()

        self._time_step = 0.2 #seconds



        
    
        #pdb.set_trace()

        # self._tvs = []
        # #storing previous xi values
        # for car in range(len(TVs_avg_loc)):
        #     self._tvs[car] = xy2frenet_pnt_specific(self._vehicle, self._last_frenet_tvs[TVs_avg_loc[car][1]], self._time_step, TVs_avg_loc[car][0], self._map, self._waypoint_buffer, self._sampling_radius) #first iteration is not accuarate becasue of lack of previous 
        #     self._last_frenet_tvs[TVs_avg_loc[car][1]] = self._tvs[car][0] #just xi (each term is in TV form)

        #pdb.set_trace()


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

            self._tvs = [] #reset self._tvs

            ###compute waypoints will have to be modified into using sensor data######
            # Getting future waypoints
            self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
            self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=15, stay_on_lane=True, active_lane_change=self.changing_lane) #returns list
            # Getting waypoint history --> history somehow starts at last wp of future wp (previous waypoints)
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
            self.curv_args, self.kappa_log = self.calculate_curvature_func_args(log=True) #gets the fitted poly
            self.curv_x0 = func_kappa(0, self.curv_args)

            
            # input for MPC controller --> wp_current, wp_next, kappa
            target_wps = [self.wp1, self.wp2, self.curv_x0]
            target_wps2 = [self.wp1, self.wp2, self.curv_args]

            ###############frenet frame calc from lidar data
            ## exclude unwanted points 
            forward_vect = self._vehicle.get_transform().get_forward_vector()

            #getting theta angle wrt to trajectory
            wp_current = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            wp_next = wp_current.next(2)[0]
            angle_wp = get_wp_angle(wp_current, wp_next) #gets angle of waypoints wrt x axis
            yaw = self._vehicle.get_transform().rotation.yaw #vehicle angle wrt x-axis
            yaw_rad = np.deg2rad(round(yaw, 3))
            vehicle_heading = wrap2pi(np.pi - yaw_rad)
            theta = wrap2pi(vehicle_heading - angle_wp) #angle offset of car from trajectory

            #lane_vect_norm = np.array([forward_vect.x*np.cos(-theta) - forward_vect.y*np.sin(-theta), forward_vect.x*np.sin(-theta) + forward_vect.y*np.cos(-theta), 0]) / np.linalg.norm(np.array([forward_vect.x*np.cos(-theta) - forward_vect.y*np.sin(-theta), forward_vect.x*np.sin(-theta) + forward_vect.y*np.cos(-theta), 0]))
            lane_vect_norm = np.array([wp_next.transform.location.x - wp_current.transform.location.x , wp_next.transform.location.y - wp_current.transform.location.y , wp_next.transform.location.z - wp_current.transform.location.z ]) / np.linalg.norm(np.array([wp_next.transform.location.x - wp_current.transform.location.x , wp_next.transform.location.y - wp_current.transform.location.y , wp_next.transform.location.z - wp_current.transform.location.z ]))
            #lane_vect_norm = np.array([v_vect.x*np.cos(theta) - v_vect.y*np.sin(theta), v_vect.x*np.sin(theta) + v_vect.y*np.cos(theta), 0]) / np.linalg.norm(np.array([v_vect.x*np.cos(theta) - v_vect.y*np.sin(theta), v_vect.x*np.sin(theta) + v_vect.y*np.cos(theta), 0])) #this vector contains coefficients for the front plane (doesn't work for steep incline/ declines)
            up_vect = self._vehicle.get_transform().get_up_vector()
            up_vect_norm = np.array([0, 0, up_vect.z]) / np.linalg.norm(np.array([0, 0, up_vect.z])) 
            #pdb.set_trace()
            side_vect_norm = np.cross(lane_vect_norm,up_vect_norm) / np.linalg.norm(np.cross(lane_vect_norm,up_vect_norm)) #this is also coefficients for side plane

            ev_location = self._vehicle.get_location() #xyz
            closest_waypoint_EV  = wp_current #plane location based on location of EV 
            ##NotE  I HAVE ACCESS TO LANE IDs of TVs with this method ABOVE
            closest_lane_location  = [closest_waypoint_EV.transform.location.x, closest_waypoint_EV.transform.location.y, closest_waypoint_EV.transform.location.z] #plane location based on location of EV 


            # Function to find shortestdistance between point and plane
            def shortest_distance(point, norm_vect, pnt_in_plane):
                #point (array): 3D point that you want to find the distance from the plane
                #norm_vector (array): the plane's normal vector 
                #pnt_in_plane (array): the pont in the plane to give a position in space (typically the EV's transform)

                a = norm_vect[0]
                b = norm_vect[1]
                c = norm_vect[2]
                last_var = -(a*pnt_in_plane[0] + b*pnt_in_plane[1] + c*pnt_in_plane[2])

                x1 = point[0]
                y1 = point[1]
                z1 = point[2]

                other = a * x1 + b * y1 + c * z1
                num = (other + last_var) #distance with sign CHECK if okay
                #den = (math.sqrt(a * a + b * b + c * c))
                shortest_dist = num
                #pdb.set_trace()
                #for efficiency possibly
                # d = norm_vect[0]*pnt_in_plane[0] + norm_vect[1]*pnt_in_plane[1] + norm_vect[2]*pnt_in_plane[2]
                # return abs((norm_vect[0] * point[0] + norm_vect[1] * point[1] + norm_vect[2] * point[2] + d)) / math.sqrt(norm_vect[0] ** 2 + norm_vect[1] ** 2 + norm_vect[2] ** 2)

                return shortest_dist
            
            #array of points and distance
            #relvnt_pnts = []

            relvnt_front_pnts_sum = [0, 0, 0]
            front_num_points = 0
            relvnt_front_pnts = [] 


            relvnt_rightside_pnts = []
            relvnt_rightside_pnts_sum = [0, 0, 0]
            right_num_points = 0

            relvnt_leftside_pnts = []
            relvnt_leftside_pnts_sum = [0, 0, 0]
            left_num_points = 0

            TVs_avg_loc = []

    
            for location in data: #data is in the form of lidarMeasurement (array of lidarDetection)
                #pdb.set_trace()
                if (location.point.z > -self._lidar_height + 0.1) and (location.intensity < 0.99): #rid of ground points and points caused by the EV itself
                    lid_location = [location.point.x*np.cos(yaw_rad) - location.point.y*np.sin(yaw_rad), location.point.x*np.sin(yaw_rad) + location.point.y*np.cos(yaw_rad), 0] #rotate the vector from lidar xyz orientation to match map xyz oreintation
                    #pdb.set_trace()
                    pnt_location = [ev_location.x + lid_location[0], ev_location.y + lid_location[1], ev_location.z + lid_location[2]] #convert from lidar centric xyz to map (problem. ang)
                    #this makes a vision rectangular box around EV 10.5 by 54
                    long_dist = shortest_distance(pnt_location, lane_vect_norm, closest_lane_location) #closest lane is fine, location is fine
                    lat_dist = shortest_distance(pnt_location, side_vect_norm, closest_lane_location)
                    #pdb.set_trace()
                    if (abs(long_dist) <= 4.5*self._car_length) and (abs(lat_dist) <= 1.3*self._lane_width): #initial box
                        #relvnt_pnts.append(location)
                        if (long_dist > 2.7) and (abs(lat_dist) <= 0.5*self._lane_width): #for cars ahead 2.7 is to ignore the front part of the EV
                            relvnt_front_pnts.append(pnt_location)
                            #print("FRONT intensity", location.intensity)
                            relvnt_front_pnts_sum = np.add(relvnt_front_pnts_sum, pnt_location)
                            front_num_points += 1
                            #pdb.set_trace()
                        elif (lat_dist > 0.5*self._lane_width): #for cars on the right side
                            relvnt_rightside_pnts.append(pnt_location)
                            #print("RIGHT intensity", location.intensity)
                            relvnt_rightside_pnts_sum = np.add(relvnt_rightside_pnts_sum, pnt_location)
                            right_num_points += 1
                            # if pnt_location[1] > -65:
                            #     pdb.set_trace()
                        elif (lat_dist < -0.5*self._lane_width): #for cars on the left side
                            relvnt_leftside_pnts.append(pnt_location)
                            relvnt_leftside_pnts_sum = np.add(relvnt_leftside_pnts_sum, pnt_location)
                            left_num_points += 1
                            #pdb.set_trace()

            #SCREW clusters just RID OF ground pints
            # #cluster for one for front points
            # # define the model
            # model = KMeans(n_clusters=2, random_state=0).fit(relvnt_front_pnts) #currently 2100 points generated total with 20 lasers shooting 20000 each
            # # fit model and predict clusters
            # yhat = model.cluster_centers_(relvnt_front_pnts)
            # # retrieve unique clusters
            # clusters = np.unique(yhat)
            #pdb.set_trace()
            print("yes front", len(relvnt_front_pnts))
            print("no left", len(relvnt_leftside_pnts))
            print("no right", len(relvnt_rightside_pnts))
            #pdb.set_trace()
            if len(relvnt_front_pnts) >= 3 :
                front_tv_loc = np.divide(relvnt_front_pnts_sum, front_num_points) #avergae of all the front points
                TVs_avg_loc.append([front_tv_loc, 'front'])
            else:
                self._last_frenet_tvs['front'] = 0

            #clustering algorithm for one left and right side 
            if len(relvnt_leftside_pnts) > 3 :
                left_tv_loc = np.divide(relvnt_leftside_pnts_sum, left_num_points) #avergae of all the left points
                TVs_avg_loc.append([left_tv_loc, 'left'])
            else:
                self._last_frenet_tvs['left'] = 0

            if len(relvnt_rightside_pnts) > 3 :
                right_tv_loc = np.divide(relvnt_rightside_pnts_sum, right_num_points) #avergae of all the right points
                pnt_loc = carla.Location(x=right_tv_loc[0], y=right_tv_loc[1], z=right_tv_loc[2])
                wp = self._map.get_waypoint(pnt_loc, project_to_road=True,
                                        lane_type=(carla.LaneType.Driving))
                if wp.lane_id != self.wp_ev.lane_id:
                    TVs_avg_loc.append([right_tv_loc, 'right'])
                    # if right_tv_loc[0] < -20:
                    #     pdb.set_trace()
            else:
                self._last_frenet_tvs['right'] = 0
            #storing previous xi values and getting frenet state for each TV
            for car in range(len(TVs_avg_loc)):
                #print("first frenet after ev")
                #pdb.set_trace()
                frenet = xy2frenet_pnt_specific(self._vehicle, self._last_frenet_tvs[TVs_avg_loc[car][1]], self._time_step, TVs_avg_loc[car][0], self._map, self._waypoint_buffer, self._sampling_radius)
                #pdb.set_trace()
                self._tvs.append(frenet) #first iteration is not accuarate becasue of lack of previous 

                self._last_frenet_tvs[TVs_avg_loc[car][1]] = self._tvs[car][0] #just xi (each term is in TV form)
                self._TV_locs.append(TVs_avg_loc[car][0]) #store all the TV locations
                #pdb.set_trace()
        
        
        
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
        #if self._tvs:
            #self._vehicle_controller.set_constraints(self.cstrf())

        # -------------------------------------------------------------------------
        # Getting control from MPC controller, define manual control if desired
        # ------------------------------------------------------------------------

        if log:
            if timestep / 30 > 8 and timestep / 30 < 9 and self.manual_control_on:
                # Setting manual control
                manual_control = [self.data_log.get('u_acceleration'), 0.01]
                target_wps.append(manual_control)
                print(manual_control)

                # apply manual control
                control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps, self.getqs, self._vehicle_controller.state,
                                                                                          self._target_speed, 
                                                                                          solve_nmpc=False, manual=True, log=log, debug=False) ####
                self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                                 'Eta': state[5],
                                 'Theta': state[6],
                                 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                 'pred_states': [x_log],
                                 'pred_control': [u_log], 'computation_time': time.time() - start_time,
                                 "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

            elif timestep / 30 > 14 and timestep / 30 < 14.6 and self.manual_control_on:
                # Setting manual control
                manual_control = [self.data_log.get('u_acceleration'), -0.02]
                target_wps.append(manual_control)

                # apply manual control
                control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps, self.getqs, self._vehicle_controller.state,
                                                                                          self._target_speed,
                                                                                          solve_nmpc=False,
                                                                                          manual=True, log=log, debug=False) ###
                self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3],
                                 'Xi': state[4],
                                 'Eta': state[5],
                                 'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                 'pred_states': [x_log],
                                 'pred_control': [u_log], 'computation_time': time.time() - start_time,
                                 "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

            else:
                if timestep % 6 == 0:
                    status , control, state, u, u_log, x_log, _ = self._vehicle_controller.mpc_control(target_wps, self.getqs, self._vehicle_controller.state, target_speed = self._target_speed, solve_nmpc= True, log=log, debug=False)###
                    # Updating logging information of the logger
                    self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                                     'Eta': state[5],
                                     'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                     'pred_states': [x_log],
                                     'pred_control': [u_log], 'computation_time': time.time() - start_time, "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}

                else:
                    control, state, prediction, u = self._vehicle_controller.mpc_control(target_wps, self.getqs, self._vehicle_controller.state, self._target_speed, solve_nmpc=False, log=log, debug=False)###
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
            #print("steering angle", self.data_log['u_steering_angle'])


        ####MPC control will also have to modified to sensor data
        else:
            if timestep % 6 == 0:
                control = self._vehicle_controller.mpc_control(target_wps, self.getqs, self._vehicle_controller.state, self._target_speed,solve_nmpc=True, log=False)
            else:
                control,_,  _, _ = self._vehicle_controller.mpc_control(target_wps,self.getqs, self._vehicle_controller.state, self._target_speed,
                                                                            solve_nmpc=False, log=log)

        # Print State information on TVs in realtion to EV
        if printer:
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
        :return: [p0, p1, p2, p3] of the 3th polynomial. Also returns Kappa (vehicle related data)
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
        #print("popt", p_opt)
        return p_opt, kappa_log


    def visualizeKappa2Carla(self, p_opt): #drawing the lines one sees in simulation

        # Rotation matrix to align current waypoint to x-axis
        angle_wp = get_wp_angle(self.wp1, self.wp2)
        R_inv = np.linalg.inv(rotmat(angle_wp))
        R_inv = inv_rotmat(angle_wp) #redundant?

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

    def dynamics_TV(self, x, u, T):
        #x: array, xi, eta, V_xi, V_eta
        #u: array, accel, speed
        #returns next state in TV form
        

        # A = [[1, T, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, T],
        #     [0, 0, 0, 1] 
        #     ]

        # B = [[0.5*(T ** 2), 0],
        #     [T, 0],
        #     [0, 0.5*(T ** 2)],
        #     [0, T]]

        # a = np.array(A)
        # b = np.array(B)

        first_term = np.matmul(self.A, x)
        second_term = np.matmul(self.B, u)
        
        next_state = first_term + second_term

        return next_state


    def Us_TV (self, xk, xref):
        #returns just next input 
        #xk: array, xi, V_xi, eta, V_eta
        #xref: 0, 0, tv_speed, 0
        diff_array = []
        if len(xk) == len(xref):
            for i in range(len(xk)):
                diff_array.append(xk[i] - xref[i]) #dxi, deta, dv_xi, dv_eta
        else:
            print("index size not matching")
        #self.K = np.array([[0, -0.55, 0, 0],[0, 0, -0.63, -1.15]]) #from the paper
        u_next = np.matmul(self.K, diff_array)
        return u_next # longitudinal acc, horizontal acc


    def getqs(self, N):
        """
        Returning the coefficients used in the constraint inequalities 
        Also utilizes the stochastic aspect
        """
        
        #U (array): input sequence
        #x0 (array): initial state
        #N (int): num of dimensions for steps
        #qx (array): coeff of case225
        # case225 = np.array([0, 1, 0, -0.5 * tv_state[5]])  #car on left
        # #qy (array): coeff of case220
        # case220 = np.array([0, -1, 0, 0.5*tv_state[5]]) #car on right
        # #qz (array): coeff of case21121
        # case21121 = np.array([-Delta * closest_front_vehicle[3], 0, 1,(-closest_front_vehicle[4] + min_dist + car_dim['length'])]) 
        # #qt (array): coeff of 210
        # case210 = np.array([-Delta * tv_state[3], 0, 1,- tv_state[4] + min_dist + car_dim['length']]) #car in front
        # order: x[11], x[5], x[4], const
        #based on conditions add or remove constrains
       # print("U", U)
        #print("N", N)


        #start_time = time.time()

        tv_state_dim = 4  


        qt = np.zeros((len(self._tvs), N, tv_state_dim))
        #ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)
        #x0 = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)[3:7] 
        # for i in range(N):
        #     qt[0][i][0] = 0 #eta
        #     qt[0][i][1] = 1 #x[5]
        #     qt[0][i][2] = 0 #x[4]
        #     if i == 0:
        #         qt[0][i][3] = -x0[2]
        #         print("first eta:", qt[0][i][3])
        #     else:
        #         qt[0][i][3] = (-0.1 + qt[0][i-1][3]) #previous time step eta 
        #         print("previous eta", qt[0][i][3])

        # return qt

        # for i in range(N):
        #     qt[0][i][0] = 0 #eta
        #     qt[0][i][1] = -1 #x[5]
        #     qt[0][i][2] = 0 #x[4]
        #     qt[0][i][3] = 2.5
        # return qt
        ##########################################

        

        qt_stay = np.zeros((len(self._tvs), N, tv_state_dim)) # initialize list of constraints
        qt_change = np.zeros((len(self._tvs), N, tv_state_dim))
        
        self.T = 0.2 #sampling time

        # vehicle size
        # length_var = self.stoch_bubble()[-1][0][0] #last timestep variance in xi
        # print("length_var=", length_var)
        car_dim = {'width': 2, 'length': 6}

        # Setting min distance to half of the current speed
        min_dist = 0.2 * get_speed(self._vehicle) #OG 0.5 0.2 did not work
        Delta = 0.2
        
        #cond_lane_dict = {}
        #cond_array = [] #to feed into lcon later 
        car_ahead = None #initialize as no car in front
        #plus1_lane = [] #holds indexes
        #minus1_lane = [] #holds indexes
        current_tv_idx = 0 #for indexing to decide which cond to eliminate for passing

        max_surveilance_rad = 20 #20 meters due to xi's limit in determining xi values

        constraints = []

        if self._tvs: ##remember this guy!        
            wp_ev = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True,
                                        lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius) #changing to frenet coordinates
            #free_lane4change = [wp_ev.lane_id + 1, wp_ev.lane_id - 1] #for knowing which lane is free to use for overtaking 
            right_lane_free = True
            left_lane_free = True
            #closest_front_vehicle = None

            all_tv_states = [0] * len(self._tvs)
            print("number of tvs", len(self._tvs))
            bob = carla.Location(x=-20.1351, y=-136.132, z=0.0)
            print("lane", self._map.get_waypoint(bob, project_to_road=True,
                                        lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk)).lane_id)

            for TV in range(len(self._tvs)): #change this for sensors implementation!
                ##with direct carla tv wp data
                #x0 = xy2frenet_wp(TV, self._map, self._waypoint_buffer, self._sampling_radius)[3:7] #changing to frenet coordinates   #0,1,2,3 ---- 3,4,5,6 index conversion #CURRENT state (EV form)
                #x0s = [x0[1], x0[0] * np.cos(x0[3]), x0[2], x0[0] * np.sin(x0[3])] #array: xi, speed, eta, V_eta (TV FORM) ####
                #with lidar data
                x0 = self._tvs[TV] #dict in TV form

                print("current car:", current_tv_idx, "x0:", x0)
                x0s = x0 #array: xi, speed, eta, V_eta (TV FORM) ####
                
                tv_states = [x0s] #subarray: xi, V_xi, eta, V_eta (TV FORM) SEQUENCE
                for k in range(N):
                    TV_ref = [0, x0[3], 0, 0] ####make dynamic
                    if k == 0:
                        tv_states.append(self.dynamics_TV(tv_states[-1], [0,0], self.T)) #initial control inputs
                    else:
                        tv_states.append(self.dynamics_TV(tv_states[-1], self.Us_TV(tv_states[-2], TV_ref), self.T)) 

                all_tv_states[current_tv_idx] = tv_states

                #ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius) #changing to frenet coordinates
               
                tv_locat = carla.Location(x= self._TV_locs[TV][0], y=self._TV_locs[TV][1], z=self._TV_locs[TV][2])
                print('TV:', self._TV_locs[TV][0], self._TV_locs[TV][1], self._TV_locs[TV][2])

                tv_transf = carla.Transform(location = tv_locat)
                #pdb.set_trace() #check location
                wp_tv = self._map.get_waypoint(tv_locat, project_to_road=True,
                                        lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

                print('TV:', TV, 'ev lane: ', wp_ev.lane_id, 'tv_lane_id: ', wp_tv.lane_id)
                #print('ev:', wp_ev, 'TV:', wp_tv)
                #signs flipped because switch from casadi to scipy
                # qz = np.array([-Delta * closest_front_vehicle[3], 0, 1,(-closest_front_vehicle[4] + min_dist + car_dim['length'])]) #lane change case21121

                if euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location) <= max_surveilance_rad or x0[1] < 30: #if within a certain radius of the ev HARD CODED FOR NOW ########### bc max range on eta calc
                    lane_diff = wp_ev.lane_id - wp_tv.lane_id
                    #print("distance:", euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location))
                    #### because there is no lane zero
                    #pdb.set_trace() #there seems to be an issue with lane_id when coming from behind
                    if wp_ev.lane_id == 1 & wp_tv.lane_id == -1 and "left constraint" not in constraints:
                        for k in range(N):
                            qt_stay[current_tv_idx][k][0] = 0
                            qt_stay[current_tv_idx][k][1] = -1
                            qt_stay[current_tv_idx][k][2] = 0
                            qt_stay[current_tv_idx][k][3] = 0.5*tv_states[k][2] #eta
                             #case 225
                            
                        print("case 225 added")
                        
                        constraints.append("left constraint")
                        #minus1_lane.append(current_tv_idx)

                        #check if either lanes have tvs that are just behind the ev (for lane changing)
                        if x0[1] >= -2*car_dim['length']:
                            left_lane_free = False

                        
                        
                    elif wp_ev.lane_id == -1 & wp_tv.lane_id == 1 and "right constraint" not in constraints:
                        for k in range(N):
                            qt_stay[current_tv_idx][k][0] = 0
                            qt_stay[current_tv_idx][k][1] = 1
                            qt_stay[current_tv_idx][k][2] = 0
                            qt_stay[current_tv_idx][k][3] = -0.5*tv_states[k][2]#case 220
                        print("case 220 added")
                        
                        constraints.append("right unq constraint")

                        #plus1_lane.append(current_tv_idx)
                        #check if either lanes have tvs that are just behind the ev (for lane changing)
                        if x0[1] >= -2*car_dim['length']:
                            right_lane_free = False
                    ####
                            
                    
                    if lane_diff == 1 and "left constraint" not in constraints:
                        #minus1_lane.append(current_tv_idx)
                        for k in range(N):
                            qt_stay[current_tv_idx][k][0] = 0
                            qt_stay[current_tv_idx][k][1] = -1
                            qt_stay[current_tv_idx][k][2] = 0
                            qt_stay[current_tv_idx][k][3] = 0.5*tv_states[k][2]
                             #case 225
                        print("case 225 added")
                        constraints.append("left constraint")
                        #check if either lanes have tvs that are just behind the ev (for lane changing)
                        if x0[1] >= -2*car_dim['length']:
                            left_lane_free = False
                            print("updated left lane")
                    elif lane_diff == -1 and "right constraint" not in constraints:
                        #plus1_lane.append(current_tv_idx)
                        for k in range(N):
                            qt_stay[current_tv_idx][k][0] = 0
                            qt_stay[current_tv_idx][k][1] = 1
                            qt_stay[current_tv_idx][k][2] = 0
                            qt_stay[current_tv_idx][k][3] = -0.5*tv_states[k][2] #case 220
                        print("case 220 added")
                        #check if either lanes have tvs that are just behind the ev (for lane changing)
                        if x0[1] >= -2*car_dim['length']:
                            right_lane_free = False
                            print("updated right lane")
                        constraints.append("right constraint")



                    elif lane_diff == 0: #tv is in same lane
                        for k in range(N):
                            qt_stay[current_tv_idx][k][0] = -Delta * tv_states[k][1] #speed
                            qt_stay[current_tv_idx][k][1] = 0
                            qt_stay[current_tv_idx][k][2] = -1
                            qt_stay[current_tv_idx][k][3] = tv_states[k][0] - min_dist - car_dim['length']  #case 210  #xi
                            
                        print("case210 added")
                        constraints.append("front constraint")
                        #pdb.set_trace()
                        # if closest_front_vehicle != None:
                        #     print("invalid index WHY", closest_front_vehicle[0], tv_states[0])
                        if tv_states[0][0] > 0:
                            car_ahead = current_tv_idx # this is fine because it is only one car in front
                            print("car ahead added")

                            
                        # if closest_front_vehicle == None:#check if not defined or if this tv is closer than previously tv that set this bound
                        #     closest_front_vehicle = x0s
                        #     print("not none", closest_front_vehicle)
                        #     print("updated closest car")
                        
                        # elif closest_front_vehicle[0] > tv_states[0][0]:
                        #     closest_front_vehicle = tv_states[0]
                        #     print("updated closest car")

                current_tv_idx += 1

            # def cond_clean(conditions, takeout):
            #     #for taking out desired TV's constraints 

            #     #cleaned = np.ndarray.tolist(conditions)
            #     cleaned = conditions
            #     #list_conditions = np.ndarray.tolist(conditions)
            #     #list_takeout= np.ndarray.tolist(takeout)
            #     print("clenaed", cleaned)
            #     #pdb.set_trace()
            #     for cond in takeout:
            #         for i in range(N): #to take out all time steps of a tv's constraints
            #             #print("ind", i)
            #             cleaned.pop(cond * N)
            #     return cleaned

            # if plus1_lane:
            #     plus1_lane.sort(reverse=True) #so indexes dont get messed up when i pop them
            # if minus1_lane:
            #     minus1_lane.sort(reverse=True) #so the indexes dont get messed up


            #print("ev yaw", ev_state[6])
            #print("lane stay case added")
            #pdb.set_trace()
            if car_ahead != None:
                print("CAR AHEAD!! BEGIN CHANGE LANES") #immediately wehn condition is applied it steers out of control. I think min dist is too high

                print("EV (xi, eta):", ev_state[4:6])
                
                if right_lane_free == True or left_lane_free == True: #either lanes free to use to overtake?
                    if right_lane_free == True:
                        print("right lane free")
                        #Delta * closest_front_vehicle[3], 0, 1,(-closest_front_vehicle[4] + min_dist + car_dim['length'])
                        #case21121 = np.array([0,-1, (closest_front_vehicle[4] - 2 * car_dim['length'])/(2.5* car_dim['width'] + closest_front_vehicle[5] - ev_state[5]), (ev_state[5]-car_dim['width'])])
                        for s in range(N):
                            qt_change[car_ahead][s][0] =  0 #kap
                            qt_change[car_ahead][s][1] =  -2.2*(all_tv_states[car_ahead][s][0] - ev_state[4]- car_dim['length']) / (car_dim['width']) #2.3 as overall worked well
                            qt_change[car_ahead][s][2] =  -1  #x[4] #xi , eta, eta
                            qt_change[car_ahead][s][3] =  all_tv_states[car_ahead][s][0] - 2.5* car_dim['length'] #const #coeff og 3.2
                        #cond_array = np.array([e.all() for e in cond_array if e.all() not in minus1_lane]) 
                        #pdb.set_trace()
                        # cond_clean(cineq, plus1_lane)
                        # print("removed right lane constraints")
                        constraints.append("right change constraint")
                    elif left_lane_free == True:
                        print("left lane free")

                        for s in range(N):
                            qt_change[car_ahead][s][0] =  0 #kap
                            qt_change[car_ahead][s][1] =  2.5*(all_tv_states[car_ahead][s][0] - ev_state[4] - car_dim['length']) / (1*car_dim['width']) #x[5]
                            qt_change[car_ahead][s][2] =  -1 
                            qt_change[car_ahead][s][3] =  all_tv_states[car_ahead][s][0] - 2.5* car_dim['length']#const eta
                        #print("tv_state", all_tv_states[car_ahead][0])
                        constraints.append("left change constraint")
                    print("qt_ change", qt_change)
                    #the problem is being able to tell which array of lane cond to remove
                        # cond_clean(cineq, minus1_lane)
                        # print("removed left lane constraints")
                        #cond_array = np.array([e.all() for e in cond_array if e.all() not in plus1_lane])
                    #cond_array = [e.all() for e in cond_array if e.all() not in car_ahead]
                    # cond_clean(cineq, car_ahead)
                    # print("removed front constraints")
                    print("returned qt_change")
                    print(constraints)
                    return qt_change
        print("returned qt_stay", qt_stay)
        print(constraints)
        #print("--- %s seconds ---" % (time.time() - start_time))

        #pdb.set_trace()
        return qt_stay # return constraints in array form (instead of list)
    # def u_retrieve():
    # def u_retrieve():
    #     x0 = pdb.set_trace

    #     cstr = [{'type': 'ineq', 'fun': lambda U: self.cstrf(U, x0, N)}]
    #     # U0 must be the guess of the input sequence (m*N x 1 array)
    #     # x0 must be a 4 dim array with the initial condition of the EV (only the four features we are concerned with)
    #     m = self._Nu
    #     Q = self._Q
    #     R = self._R
    #     S = self._S
    #     u_1 = self._last_control
    #     print("last_control", u_1)
    #     #num of timesteps
    #     N = 10

    #     #x prev
    #     xref = [80/3.6,0,0,0]

    #     #U0
    #     if not self.guess:
    #         U0 = np.zeros(N * m)
    #     else:
    #         U0 = self.guess
    #     res = opt.minimize(costf, U0, args=(N, x0, xref, u_1, Q, R, S, m),
    #                 method="SLSQP", constraints=cstr)
    #     u = res.x.reshape(-1, 1)
    #     return u
   
# #!/usr/bin/env python3

# #
# # authors: Michael Seegerer (michael.seegerer@tum.de)
# #
# # This work is licensed under the terms of the MIT license.
# # For a copy, see <https://opensource.org/licenses/MIT>.

# """ This module contains a model predictive controller to perform low-level waypoint following. """
# #performs MPC step calc, polynomial fit, and condition creation/application
# import glob
# import os
# import sys
# import time

# from enum import Enum
# from collections import deque
# import random
# import numpy as np
# from termcolor import colored
# import mpctools as mpc
# from scipy.optimize import curve_fit

# try:
#     # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#     #     sys.version_info.major,
#     #     sys.version_info.minor,
#     #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#     import carla

#     # from agents.navigation.controller import VehiclePIDController
#     from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed, is_within_distance_ahead
# except ImportError:
#     print("Carla libary is not installed !!")

# from mpcCARLA.road_aligned_mpc import CurvMPCController, converting_mpc_u_to_control, wrap2pi
# from mpcCARLA.waypoint_utilities import *


# class VehicleCurvMPC(object):
#     """
#     Model Predictive Controller implements the basic behavior of following a trajectory of waypoints that is generated
#      on-the-fly.

#     When multiple paths are available (intersections) this controller makes a random choice.
#     """

#     MIN_DISTANCE_PERCENTAGE = 0.9
#     #define all paramters in the class Vehicle Cruve MPC
#     def __init__(self, vehicle, tvs=dict(), opt_dict=None):
#         self._vehicle = vehicle
#         self._map = self._vehicle.get_world().get_map()

#         self._dt = None
#         self._target_speed = None
#         self._sampling_radius = None
#         self._min_distance = None
#         self._current_waypoint = None
#         self._target_road_option = None
#         self._last_control = np.array([0, 0])
#         self._opt_dict = opt_dict
#         self.curv_args = np.array([0, 0, 0, 0])
#         self.curv_x0 = 0
#         self.desired_lane_id = 0

#         # Option for the MPC controller
#         self.manual_control_on = False

#         # list of other target vehicles in the secenario
#         self._tvs = tvs


#         # queue with tuples of (waypoint, RoadOption)
#         self._waypoints_queue = deque(maxlen=30)
#         self._buffer_size = 21
#         self._waypoint_buffer = deque(maxlen=self._buffer_size)
#         self._passed_waypoints = deque(maxlen= 15)
#         self._waypoint_buffer_mpc = None

#         # define size of states, inputs, etc., for Casadi
#         self.Nx = 12 # Number of states
#         self.Nu = 2  # Number of Inputs (steer and acceleration)
#         self.Nt = 10  # Number of Steps

#         # initializing controller
#         self._init_controller(opt_dict)

#     def __del__(self):
#         if self._vehicle:
#             self._vehicle.destroy()
#         print("Destroying ego-vehicle!")

#     def reset_vehicle(self):
#         self._vehicle = None
#         print("Resetting ego-vehicle!")

#     def _init_controller(self, opt_dict):
#         """
#         Controller initialization.

#         :param opt_dict: dictionary of arguments.
#         :return:
#         """
#         # default params
#         #self._dt = 1.0 / 30.0
#         self._dt = 0.2
#         self._target_speed = 80.0  # Km/h
#         self._sampling_radius = calculate_step_distance(self._target_speed, self._dt,
#                                                         factor=1)  # factor 11 --> prediction horizon 10 steps
#         self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

#         self.data_log = dict()

#         # parameters overload
#         if opt_dict:
#             if 'dt' in opt_dict:
#                 self._dt = opt_dict['dt']
#             if 'target_speed' in opt_dict:
#                 self._target_speed = opt_dict['target_speed']
#             if 'sampling_radius' in opt_dict:
#                 self._sampling_radius = calculate_step_distance(self._target_speed, opt_dict['sampling_radius'],
#                                                                 factor=5)  # factor 11 --> prediction horizon 10 steps

#         self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#         self.desired_lane_id = self._current_waypoint.lane_id #stores initial lane

#         # fill waypoint trajectory queue
#         self._waypoints_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=200)

#         # Set Vehicle controller
#         state_dim = {'Nx': self.Nx, 'Nu': self.Nu, 'Nt': self.Nt}
#         self._vehicle_controller = CurvMPCController(vehicle=self._vehicle, dt=self._dt, args_state_dimension=state_dim)

#     def set_speed(self, speed=None):
#         """
#         Request new target speed.

#         :param speed: new target speed in Km/h
#         :return:
#         """
#         if speed is None:
#             speed = self.current_traffic_speed_limit
#         self._target_speed = speed

#     @property
#     def current_traffic_speed_limit(self):
#         return self._vehicle.get_speed_limit()

#     @property
#     def changing_lane(self):
#             return self._current_waypoint.lane_id - self.desired_lane_id

#     def get_state(self):
#         return self._vehicle_controller.state

#     def set_lane_change(self, lane_change: int):
#         """
#         Trigger a lane change of the ego vehicle. Therefore, the desired lane id of the controller will be increased
#         or decreased by the integer number in lane_change.
#         If desired_lane_id and current lane id of ego vehicle dont match, controller will trigger the future waypoints
#         on the desired lane id.

#         :param lane_change: integer number between [-1, 1]
#         :return:
#         """
#         self.desired_lane_id = self.desired_lane_id - lane_change



#     def run_step(self,timestep:int,  debug=True, log=False, print=True):
#         """
#         Execute one step of classic mpc controller which follow the waypoints trajectory.

#         :param debug: boolean flag to activate waypoints debugging
#         :return:
#         """

#         start_time = time.time()

#         # Update target velocity to current speed limit
#         self.set_speed()

#         # Trigger a lane change
#         # if timestep / 30 == 12:
#         #     self.set_lane_change(-1)

#         # ----------------------------------------------------------------
#         # Updating reference wp line --> only needed if nmpc is solved
#         # ----------------------------------------------------------------
#         if timestep % 6 == 0 or True:
#             # not enough waypoints in the horizon? => Sample new ones
#             self._sampling_radius = calculate_step_distance(get_speed(self._vehicle), 0.2, factor=1)
#             if self._sampling_radius < 2:
#                 self._sampling_radius = 3



#             ###compute waypoints will have to be modified into using sensor data######
#             # Getting future waypoints
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=15, stay_on_lane=True, active_lane_change=self.changing_lane) #returns list
#             # Getting waypoint history --> history somehow starts at last wp of future wp (previous waypoints)
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self._previous_wp_queue = compute_previous_waypoints(self._current_waypoint, d=self._sampling_radius, k=5, stay_on_lane=True, active_lane_change=self.changing_lane)

#             # concentrate history, current waypoint, future
#             self._waypoint_buffer = deque(maxlen=self._buffer_size)
#             self._waypoint_buffer.extendleft(self._previous_wp_queue)
#             self._waypoint_buffer.append((self._map.get_waypoint(self._vehicle.get_location()), RoadOption.LANEFOLLOW))
#             self._waypoint_buffer.extend(self._next_wp_queue)

#             self._waypoints_queue = self._next_wp_queue

#             # target waypoint for Frenet calculation
#             self.wp1 = self._map.get_waypoint(self._vehicle.get_location())
#             self.wp2, _ = self._next_wp_queue[0]

#             # Flipping x-axis of wp1 for calculation
#             # self.wp1.transform.location.x = -1 * self.wp1.transform.location.x

#             # calculation of kappa --> only needed if nmpc problem is solved.
#             self.kappa_log = dict()
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self.curv_args, self.kappa_log = self.calculate_curvature_func_args(log=True) #gets the fitted poly
#             self.curv_x0 = func_kappa(0, self.curv_args)

#             # input for MPC controller --> wp_current, wp_next, kappa
#             target_wps = [self.wp1, self.wp2, self.curv_x0]
#             target_wps2 = [self.wp1, self.wp2, self.curv_args]

#         # If no more waypoints in queue, returning emergency braking control
#         if len(self._waypoints_queue) == 0:
#             control = carla.VehicleControl()
#             control.steer = 0.0
#             control.throttle = 0.0
#             control.brake = 1.0
#             control.hand_brake = False
#             control.manual_gear_shift = False
#             print("Applied emergency control !!!")

#             return control

#         # Update the constraints for the current scene
#         if self._tvs:
#             self._vehicle_controller.set_constraints(self.get_constraints())

#         # -------------------------------------------------------------------------
#         # Getting control from MPC controller, define manual control if desired
#         # ------------------------------------------------------------------------

#         if log:
#             if timestep / 30 > 8 and timestep / 30 < 9 and self.manual_control_on:
#                 # Setting manual control
#                 manual_control = [self.data_log.get('u_acceleration'), 0.01]
#                 target_wps.append(manual_control)
#                 print(manual_control)

#                 # apply manual control
#                 control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
#                                                                                           self._target_speed, 
#                                                                                           solve_nmpc=False, manual=True, log=log, debug=False) ####
#                 self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                  'Eta': state[5],
#                                  'Theta': state[6],
#                                  'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                  'pred_states': [x_log],
#                                  'pred_control': [u_log], 'computation_time': time.time() - start_time,
#                                  "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

#             elif timestep / 30 > 14 and timestep / 30 < 14.6 and self.manual_control_on:
#                 # Setting manual control
#                 manual_control = [self.data_log.get('u_acceleration'), -0.02]
#                 target_wps.append(manual_control)

#                 # apply manual control
#                 control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
#                                                                                           self._target_speed,
#                                                                                           solve_nmpc=False,
#                                                                                           manual=True, log=log, debug=False) ###
#                 self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3],
#                                  'Xi': state[4],
#                                  'Eta': state[5],
#                                  'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                  'pred_states': [x_log],
#                                  'pred_control': [u_log], 'computation_time': time.time() - start_time,
#                                  "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

#             else:
#                 if timestep % 6 == 0:
#                     status , control, state, u, u_log, x_log, _ = self._vehicle_controller.mpc_control(target_wps2, self._target_speed, solve_nmpc=True, log=log, debug=False)###
#                     # Updating logging information of the logger
#                     self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                      'Eta': state[5],
#                                      'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                      'pred_states': [x_log],
#                                      'pred_control': [u_log], 'computation_time': time.time() - start_time, "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}

#                 else:
#                     control, state, prediction, u = self._vehicle_controller.mpc_control(target_wps,self._target_speed, solve_nmpc=False, log=log, debug=False)###
#                     # Updating logging information of the logger
#                     self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                      'Eta': state[5],
#                                      'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                      'pred_states': [prediction],
#                                      'pred_control': self.data_log.get("pred_control"), 'computation_time': time.time() - start_time,
#                                      "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}
#                 self.data_log["velocity_error"] = state[3] - self._target_speed / 3.6
#                 # self.data_log['kappa_state'] = state[9]
#                 self.data_log['target_velocity'] = self._target_speed
#         else: ####MPC control will also have to modified to sensor data
#             if timestep % 6 == 0:
#                 control = self._vehicle_controller.mpc_control(target_wps, self._target_speed,solve_nmpc=True, log=False)
#             else:
#                 control,_,  _, _ = self._vehicle_controller.mpc_control(target_wps, self._target_speed,
#                                                                          solve_nmpc=False, log=log)

#         # Print State information on TVs in realtion to EV
#         if print:
#             for tv_name, tv in self._tvs.items():
#                 print(colored(tv_name+' current state:'+
#                             str(xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)) +
#                             '   euclidean distance: '+
#                             str(distance_vehicle(self.wp1, tv.get_transform())), 'white', 'on_red'))

#         if debug and timestep % 6 == 0:
#             prediction_location = []
#             current_location = self._vehicle.get_location()
#             for i in range(self.Nt):
#                 loc = carla.Location(x=-1 * x_log[i, 0], y=x_log[i, 1], z=current_location.z + 0.5)
#                 prediction_location.append(loc)

#             draw_prediction_trajectory(self._vehicle.get_world(), prediction_location)
#             last_wp, _ = self._waypoint_buffer[len(self._waypoint_buffer)-1]
#             draw_waypoints_debug(self._vehicle.get_world(), [
#                 self.wp1, self.wp2], self._vehicle.get_location().z + 1.0)
#             draw_waypoints_debug(self._vehicle.get_world(), [
#                 last_wp], self._vehicle.get_location().z + 1.0, color=(0, 255, 0))
#             self.visualizeKappa2Carla(self.curv_args)


#         return control, self.data_log, self.kappa_log if log else control


#     def calculate_curvature_func_args(self, log=False):
#         """
#         Function to calculate the curvature function arguments.
#         This is done by fitting all reference waypoint from the waypoint buffer to 3th-polynomial function.
#         :return: [p0, p1, p2, p3] of the 3th polynomial. Also returns Kappa (vehicle related data)
#         """
#         kappa_log = dict()
#         # Current waypoint as reference point to center all waypoints
#         ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])

#         # Rotation matrix to align current waypoint to x-axis
#         angle_wp = get_wp_angle(self.wp1, self.wp2)
#         R = rotmat(angle_wp)

#         # Getting reduced waypoint matrix
#         wp_mat = np.zeros([self._buffer_size, 2])
#         wp_mat_0 = np.zeros([self._buffer_size, 2])  # saving original locations of wps for debug

#         counter_mat = 0
#         for i in range(self._buffer_size):
#             wp = self._waypoint_buffer[i][0]
#             loc = get_localization_from_waypoint(wp)
#             point = np.array([loc.x,  loc.y])  # init point
#             point = point.T - ref_point.T  # center traj to origin
#             point = mpc.mtimes(R, point)  # rotation of point ot allign with x-axis

#             wp_mat_0[counter_mat] = np.array([loc.x, loc.y])
#             wp_mat[counter_mat] = point
#             counter_mat += 1

#         # Getting optimal parameters for 3th polynomial by fitting the a curve
#         # Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#         # p_opt, _ = curve_fit(polynomial3, wp_mat[:, 0], wp_mat[:, 1])
#         p_opt = np.polynomial.polynomial.polyfit(wp_mat[:, 0], wp_mat[:, 1], deg=3)
#         # Logging all necessary information
#         if log:
#             kappa_log = {
#                 'wp_mat': [wp_mat],
#                 'wp_mat_0': [wp_mat_0],
#                 'refernce_point': [ref_point],
#                 'rotations_mat': [R],
#                 'angle_wp': [angle_wp],
#                 'p_opt': [p_opt]
#             }

#         return p_opt, kappa_log


#     def visualizeKappa2Carla(self, p_opt): #drawing the lines one sees in simulation

#         # Rotation matrix to align current waypoint to x-axis
#         angle_wp = get_wp_angle(self.wp1, self.wp2)
#         R_inv = np.linalg.inv(rotmat(angle_wp))
#         R_inv = inv_rotmat(angle_wp) #redundant?

#         # Current waypoint as reference point to center all waypoints
#         ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])


#         # Define polynomial function
#         p = np.poly1d(p_opt[-1::])
#         p = np.poly1d(p_opt)

#         # Sammpling points on refernce curve
#         number_sample = 100
#         number_hist_samples = 10
#         ref_curve_points = np.zeros([number_sample + number_hist_samples,2])
#         for k in range(number_hist_samples):
#             x= -2*(number_hist_samples - k)
#             y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
#             point = np.array([x, y])


#             # Roation
#             point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
#             # Translation
#             point = point.T + ref_point.T  # center traj to origin
#             # Converting Point into Carla coordination system
#             ref_curve_points[k] = np.array([-1*point[0], point[1]])


#         for k in range(number_sample):
#             x= 2*k
#             y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
#             point = np.array([x, y])


#             # Roation
#             point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
#             # Translation
#             point = point.T + ref_point.T  # center traj to origin
#             # Converting Point into Carla coordination system
#             ref_curve_points[k+number_hist_samples] = np.array([-1*point[0], point[1]])

#         # Convert to a list carla location objects
#         ref_curve_location = []
#         current_location = self._vehicle.get_location()
#         for i in range(number_sample):
#             loc = carla.Location(x= ref_curve_points[i, 0], y=ref_curve_points[i, 1], z=current_location.z + 0.5)
#             ref_curve_location.append(loc)

#         draw_prediction_trajectory(self._vehicle.get_world(), ref_curve_location, color=[0, 255, 0], thickness=0.1) # Draw green line


#     def get_constraints(self):
#         """
#         Returning constraints inequality function based on current scene.
#         :return: constraints function
#         """

#         # define inequality constraint for disctance between TV and EV


#         # vehicle size
#         car_dim = {'width': 2, 'length': 6}


#         # Case 210: one TV vehicle in front is consider for constraints --> vertical xi constraint
#         # Case 222: case 210 + one TV on right lane of EV consider in constraints
#         # case 21121: case 210 but without vertical linear constraint, instead linear inclined constraint to overtake TV is used
#         case = 21121


#         # Setting min distance to half of the current speed
#         min_dist = 0.5 * get_speed(self._vehicle)
#         Delta = 0.2


#         # cond_array = [] #to feed into lcon later
#         # closest_front_vehicle = None
        
#         if self._tvs: ##remember this guy!
#         # ############################################################################# Time to make a general case for TVs
#         #     for TV in self._tvs: #change this for sensors implementation!
#         #         tv_state = xy2frenet_wp(self._tvs[TV], self._map, self._waypoint_buffer, self._sampling_radius) #changing to frenet coordinates
#         #         ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)

#         #         ###########
#         #         x = np.zeros(15) #dummy temporary instantiators 
#         #         s = np.zeros(15) #dummy temporary instantiators 
#         #         case210 = - x[5] + (0.5 * tv_state[5]) -s[1]
#         #         case220 = x[4] - (tv_state[4] + x[11] * Delta * tv_state[3] - min_dist - car_dim['length']) - s[0]
#         #         case21121 = - x[5] + ((2.5* car_dim['width'] + tv_state[5] - ev_state[5]) / (tv_state[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1]
#         #         ###########

#         #         # Getting nearest waypoint to the TVs (with wp the current lane number of the TV can be easily evaluated)
#         #         wp_ev = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True,
#         #                                    lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
#         #         wp_tv = self._map.get_waypoint(self._tvs[TV].get_location(), project_to_road=True,
#         #                                    lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

                
                
#         #         if euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location) <= 30: #if within a certain radius of the ev
#         #             if abs(wp_ev.lane_id - wp-tv.lane_id) == 1 #check if the tv is on the adjacent lanes 
#         #                 #apply case 220 but only car on the right on the tv
#         #                 if case220 not in cond_array:
#         #                     cond_array.append(case220)



#         #             elif abs(wp_ev.lane_id - wp-tv.lane_id) == 0: #tv is in same lane
#         #                 #apply case 210 to the tv
#         #                 if case210 not in cond_array:
#         #                     cond_array.append(case210)

#         #                 if not closest_front_vehicle | closest_front_vehicle
#         #                 #check if not defined or if this tv is closer than previously tv that set this bound
#         #                     closest_front_vehicle = tv_state

#         #                 if tv_state[5] > 20 & :
#         #                     #if tvs arent in the way lane change When the 

#         #                 #figure out when to add in 

            




#         #     lcon = lambda x, s: mpc.vcat(cond_array)
            
#             ###########################################
#             # if case == 210:
#             #      dim_lin_cond = 1

#             #      # upper bound function --> equals TV dimension + 2m safety area around vehicle
#             #      ub_cond['xi'] = lambda t: mpc.vcat([tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      # soft constraint function to ensure the desired min distance --> x[11] is a counter state for the discrete step of the optimization problem
#             #      lcon = lambda x, s: mpc.vcat([x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s]) #smaller box?

#             # if case == 220:
#             #      dim_lin_cond = 2
#             #      ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[1]) * (car_dim['width'] + 1)])
#             #      lcon = lambda x, s: mpc.vcat([  x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                                      - x[5] + (0.5 * tv2_state[5]) -s[1]])


#             ############################################################################# original
#             # Getting frenet state representation for TV1, TV2 and EV
#             tv1_state = xy2frenet_wp(self._tvs['TV1'], self._map, self._waypoint_buffer, self._sampling_radius)
#             tv2_state = xy2frenet_wp(self._tvs['TV2'], self._map, self._waypoint_buffer, self._sampling_radius)
#             ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)

#             print("ETA OF tv1= ", tv1_state[4])
#             print("//////////////")
#              # Getting nearest waypoint to the TVs (with wp the current lane number of the TV can be easily evaluated)
#             wp_tv1 = self._map.get_waypoint(self._tvs['TV1'].get_location(), project_to_road=True,
#                                             lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
#             wp_tv2 = self._map.get_waypoint(self._tvs['TV2'].get_location(), project_to_road=True,
#                                             lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

#             ###################################################################
#             # Init upper and lower bounds for frenet state space
#             ub_cond = dict()
#             lb_cond = dict()
#             lb_cond['eta'] = lambda t: mpc.vcat([t * 0 - np.inf])
#             ub_cond['xi'] = lambda t: mpc.vcat([t * 0  + np.inf])

#             take_over_counter = 0


#             # if case == 210:
#             #      dim_lin_cond = 1

#             #      # upper bound function --> equals TV dimension + 2m safety area around vehicle
#             #      ub_cond['xi'] = lambda t: mpc.vcat([tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      # soft constraint function to ensure the desired min distance --> x[11] is a counter state for the discrete step of the optimization problem
#             #      lcon = lambda x, s: mpc.vcat([x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s]) #smaller box?

#             # if case == 220:
#             #      dim_lin_cond = 2
#             #      ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[1]) * (car_dim['width'] + 1)])
#             #      lcon = lambda x, s: mpc.vcat([  x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                                      - x[5] + (0.5 * tv2_state[5]) -s[1]])

#             if case == 21121:
#                 dim_lin_cond = 2
#                 # makes problems by overtaking --> by lane switch a big initial eta is not avoidable
#                 #lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[5]) * (car_dim['width'] + 1)])

#                 # soft constraint function to ensure the overtaking by a linear constraint --> x[11] is a counter state for the discrete step of the optimization problem, makes constraint time varying
                
#                 lcon = lambda x, s: mpc.vcat([  #x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#                                                  - x[5] + (0.75 * tv2_state[5]) -s[0],  # eta constraint for TV2
#                                                  - x[5] + ((2.5* car_dim['width'] + tv1_state[5] - ev_state[5]) / (tv1_state[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1] # overtaking constraint for TV1
#                                                  ])

#                                                 ####
#                 # if EV is not in the same lane as TV1 anymore, reinit constraint without overtaking constraint
#                 if abs(wp_tv1.lane_id) > abs(self.wp1.lane_id):
#                     take_over_counter += 1
#                     lcon = lambda x, s: mpc.vcat(
#                         [  # x[4] - (tv1_state[0] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#                             - x[5] + (0.5 * tv2_state[5]) - s[0],
#                             #- x[5] + (car_dim['width']) - s[0],
#                             #- x[5] + (0.5 * car_dim['width']) - s[1],
#                             #- x[5] + (tv1_state[5] - np.sign(tv1_state[5]) * car_dim['width']) - s[1],   # at lane change EV does not fullfill condition, leading to a rapid lane change
#                             s[1]
#                         ])
            
#             # Dict with all the conditions (state space + time-varying)
#             cond = {'x_bounds': ub_cond, 'linear_cond': lcon, 'x_bounds_low': lb_cond, 'dim_lin_cond': dim_lin_cond}

#         return cond

#     def get_constraints2(self):
#         """
#         Returning constraints inequality function based on current scene.
#         :return: constraints function
#         """

#         # define inequality constraint for disctance between TV and EV

#         # vehicle size
#         car_dim = {'width': 2, 'length': 6}

#         min_dist = 0.5 * get_speed(self._vehicle)
#         Delta = 0.2

#         self._tv_states = {}

#         if self._tvs:
#             for tv_name, tv in self._tvs.items():
#                 if "TV1" in tv_name:
#                     self._tv_states[tv_name] = xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)

#                     wp_tv = self._map.get_waypoint(tv.get_location(), project_to_road=True,
#                                                    lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

#                     # Case B --> TV on same lane and before EV
#                     if self.wp1.lane_id == wp_tv.lane_id and self._tv_states[tv_name][0] > 0:
#                         ub_cond = lambda t: mpc.vcat([self._tv_states[tv_name][0] + t * Delta *
#                                                       self._tv_states[tv_name][3] - car_dim['length'] - 2])
#                         lcon = lambda x, s: mpc.vcat([x[4] - (
#                                     self._tv_states[tv_name][0] + x[11] * Delta * self._tv_states[tv_name][
#                                 3] - min_dist - car_dim['length']) - s])

#                     # if self.wp1.lane_id != wp_tv.lane_id:
#                     #     diff_lane = abs(self.wp1.lane_id) - abs(wp_tv.lane_id)
#                     #
#                     #     if

#             cond = {'x_bounds': ub_cond, 'linear_cond': lcon}

#         return cond


#!/usr/bin/env python3

#
# authors: Michael Seegerer (michael.seegerer@tum.de)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# """ This module contains a model predictive controller to perform low-level waypoint following. """
# #performs MPC step calc, polynomial fit, and condition creation/application
# import glob
# import os
# from signal import SIG_DFL
# import sys
# import time

# from enum import Enum
# from collections import deque
# import random
# import numpy as np
# from termcolor import colored
# import mpctools as mpc
# from scipy.optimize import curve_fit
# import pdb

# try:
#     # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#     #     sys.version_info.major,
#     #     sys.version_info.minor,
#     #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#     import carla

#     # from agents.navigation.controller import VehiclePIDController
#     from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed, is_within_distance_ahead
# except ImportError:
#     print("Carla libary is not installed !!")

# from mpcCARLA.road_aligned_mpc import CurvMPCController, converting_mpc_u_to_control, wrap2pi
# from mpcCARLA.waypoint_utilities import *


# class VehicleCurvMPC(object):
#     """
#     Model Predictive Controller implements the basic behavior of following a trajectory of waypoints that is generated
#      on-the-fly.
#     When multiple paths are available (intersections) this controller makes a random choice.
#     """

#     MIN_DISTANCE_PERCENTAGE = 0.9
#     #define all paramters in the class Vehicle Cruve MPC
#     def __init__(self, vehicle, tvs=dict(), opt_dict=None):
#         self._vehicle = vehicle
#         self._map = self._vehicle.get_world().get_map()

#         self._dt = None
#         self._target_speed = None
#         self._sampling_radius = None
#         self._min_distance = None
#         self._current_waypoint = None
#         self._target_road_option = None
#         self._last_control = np.array([0, 0])
#         self._opt_dict = opt_dict
#         self.curv_args = np.array([0, 0, 0, 0])
#         self.curv_x0 = 0
#         self.desired_lane_id = 0

#         # Option for the MPC controller
#         self.manual_control_on = False

#         # list of other target vehicles in the secenario
#         self._tvs = tvs


#         # queue with tuples of (waypoint, RoadOption)
#         self._waypoints_queue = deque(maxlen=30)
#         self._buffer_size = 21
#         self._waypoint_buffer = deque(maxlen=self._buffer_size)
#         self._passed_waypoints = deque(maxlen= 15)
#         self._waypoint_buffer_mpc = None

#         # define size of states, inputs, etc., for Casadi
#         self.Nx = 12 # Number of states
#         self.Nu = 2  # Number of Inputs (steer and acceleration)
#         self.Nt = 10  # Number of Steps

#         # initializing controller
#         self._init_controller(opt_dict)

#     def __del__(self):
#         if self._vehicle:
#             self._vehicle.destroy()
#         print("Destroying ego-vehicle!")

#     def reset_vehicle(self):
#         self._vehicle = None
#         print("Resetting ego-vehicle!")

#     def _init_controller(self, opt_dict):
#         """
#         Controller initialization.
#         :param opt_dict: dictionary of arguments.
#         :return:
#         """
#         # default params
#         #self._dt = 1.0 / 30.0
#         self._dt = 0.2
#         self._target_speed = 80.0  # Km/h
#         self._sampling_radius = calculate_step_distance(self._target_speed, self._dt,
#                                                         factor=1)  # factor 11 --> prediction horizon 10 steps
#         self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

#         self.data_log = dict()

#         # parameters overload
#         if opt_dict:
#             if 'dt' in opt_dict:
#                 self._dt = opt_dict['dt']
#             if 'target_speed' in opt_dict:
#                 self._target_speed = opt_dict['target_speed']
#             if 'sampling_radius' in opt_dict:
#                 self._sampling_radius = calculate_step_distance(self._target_speed, opt_dict['sampling_radius'],
#                                                                 factor=5)  # factor 11 --> prediction horizon 10 steps

#         self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#         self.desired_lane_id = self._current_waypoint.lane_id #stores initial lane

#         # fill waypoint trajectory queue
#         self._waypoints_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=200)

#         # Set Vehicle controller
#         state_dim = {'Nx': self.Nx, 'Nu': self.Nu, 'Nt': self.Nt}
#         self._vehicle_controller = CurvMPCController(vehicle=self._vehicle, dt=self._dt, args_state_dimension=state_dim)

#     def set_speed(self, speed=None):
#         """
#         Request new target speed.
#         :param speed: new target speed in Km/h
#         :return:
#         """
#         if speed is None:
#             speed = self.current_traffic_speed_limit
#         self._target_speed = speed

#     @property
#     def current_traffic_speed_limit(self):
#         return self._vehicle.get_speed_limit()

#     @property
#     def changing_lane(self):
#             return self._current_waypoint.lane_id - self.desired_lane_id

#     def get_state(self):
#         return self._vehicle_controller.state

#     def set_lane_change(self, lane_change: int):
#         """
#         Trigger a lane change of the ego vehicle. Therefore, the desired lane id of the controller will be increased
#         or decreased by the integer number in lane_change.
#         If desired_lane_id and current lane id of ego vehicle dont match, controller will trigger the future waypoints
#         on the desired lane id.
#         :param lane_change: integer number between [-1, 1]
#         :return:
#         """
#         self.desired_lane_id = self.desired_lane_id - lane_change



#     def run_step(self, timestep:int,  debug=True, log=False, print=True):
#         """
#         Execute one step of classic mpc controller which follow the waypoints trajectory.
#         :param debug: boolean flag to activate waypoints debugging
#         :return:
#         """

#         start_time = time.time()

#         # Update target velocity to current speed limit
#         self.set_speed()

#         # Trigger a lane change
#         # if timestep / 30 == 12:
#         #     self.set_lane_change(-1)

#         # ----------------------------------------------------------------
#         # Updating reference wp line --> only needed if nmpc is solved
#         # ----------------------------------------------------------------
#         if timestep % 6 == 0 or True:
#             # not enough waypoints in the horizon? => Sample new ones
#             self._sampling_radius = calculate_step_distance(get_speed(self._vehicle), 0.2, factor=1)
#             if self._sampling_radius < 2:
#                 self._sampling_radius = 3



#             ###compute waypoints will have to be modified into using sensor data######
#             # Getting future waypoints
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=15, stay_on_lane=True, active_lane_change=self.changing_lane) #returns list
#             # Getting waypoint history --> history somehow starts at last wp of future wp (previous waypoints)
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self._previous_wp_queue = compute_previous_waypoints(self._current_waypoint, d=self._sampling_radius, k=5, stay_on_lane=True, active_lane_change=self.changing_lane)

#             # concentrate history, current waypoint, future
#             self._waypoint_buffer = deque(maxlen=self._buffer_size)
#             self._waypoint_buffer.extendleft(self._previous_wp_queue)
#             self._waypoint_buffer.append((self._map.get_waypoint(self._vehicle.get_location()), RoadOption.LANEFOLLOW))
#             self._waypoint_buffer.extend(self._next_wp_queue)

#             self._waypoints_queue = self._next_wp_queue

#             # target waypoint for Frenet calculation
#             self.wp1 = self._map.get_waypoint(self._vehicle.get_location())
#             self.wp2, _ = self._next_wp_queue[0]

#             # Flipping x-axis of wp1 for calculation
#             # self.wp1.transform.location.x = -1 * self.wp1.transform.location.x

#             # calculation of kappa --> only needed if nmpc problem is solved.
#             self.kappa_log = dict()
#             self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
#             self.curv_args, self.kappa_log = self.calculate_curvature_func_args(log=True) #gets the fitted poly
#             self.curv_x0 = func_kappa(0, self.curv_args)

#             # input for MPC controller --> wp_current, wp_next, kappa
#             target_wps = [self.wp1, self.wp2, self.curv_x0]
#             target_wps2 = [self.wp1, self.wp2, self.curv_args]

#         # If no more waypoints in queue, returning emergency braking control
#         if len(self._waypoints_queue) == 0:
#             control = carla.VehicleControl()
#             control.steer = 0.0
#             control.throttle = 0.0
#             control.brake = 1.0
#             control.hand_brake = False
#             control.manual_gear_shift = False
#             print("Applied emergency control !!!")

#             return control

#         # Update the constraints for the current scene
#         if self._tvs:
#             self._vehicle_controller.set_constraints(self.get_constraints())

#         # -------------------------------------------------------------------------
#         # Getting control from MPC controller, define manual control if desired
#         # ------------------------------------------------------------------------

#         if log:
#             if timestep / 30 > 8 and timestep / 30 < 9 and self.manual_control_on:
#                 # Setting manual control
#                 manual_control = [self.data_log.get('u_acceleration'), 0.01]
#                 target_wps.append(manual_control)
#                 print(manual_control)

#                 # apply manual control
#                 control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
#                                                                                           self._target_speed, 
#                                                                                           solve_nmpc=False, manual=True, log=log, debug=False) ####
#                 self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                  'Eta': state[5],
#                                  'Theta': state[6],
#                                  'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                  'pred_states': [x_log],
#                                  'pred_control': [u_log], 'computation_time': time.time() - start_time,
#                                  "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

#             elif timestep / 30 > 14 and timestep / 30 < 14.6 and self.manual_control_on:
#                 # Setting manual control
#                 manual_control = [self.data_log.get('u_acceleration'), -0.02]
#                 target_wps.append(manual_control)

#                 # apply manual control
#                 control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
#                                                                                           self._target_speed,
#                                                                                           solve_nmpc=False,
#                                                                                           manual=True, log=log, debug=False) ###
#                 self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3],
#                                  'Xi': state[4],
#                                  'Eta': state[5],
#                                  'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                  'pred_states': [x_log],
#                                  'pred_control': [u_log], 'computation_time': time.time() - start_time,
#                                  "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

#             else:
#                 if timestep % 6 == 0:
#                     status , control, state, u, u_log, x_log, _ = self._vehicle_controller.mpc_control(target_wps2, self._target_speed, solve_nmpc=True, log=log, debug=False)###
#                     # Updating logging information of the logger
#                     self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                      'Eta': state[5],
#                                      'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                      'pred_states': [x_log],
#                                      'pred_control': [u_log], 'computation_time': time.time() - start_time, "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}

#                 else:
#                     control, state, prediction, u = self._vehicle_controller.mpc_control(target_wps,self._target_speed, solve_nmpc=False, log=log, debug=False)###
#                     # Updating logging information of the logger
#                     self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
#                                      'Eta': state[5],
#                                      'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
#                                      'pred_states': [prediction],
#                                      'pred_control': self.data_log.get("pred_control"), 'computation_time': time.time() - start_time,
#                                      "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}
#                 self.data_log["velocity_error"] = state[3] - self._target_speed / 3.6
#                 # self.data_log['kappa_state'] = state[9]
#                 self.data_log['target_velocity'] = self._target_speed
#             #print("steering angle", self.data_log['u_steering_angle'])


#         ####MPC control will also have to modified to sensor data
#         else:
#             if timestep % 6 == 0:
#                 control = self._vehicle_controller.mpc_control(target_wps, self._target_speed,solve_nmpc=True, log=False)
#             else:
#                 control,_,  _, _ = self._vehicle_controller.mpc_control(target_wps, self._target_speed,
#                                                                             solve_nmpc=False, log=log)

#         # Print State information on TVs in realtion to EV
#         if print:
#             for tv_name, tv in self._tvs.items():
#                 print(colored(tv_name+' current state:'+
#                             str(xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)) +
#                             '   euclidean distance: '+
#                             str(distance_vehicle(self.wp1, tv.get_transform())), 'white', 'on_red'))

#         if debug and timestep % 6 == 0:
#             prediction_location = []
#             current_location = self._vehicle.get_location()
#             for i in range(self.Nt):
#                 loc = carla.Location(x=-1 * x_log[i, 0], y=x_log[i, 1], z=current_location.z + 0.5)
#                 prediction_location.append(loc)

#             draw_prediction_trajectory(self._vehicle.get_world(), prediction_location)
#             last_wp, _ = self._waypoint_buffer[len(self._waypoint_buffer)-1]
#             draw_waypoints_debug(self._vehicle.get_world(), [
#                 self.wp1, self.wp2], self._vehicle.get_location().z + 1.0)
#             draw_waypoints_debug(self._vehicle.get_world(), [
#                 last_wp], self._vehicle.get_location().z + 1.0, color=(0, 255, 0))
#             self.visualizeKappa2Carla(self.curv_args)


#         return control, self.data_log, self.kappa_log if log else control


#     def calculate_curvature_func_args(self, log=False):
#         """
#         Function to calculate the curvature function arguments.
#         This is done by fitting all reference waypoint from the waypoint buffer to 3th-polynomial function.
#         :return: [p0, p1, p2, p3] of the 3th polynomial. Also returns Kappa (vehicle related data)
#         """
#         kappa_log = dict()
#         # Current waypoint as reference point to center all waypoints
#         ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])

#         # Rotation matrix to align current waypoint to x-axis
#         angle_wp = get_wp_angle(self.wp1, self.wp2)
#         R = rotmat(angle_wp)

#         # Getting reduced waypoint matrix
#         wp_mat = np.zeros([self._buffer_size, 2])
#         wp_mat_0 = np.zeros([self._buffer_size, 2])  # saving original locations of wps for debug

#         counter_mat = 0
#         for i in range(self._buffer_size):
#             wp = self._waypoint_buffer[i][0]
#             loc = get_localization_from_waypoint(wp)
#             point = np.array([loc.x,  loc.y])  # init point
#             point = point.T - ref_point.T  # center traj to origin
#             point = mpc.mtimes(R, point)  # rotation of point ot allign with x-axis

#             wp_mat_0[counter_mat] = np.array([loc.x, loc.y])
#             wp_mat[counter_mat] = point
#             counter_mat += 1

#         # Getting optimal parameters for 3th polynomial by fitting the a curve
#         # Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#         # p_opt, _ = curve_fit(polynomial3, wp_mat[:, 0], wp_mat[:, 1])
#         p_opt = np.polynomial.polynomial.polyfit(wp_mat[:, 0], wp_mat[:, 1], deg=3)
#         # Logging all necessary information
#         if log:
#             kappa_log = {
#                 'wp_mat': [wp_mat],
#                 'wp_mat_0': [wp_mat_0],
#                 'refernce_point': [ref_point],
#                 'rotations_mat': [R],
#                 'angle_wp': [angle_wp],
#                 'p_opt': [p_opt]
#             }

#         return p_opt, kappa_log


#     def visualizeKappa2Carla(self, p_opt): #drawing the lines one sees in simulation

#         # Rotation matrix to align current waypoint to x-axis
#         angle_wp = get_wp_angle(self.wp1, self.wp2)
#         R_inv = np.linalg.inv(rotmat(angle_wp))
#         R_inv = inv_rotmat(angle_wp) #redundant?

#         # Current waypoint as reference point to center all waypoints
#         ref_point = np.array([-1 * self.wp1.transform.location.x, self.wp1.transform.location.y])


#         # Define polynomial function
#         p = np.poly1d(p_opt[-1::])
#         p = np.poly1d(p_opt)

#         # Sammpling points on refernce curve
#         number_sample = 100
#         number_hist_samples = 10
#         ref_curve_points = np.zeros([number_sample + number_hist_samples,2])
#         for k in range(number_hist_samples):
#             x= -2*(number_hist_samples - k)
#             y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
#             point = np.array([x, y])


#             # Roation
#             point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
#             # Translation
#             point = point.T + ref_point.T  # center traj to origin
#             # Converting Point into Carla coordination system
#             ref_curve_points[k] = np.array([-1*point[0], point[1]])


#         for k in range(number_sample):
#             x= 2*k
#             y = polynomial3(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3])
#             point = np.array([x, y])


#             # Roation
#             point = mpc.mtimes(R_inv, point)  # rotation of point ot allign with x-axis
#             # Translation
#             point = point.T + ref_point.T  # center traj to origin
#             # Converting Point into Carla coordination system
#             ref_curve_points[k+number_hist_samples] = np.array([-1*point[0], point[1]])

#         # Convert to a list carla location objects
#         ref_curve_location = []
#         current_location = self._vehicle.get_location()
#         for i in range(number_sample):
#             loc = carla.Location(x= ref_curve_points[i, 0], y=ref_curve_points[i, 1], z=current_location.z + 0.5)
#             ref_curve_location.append(loc)

#         draw_prediction_trajectory(self._vehicle.get_world(), ref_curve_location, color=[0, 255, 0], thickness=0.1) # Draw green line


#     def get_constraints(self):
#         """
#         Returning constraints inequality function based on current scene.
#         :return: constraints function
#         """

#         # define inequality constraint for disctance between TV and EV


#         # vehicle size
#         car_dim = {'width': 2, 'length': 6}


#         # Case 210: one TV vehicle in front is consider for constraints --> vertical xi constraint
#         # Case 222: case 210 + one TV on right lane of EV consider in constraints
#         # case 21121: case 210 but without vertical linear constraint, instead linear inclined constraint to overtake TV is used
#         #case = 21121


#         # Setting min distance to half of the current speed
#         min_dist = 0.25 * get_speed(self._vehicle) #OG 0.5
#         Delta = 0.2

#         #cond_lane_dict = {}
#         cond_array = [] #to feed into lcon later 
#         car_ahead = [] #initialize as no car in front
#         plus1_lane = [] #holds indexes
#         minus1_lane = [] #holds indexes
#         current_tv_idx = 0 #for indexing to decide which cond to eliminate for passing

#         max_surveilance_rad = 20 #20 meters due to xi's limit in determining xi values

#         if self._tvs: ##remember this guy!
#         ############################################################################# Time to make a general case for TVs
#             ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)

#             wp_ev = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True,
#                                            lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
#             #free_lane4change = [wp_ev.lane_id + 1, wp_ev.lane_id - 1] #for knowing which lane is free to use for overtaking 
#             right_lane_free = True
#             left_lane_free = True
#             ahead_free = True
#             closest_front_vehicle = None
            

#             #print("tvs", self._tvs)
#             for TV in self._tvs.values(): #change this for sensors implementation!
#                 tv_state = xy2frenet_wp(TV, self._map, self._waypoint_buffer, self._sampling_radius) #changing to frenet coordinates\
#                 #ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)
#                 #print("tv:", TV.get_location())
#                 ###########
#                 #x = np.zeros(15) #dummy temporary instantiators 
#                 #s = np.zeros(15) #dummy temporary instantiators 
#                 #case220 = lambda x,s: - x[5] + (0.5 * tv_state[5]) -s[1] 
#                 #case220 = np.array([0, -1, 0, -1, 0, (0.5 * tv_state[5])]) #with S
#                 #case220 = tv_state[5]-x[5] > 0.5 * tv_state[5] #MY CONDITION (car on right)
#                 case220 = np.array([0, -1, 0, 0.5*tv_state[5]]) #my condition array 
#                 #case225 = tv_state[5]-x[5] < 0.5 * tv_state[5] #MY CONDITION (car on left)
#                 case225 = np.array([0, 1, 0, -0.5 * tv_state[5]])  #car on left
#                 #case220 = np.array([0, -1, 0, (0.5 * tv_state[5])]) #no S
#                 #case210 = lambda x,s: x[4] - (tv_state[4] + x[11] * Delta * tv_state[3] - min_dist - car_dim['length']) - s[0] #x[4] -tv_state[4] - x[11] *Delta * tv_state[3] + min_dist +car_dim['length']
#                 #case210 = np.array([-Delta * tv_state[3], 0, 1, 0, -1, tv_state[4] + min_dist + car_dim['length']]) #with S
#                 case210 = np.array([-Delta * tv_state[3], 0, 1,- tv_state[4] + min_dist + car_dim['length']]) #no S
#                 #case21121 = - x[5] + ((2.5* car_dim['width'] + tv_state[5] - ev_state[5]) / (tv_state[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1]
#                 ###########

#                 # Getting nearest waypoint to the TVs (with wp the current lane number of the TV can be easily evaluated)
#                 #wp_ev = self._map.get_waypoint(self._vehicle.get_location(), project_to_road=True,
#                                            #lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
#                 wp_tv = self._map.get_waypoint(TV.get_location(), project_to_road=True,
#                                            lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

                
#                 # print("ev lane id:", wp_ev.lane_id)
#                 # print("ev pos:", wp_ev.transform.location)

#                 # print("tv lane id:", wp_tv.lane_id)
#                 # print("tv pos:", wp_tv.transform.location)

#                 # print("1st tv within range:", euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location) <= max_surveilance_rad)
#                 #print("distance:", euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location))
#                 if euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location) <= max_surveilance_rad or tv_state[4] < 30: #if within a certain radius of the ev HARD CODED FOR NOW ########### bc max range on eta calc
#                     lane_diff = wp_ev.lane_id - wp_tv.lane_id
#                     print("distance:", euclidean_distance(self._vehicle.get_location(),wp_tv.transform.location))
#                     #### because there is no lane zero
#                     if wp_ev.lane_id == 1 & wp_tv.lane_id == -1:
#                         #if case220 not in cond_array:
#                         cond_array.append(case225)
#                         print("case 225 added")
#                         minus1_lane.append(current_tv_idx)
#                         #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         if tv_state[4] >= -1.5*car_dim['length']:
#                             left_lane_free = False
#                         # elif wp_tv == 2:
#                         #     if case220 not in cond_array:
#                         #         cond_array.append(case220)

#                         #         plus1_lane.append(case220)
#                         #         #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         #         if tv_state[4] >= -1.5*car_dim['length']:
#                         #             right_lane_free = False


                                
                        
#                     elif wp_ev.lane_id == -1 & wp_tv.lane_id == 1:
#                         # if wp_tv == -2:
#                         #     if case220 not in cond_array:
#                         #         cond_array.append(case220)
                                
#                         #         minus1_lane.append(case220)
#                         #         #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         #         if tv_state[4] >= -1.5*car_dim['length']:
#                         #             left_lane_free = False
#                         #if case220 not in cond_array:
#                         cond_array.append(case220)
#                         print("case 220 added")

#                         plus1_lane.append(current_tv_idx)
#                             #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         if tv_state[4] >= -1.5*car_dim['length']:
#                             right_lane_free = False
#                     ####
                            
                    
#                     if lane_diff == 1:
#                         minus1_lane.append(current_tv_idx)
#                         cond_array.append(case225)
#                         print("case 220 added")
#                         #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         if tv_state[4] >= -1.5*car_dim['length']:
#                             left_lane_free = False
#                             print("updated left lane")
#                     elif lane_diff == -1:
#                         plus1_lane.append(current_tv_idx)
#                         cond_array.append(case220)
#                         print("case 220 added")
#                         #check if either lanes have tvs that are just behind the ev (for lane changing)
#                         if tv_state[4] >= -1.5*car_dim['length']:
#                             right_lane_free = False
#                             print("updated right lane")



#                     elif lane_diff == 0: #tv is in same lane
#                         #apply case 210 to the tv
#                         #if case210 not in cond_array:
#                         cond_array.append(case210)
#                         print("case210 added")
#                         if tv_state[4] > 0:
#                             car_ahead.append(current_tv_idx) # 
#                             print("car ahead added")
#                         if not closest_front_vehicle:#check if not defined or if this tv is closer than previously tv that set this bound
#                             closest_front_vehicle = tv_state
#                             print("updated closest car")
#                         elif closest_front_vehicle[4] > tv_state[4]:
                        
#                             closest_front_vehicle = tv_state
#                             print("updated closest car")

#                 current_tv_idx += 1

#             def cond_clean(conditions, takeout):
#                 #cleaned = np.ndarray.tolist(conditions)
#                 cleaned = conditions
#                 #list_conditions = np.ndarray.tolist(conditions)
#                 #list_takeout= np.ndarray.tolist(takeout)
#                 for cond in takeout:
#                     cleaned.pop(cond)
#                 return cleaned

#             if plus1_lane:
#                 plus1_lane.sort(reverse=True) #so indexes dont get messed up when i pop them
#             if minus1_lane:
#                 minus1_lane.sort(reverse=True) #so the indexes dont get messed up
#             print("ev yaw", ev_state[6])
#             print("min dist:", Delta*tv_state[3]+min_dist)
#             if car_ahead:
#                 print("CAR AHEAD!! BEGIN CHANGE LANES") #immediately wehn condition is applied it steers out of control. I think min dist is too high
#                 #case21121 = lambda x,s: - x[5] + ((2.5* car_dim['width'] + closest_front_vehicle[5] - ev_state[5]) / (closest_front_vehicle[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1]
#                 #case21121 = np.array([0,-1, (closest_front_vehicle[4] - 2 * car_dim['length'])/(2.5* car_dim['width'] + closest_front_vehicle[5] - ev_state[5]), -1,0, (ev_state[5]-car_dim['width'])]) #nwith s
#                 #case21121 = np.array([0,-1, (closest_front_vehicle[4] - 2 * car_dim['length'])/(2.5* car_dim['width'] + closest_front_vehicle[5] - ev_state[5]), (ev_state[5]-car_dim['width'])]) #no S GOOD kinda
#                 #min_dist = (closest_front_vehicle[4]-car_dim['length']-min_dist-(Delta * closest_front_vehicle[3])) / 
#                 #minim_dist = (closest_front_vehicle[4]-car_dim['length']-min_dist-(Delta * closest_front_vehicle[3])) > (15 / 2) * closest_front_vehicle[5] + 15
#                 #case21121 = np.array([0, -15/2, 0, -car_dim['length']-min_dist-(Delta * closest_front_vehicle[3]) - 15+closest_front_vehicle[4]])
#                 #case21121 = np.array([0, 0, -1, -(car_dim['length']+min_dist-closest_front_vehicle[4])])
#                 #closest_front_vehicle[4] - x[4] -car_dim['length'] > x[11]*Delta* closest_front_vehicle[3] + min_dist
#                 case21121 = np.array([-Delta * closest_front_vehicle[3], 0, 1,(-closest_front_vehicle[4] + min_dist + car_dim['length'])]) 
#                 prev_state = xy2frenet_wp_specific(self._vehicle, self._map, self._previous_wp_queue[0] , self._sampling_radius)
#                 print("prev eta:", prev_state)
#                 case21121 = np.array([0, 1, 0, -(prev_state[5] - 0.2)])  #x[5] < self._previous_wp_queue[0] - 0.2
#                 print("tv to ev", closest_front_vehicle[4])
                
#                 if right_lane_free == True or left_lane_free == True: #either lanes free to use to overtake?
#                     if right_lane_free == True:
#                         print("right lane free")
#                         #cond_array = np.array([e.all() for e in cond_array if e.all() not in minus1_lane]) 
#                         cond_clean(cond_array, minus1_lane)
#                         print("removed right lane constraints")
#                     if left_lane_free == True:
#                         print("left lane free")
#                        #the problem is being able to tell which array of lane cond to remove
#                         cond_clean(cond_array, plus1_lane)
#                         print("removed left lane constraints")
#                         #cond_array = np.array([e.all() for e in cond_array if e.all() not in plus1_lane])
#                     #cond_array = [e.all() for e in cond_array if e.all() not in car_ahead]
#                     cond_clean(cond_array, car_ahead)
#                     print("removed front constraints")
#                     cond_array.append(case21121)
#                     print("case21121 added")

#             dim_lin_cond = len(cond_array) 
#             #if dim_lin_cond == 0:
#                 #dim_lin_cond = 1
            
#             #print("dim_lin_cond is Ns", dim_lin_cond)
#             print("cond_array:", cond_array)
            
#             #lcon = lambda x: np.matmul(mpc.vcat(cond_array), np.array([x[11], x[5], x[4], 1]).T)
#             # lcon = lambda x: np.matmul(np.vstack(cond_array), np.array([x[11], x[5], x[4], 1]).T)
#             lcon = lambda x: pdb.set_trace()# np.matmul(np.vstack(cond_array), np.array([x[11], x[5], x[4], 1]).T)

#             #print("cond_array:", cond_array)
            

#             ###########################################
#             # if case == 210:
#             #      dim_lin_cond = 1

#             #      # upper bound function --> equals TV dimension + 2m safety area around vehicle
#             #      ub_cond['xi'] = lambda t: mpc.vcat([tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      # soft constraint function to ensure the desired min distance --> x[11] is a counter state for the discrete step of the optimization problem
#             #      lcon = lambda x, s: mpc.vcat([x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s]) #smaller box?

#             # if case == 220:
#             #      dim_lin_cond = 2
#             #      ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[1]) * (car_dim['width'] + 1)])
#             #      lcon = lambda x, s: mpc.vcat([  x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                                      - x[5] + (0.5 * tv2_state[5]) -s[1]])


#             ############################################################################# original
#             # Getting frenet state representation for TV1, TV2 and EV
#             # tv1_state = xy2frenet_wp(self._tvs['TV1'], self._map, self._waypoint_buffer, self._sampling_radius)
#             # tv2_state = xy2frenet_wp(self._tvs['TV2'], self._map, self._waypoint_buffer, self._sampling_radius)
#             # ev_state = xy2frenet_wp(self._vehicle, self._map, self._waypoint_buffer, self._sampling_radius)


#             #  # Getting nearest waypoint to the TVs (with wp the current lane number of the TV can be easily evaluated)
#             # wp_tv1 = self._map.get_waypoint(self._tvs['TV1'].get_location(), project_to_road=True,
#             #                                 lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
#             # wp_tv2 = self._map.get_waypoint(self._tvs['TV2'].get_location(), project_to_road=True,
#             #                                 lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

#             ###################################################################
#             # Init upper and lower bounds for frenet state space
#             ub_cond = dict()
#             lb_cond = dict()
#             lb_cond['eta'] = lambda t: mpc.vcat([t * 0 - np.inf])
#             ub_cond['xi'] = lambda t: mpc.vcat([t * 0  + np.inf])

#             take_over_counter = 0


#             # if case == 210:
#             #      dim_lin_cond = 1

#             #      # upper bound function --> equals TV dimension + 2m safety area around vehicle
#             #      ub_cond['xi'] = lambda t: mpc.vcat([tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      # soft constraint function to ensure the desired min distance --> x[11] is a counter state for the discrete step of the optimization problem
#             #      lcon = lambda x, s: mpc.vcat([x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s]) #smaller box?

#             # if case == 220:
#             #      dim_lin_cond = 2
#             #      ub_cond['xi'] = lambda t: mpc.vcat([  tv1_state[4] + t * Delta * tv1_state[3] - car_dim['length'] - 2])
#             #      lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[1]) * (car_dim['width'] + 1)])
#             #      lcon = lambda x, s: mpc.vcat([  x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                                      - x[5] + (0.5 * tv2_state[5]) -s[1]])

#             # if case == 21121:
#             #     dim_lin_cond = 2
#             #     # makes problems by overtaking --> by lane switch a big initial eta is not avoidable
#             #     #lb_cond['eta'] = lambda t: mpc.vcat([tv2_state[5] - np.sign(tv2_state[5]) * (car_dim['width'] + 1)])

#             #     # soft constraint function to ensure the overtaking by a linear constraint --> x[11] is a counter state for the discrete step of the optimization problem, makes constraint time varying
                
#             #     lcon = lambda x, s: mpc.vcat([  #x[4] - (tv1_state[4] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                                      - x[5] + (0.75 * tv2_state[5]) -s[0],  # eta constraint for TV2
#             #                                      - x[5] + ((2.5* car_dim['width'] + tv1_state[5] - ev_state[5]) / (tv1_state[4] - 2 * car_dim['length']) * x[4] + (ev_state[5] - 1 * car_dim['width'])) - s[1] # overtaking constraint for TV1
#             #                                      ])

#             #                                     ####
#             #     # if EV is not in the same lane as TV1 anymore, reinit constraint without overtaking constraint
#             #     if abs(wp_tv1.lane_id) > abs(self.wp1.lane_id):
#             #         take_over_counter += 1
#             #         lcon = lambda x, s: mpc.vcat(
#             #             [  # x[4] - (tv1_state[0] + x[11] * Delta * tv1_state[3] - min_dist - car_dim['length']) - s[0],
#             #                 - x[5] + (0.5 * tv2_state[5]) - s[0],
#             #                 #- x[5] + (car_dim['width']) - s[0],
#             #                 #- x[5] + (0.5 * car_dim['width']) - s[1],
#             #                 #- x[5] + (tv1_state[5] - np.sign(tv1_state[5]) * car_dim['width']) - s[1],   # at lane change EV does not fullfill condition, leading to a rapid lane change
#             #                 s[1]
#             #             ])
            
#             # Dict with all the conditions (state space + time-varying)
#             cond = {'x_bounds': ub_cond, 'linear_cond': lcon, 'x_bounds_low': lb_cond, 'dim_lin_cond': dim_lin_cond}

#         return cond

#     def get_constraints2(self):
#         """
#         Returning constraints inequality function based on current scene.
#         :return: constraints function
#         """

#         # define inequality constraint for disctance between TV and EV

#         # vehicle size
#         car_dim = {'width': 2, 'length': 6}

#         min_dist = 0.5 * get_speed(self._vehicle)
#         Delta = 0.2

#         self._tv_states = {}

#         if self._tvs:
#             for tv_name, tv in self._tvs.items():
#                 if "TV1" in tv_name:
#                     self._tv_states[tv_name] = xy2frenet_wp(tv, self._map, self._waypoint_buffer, self._sampling_radius)

#                     wp_tv = self._map.get_waypoint(tv.get_location(), project_to_road=True,
#                                                    lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

#                     # Case B --> TV on same lane and before EV
#                     if self.wp1.lane_id == wp_tv.lane_id and self._tv_states[tv_name][0] > 0:
#                         ub_cond = lambda t: mpc.vcat([self._tv_states[tv_name][0] + t * Delta *
#                                                       self._tv_states[tv_name][3] - car_dim['length'] - 2])
#                         lcon = lambda x, s: mpc.vcat([x[4] - (
#                                     self._tv_states[tv_name][0] + x[11] * Delta * self._tv_states[tv_name][
#                                 3] - min_dist - car_dim['length']) - s])

#                     # if self.wp1.lane_id != wp_tv.lane_id:
#                     #     diff_lane = abs(self.wp1.lane_id) - abs(wp_tv.lane_id)
#                     #
#                     #     if

#             cond = {'x_bounds': ub_cond, 'linear_cond': lcon}

#         return cond
