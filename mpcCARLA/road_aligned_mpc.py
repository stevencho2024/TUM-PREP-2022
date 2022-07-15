""" This module contains PID controllers to perform lateral and longitudinal control. """
import os
import time

import numpy as np
import mpctools as mpc
from collections import deque
from termcolor import colored

import carla
from agents.tools.misc import distance_vehicle, get_speed
import mpcCARLA.waypoint_utilities as wp


def wrap2pi(rad: float):
    """
    Wrap angle in rad to [- pi, pi]
    :param rad: Input radians value
    :return rad_wrap btw [-pi , pi]
    """
    rad_wrap = rad % (2 * np.pi)
    if abs(rad_wrap) > np.pi:
        rad_wrap -= 2 * np.pi
    return rad_wrap


def converting_mpc_u_to_control(u: np.array, debug=False):
    # apply the acceleration input, acceleration input and brake
    # input are in CARLA two different control therefore it needs to be modified
    # additionally a normalization is done
    # acceleration is 0 to 1 so as braking

    Input_acceleration_EV = 0
    Input_brake_EV = 0
    Input_steering_EV = 0

    # u is changed from +-5 to +-1 (100% gas / brake pedal)
    if u[0] >= 0:
        Input_acceleration_EV = u[0] / 5
        Input_brake_EV = 0
    if u[0] < 0:
        Input_acceleration_EV = 0
        Input_brake_EV = u[0] / 5

    # steering is from -1 to 1 the maximum steering angle is 60 degrees
    # As before a normalization is done
    # Mirrored to meet the mirroring of the x-axis.
    Input_steering_EV = -1 * u[1]

    control = carla.VehicleControl(
        throttle=Input_acceleration_EV, steer=Input_steering_EV, brake=Input_brake_EV, )

    if debug:
        print("Applied Controll: ", control)

    return control


def uselastguess(var, prevguess, x0):
    if "status" not in var or var["status"] != "Solve_Succeeded":
        print('using last guess !!')
        return prevguess
    else:
        # Shifting the prediction entries --> u'_o = u_1, u'_10 = u_10
        guess_x = var['x'].copy()
        guess_u = var['u'].copy()


        guess_x[0:-1, :] = guess_x[1:, :]
        guess_x[-1, :] = guess_x[-2, :]

        guess_u[0:-1, :] = guess_u[1:, :]
        guess_u[-1, :] = guess_u[-2, :]
        return {"x" : guess_x, "u" : guess_u}


class MPCController:
    """
    MPCController is casADI based model predictive controller to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, dt=0.2,
                 args_state_dimension=None,
                 args_vehicle_param=None, ):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_state_dimension: dictionary of arguments to set the dimension of the MPC control problem using the
        following semantics:
                             Nx -- State dimensions
                             Nu -- Control dimensions
                             Nt -- Prediction horizon
        :param args_vehicle_param: dictionary of arguments to set the specific vehicle parameters using the following
        semantics:
                             l_r            -- Distance to the rear from the vehicle center of gravity
                             l_f            -- Distance to the front from the vehicle center of gravity
                             last_u         -- Last applied control to the vehicle.
                             target_speed   -- Target speed in Km/h.
        """
        if args_state_dimension is None:
            args_state_dimension = {'Nx': 6, 'Nu': 2, 'Nt': 5, 'Ne': 0, 'Ns': 1}
        if args_vehicle_param is None:
            args_vehicle_param = {'l_r': 1.9, 'l_f': 1.9, 'last_u': np.array([0, 0])}

        # Init of essential carla variables
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        # Init some essential variables
        self._state = None
        self._controller = False  # controller is not initialized
        self._target = self._map.get_waypoint(
            self._vehicle.get_location())
        self._dt = dt
        self._fps = 30

        self._heading0 = np.deg2rad(round(self._target.transform.rotation.yaw, 3))

        self.nclon_set = False

        # Some option to imporove optimization by using last guess
        self.use_lastguess = True
        self.guess = {}
        self.opt = {}

        # Init state dimensions for the control problem
        self._Nx = args_state_dimension.get('Nx')
        self._Nu = args_state_dimension.get('Nu')
        self._Nt = args_state_dimension.get('Nt')
        self._Ne = args_state_dimension.get('Ne')
        self._Ns = args_state_dimension.get('Ns')

        # Init vehicle parameters
        self._lr = args_vehicle_param.get('l_r')
        self._lf = args_vehicle_param.get('l_f')
        self._last_control = args_vehicle_param.get('last_u')
        self._target_speed = args_vehicle_param.get('target_speed')

    @property
    def state(self):
        """
        xy-position state of the ego vehicle.
        :return: [X, Y, PSI, Velocity(m/s), last_u1, last_u2]
        """
        vehicle_location = self._vehicle.get_location()
        return np.array([vehicle_location.x,  # X
                         vehicle_location.y,  # Y
                         wrap2pi(np.deg2rad(round(self._vehicle.get_transform().rotation.yaw, 3))),   # PSI [rad]
                         get_speed(self._vehicle) / 3.6,  # Velocity [m/s]
                         self._last_control[0],  # last u0
                         self._last_control[1],  # last u1
                         ])

    @property
    def target(self):
        """
        Returning target state array.
        :return: [X, Y, PSI, Velocity(m/s), 0, 0]
        """
        location = self._target.transform.location
        return np.array([location.x,  # X
                         location.y,  # Y
                         wrap2pi(np.deg2rad(round(self._target.transform.rotation.yaw, 3))),   # PSI [rad]
                         self._target_speed / 3.6,  # Velocity [m/s]
                         0,
                         0,
                         ])


    def set_constraints(self, func_constraints):
        """
        Setter function to define OCP solving constraints
        :param func_constraints: kappa function of the inequalities
        :return:
        """
        self._nlcon = func_constraints
        self._Ne = func_constraints['dim_lin_cond']
        self._Ns = func_constraints['dim_lin_cond']
        if self._Ns > 1:
            self._C = np.eye(self._Ns)


        #if isinstance(func_constraints(self.state), float) else len(func_constraints(self.state))

    def _init_weights(self):
        """
        Initialize weights for the cost function.
        :return:
        """
        # Define normalization weight matrices
        self._Qn = np.diag([0.1, 0.1, 5.0, 0.5])  # typical max values: d_v_x = 4 m/s, d_eta = 1.0 m, d_theta = 0.5 rad (~30deg)
        self._Rn = np.diag([0.2, 12.5])  # typical max values: a_x = 5 m/s, d_f = 0.4 rad
        self._Sn = np.diag([0.5, 20.0])  # typical max values: d_a_x = 2 m/s, d_d_f = 0.1 rad

        self._Qn = np.diag([1, 1, 1, 1])
        self._Rn = np.diag([1, 1])
        self._Sn = np.diag([1, 1])

        # Define weight matrix
        self._Q = np.diag([0.05, 0, 10, 1])
        self._R = np.diag([0.1, 0.01])
        self._S = np.diag([1, 1])

        self._Q = np.array([[0.4, 0, 0, 0],
                            [0,  0.4, 0, 0],
                            [0, 0, 5, 0],
                            [0, 0, 0, 4 * 0.25], ])
        self._R = np.array([[0.03, 0],
                            [0, 15], ])
        self._S = np.array([[0.33, 0],
                            [0, 4 * 15], ])
        self._C = np.diag(self._Ns)

    def _init_controller(self, args_mpc_functions=None):
        """
        Controller initialization.
        :return:
        """

        # Init weights for cost function
        self._init_weights()

        # Init lambda functions to get delta_x & delta_u
        self.x_diff = lambda x: x[:4] - self.target[:4]
        self.x_diff = lambda x: np.array([x[0] - self.target[0], x[1] - self.target[1], np.sqrt(x[2] ** 2)- np.sqrt(self.target[2] ** 2), x[3] - self.target[3]])
        self.u_diff = lambda x, u: u - x[-2:]

        # Defining lambda functions for alpha & dxi for system equation
        self.alpha = lambda u: np.arctan((self._lr / (self._lf + self._lr)) * np.tan(u[1]))

        if args_mpc_functions is None:
            args_mpc_functions = {'sys': lambda x, u: np.array(
                [self._dt * (x[3] * np.cos(x[2] + self.alpha(u))) + x[0],  # X
                 self._dt * (x[3] * np.sin(x[2] + self.alpha(u))) + x[1],  # Y
                 self._dt * ((x[3] / self._lr) * np.sin(self.alpha(u))) + x[2],  # psi
                 self._dt * (u[0]) + x[3],  # velocity
                 u[0],  # u0
                 u[1]  # u1
                 ]),
                                  'lfunc': lambda x, u: mpc.mtimes(self.x_diff(x).T, self._Qn, self._Q, self.x_diff(x))
                                                        + mpc.mtimes(u.T, self._Rn, self._R, u) +
                                                        mpc.mtimes(self.u_diff(x, u).T, self._Sn, self._S,
                                                                   self.u_diff(x, u))}

        self._sys = args_mpc_functions.get('sys')
        self._lfunc = args_mpc_functions.get('lfunc')

        self._controller = True

    # Define Model Predictive control with a linearized Kinematic Model
    def mpc_control(self, waypoint, target_speed=None,solve_nmpc=True, manual=False, last_u=None, log=False, debug=True):
        """
        Execute one step of control to reach given waypoint as closely as possible with a given target speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        start_time = time.time()

        # Choose between NMPC and LMPC to solve MPC Problem
        mpc_solver = 'NMPC'


        # Init controller if necessary
        if not self._controller:
            self._init_controller()

        # Getting initial state
        x0 = self.state

        # Updating the last applied control
        if last_u:
            self._last_control = last_u
        # Updating the target speed
        if target_speed:
            self._target_speed = target_speed

        # Updating target location
        if isinstance(waypoint, carla.Waypoint):
            self._target = waypoint

        # Load Model
        f = mpc.getCasadiFunc(self._sys, [self._Nx, self._Nu], ["x", "u"], "f")

        # Bounds on u.
        lb = {"u": np.array([-5, -0.5 - 0.1])}
        ub = {"u": np.array([5, 0.5 + 0.1])}
        upperbx = 10000 * np.ones((self._Nt + 1, self._Nx))

        # define stage cost
        l = mpc.getCasadiFunc(self._lfunc, [self._Nx, self._Nu, self._Ns], ["x", "u", "s"], "l")

        # Make optimizers
        funcargs = {"f": ["x", "u"], "l": ["x", "u", "s"]}  # Define pointer



        # Setting verbosity level of casADI solver output
        verbs = 0 if debug else 0

        # Setting the commonargs for the NMPC solver
        commonargs = dict(
            verbosity=verbs,
            l=l,
            x0=x0,
            Pf=None,
            lb=lb,
            ub =ub,
            uprev=self._last_control
        )

        # Build controller and adjust some ipopt options.
        solvers = {}
        Nnonlin = {"x": self._Nx, "u": self._Nu, "t": self._Nt, "s": self._Ns}

        Nlin = Nnonlin.copy()
        #Nnonlin["c"] = 2  # Use collocation to discretize.


        # Initialize inequality constraint
        if self._Ne > 0:
            # Init Bounds for x with -inf and inf.
            xub = np.inf * np.ones((self._Nt + 1, self._Nx))
            xlb = -xub

            # Setting upper bounds and lower bounds based on the parsed cond function for every discrete timestep of optimization
            xub[:, 4] = self._nlcon['x_bounds']['xi'](np.arange(0, self._Nt + 1, 1))
            xlb[:, 5] = self._nlcon['x_bounds_low']['eta'](np.arange(0, self._Nt + 1, 1))
            lb['x'] = xlb
            ub['x'] = xub


            # Setting the time varying constraints
            e = mpc.getCasadiFunc(self._nlcon['linear_cond'], [self._Nx, self._Ns], ["x", "s"], "e")
            funcargs['e'] = ['x', 's']
            Nnonlin['e'] = self._Ne
            commonargs['e'] = e

        if self.use_lastguess:
            self.guess = uselastguess(self.opt, self.guess, x0)
            commonargs['guess'] = self.guess

        commonargs['funcargs'] = funcargs
        # solvers['NMPC'] = mpc.nmpc(f, l, Nnonlin, x0, lb, ub, uprev=self._last_control,
        #                   funcargs=funcargs, Pf=None, verbosity=verbs)
        solvers['NMPC'] = mpc.nmpc(f=f, N=Nnonlin, **commonargs)
        # solvers["LMPC"] = mpc.nmpc(F, l, Nlin, x0, lb, ub, uprev=self._last_control,
        #                   funcargs=funcargs, Pf=None, verbosity=verbs)


        # Fix initial state.
        solvers[mpc_solver].fixvar("x", 0, x0)

        # Solve nlp.
        solvers[mpc_solver].solve()

        self.opt = solvers[mpc_solver].vardict
        self.opt["status"] = "Solve_Succeeded"


        if solvers[mpc_solver].stats["status"] != "Solve_Succeeded":
            status = False
            print(colored('WARNING: MPC problem was not successfully solved !', 'red'))
        else:
            status = True  # OCP got solved.

        x = np.squeeze(solvers[mpc_solver].var["x", 1])
        u = np.squeeze(solvers[mpc_solver].var["u", 0, :])


        if debug and False:
            print("---- print us -------")
            print(np.squeeze(solvers[mpc_solver].var["u", 0, :]))
            print(np.squeeze(solvers[mpc_solver].var["u", 1, :]))
            print(np.squeeze(solvers[mpc_solver].var["u", 2, :]))
            print(np.squeeze(solvers[mpc_solver].var["u", 3, :]))
            print(np.squeeze(solvers[mpc_solver].var["u", 4, :]))
            print("---- print prediction states -------")
            print(np.squeeze(        solvers[mpc_solver].var["x", 0]))
            print(np.squeeze(        solvers[mpc_solver].var["x", 1]))
            print(np.squeeze(        solvers[mpc_solver].var["x", 2]))
            print(np.squeeze(        solvers[mpc_solver].var["x", 3]))
            print(np.squeeze(        solvers[mpc_solver].var["x", 4]))
            print("-----General state information ------")
            print("Current state: \n", self.state)
            print("Target: \n", self.target)
            print("Delta state: \n", self.x_diff(self.state))

        # Loads predicted input and states into variables which can be given out with print
        u_data1 = np.squeeze(        solvers[mpc_solver].var["u", :, 0])
        u_data2 = np.squeeze(        solvers[mpc_solver].var["u", :, 1])
        u_log = np.array([u_data1, u_data2])
        t_log = 0
        x_log = np.zeros((self._Nt, self._Nx))
        for t_log in range(self._Nt):
            x_log[t_log, :] = np.squeeze(        solvers[mpc_solver].var["x", t_log])

        # reset last control
        self._last_control = u
        
        if debug or abs(self._last_control[1]) > 0.2:
            loc = carla.Location(-1 * x_log[self._Nt-1, 0], x_log[self._Nt-1, 1], 1)
            wp.draw_vehicle_bounding_box(self._world, loc, x_log[self._Nt-1, 2])
        
        if log:
            return status, converting_mpc_u_to_control(u, debug),self.state, self._last_control, u_log, x_log, time.time() - start_time
        else:
            return status, converting_mpc_u_to_control(u, debug)


class CurvMPCController(MPCController):
    """
    MPC controller with advanced curvature handling by using frenet states.
    This controller is based on the MPC controller definded direclty above.
    """

    def __init__(self, vehicle, dt=0.2,
                 args_state_dimension=None,
                 args_vehicle_param=None, ):
        super().__init__(vehicle)

        if args_state_dimension is None:
            args_state_dimension = {'Nx': 10, 'Nu': 2, 'Nt': 10}
        if args_vehicle_param is None:
            args_vehicle_param = {'l_r': 1.9, 'l_f': 1.9, 'last_u': np.array([0, 0])}

        # Init state dimensions for the control problem
        self._Nx = args_state_dimension.get('Nx')
        self._Nu = args_state_dimension.get('Nu')
        self._Nt = args_state_dimension.get('Nt')

        # Init vehicle parameters
        self._lr = args_vehicle_param.get('l_r')
        self._lf = args_vehicle_param.get('l_f')
        self._last_control = args_vehicle_param.get('last_u')
        self._target_speed = args_vehicle_param.get('target_speed')

        # Init current waypoint and next waypoint with dummy values
        self._wp_current = self._map.get_waypoint(self._vehicle.get_location())
        self._wp_next = self._wp_current.next(2)[0]
        self._kr = 0
        self._dt = dt

        # Error buffer for velocity, eta, theta
        self._eta_e_buffer = deque(maxlen=10)
        self._theta_e_buffer = deque(maxlen=10)
        self._velocity_e_buffer = deque(maxlen=10)
        self.velocity_error = lambda x: x[3] - self.target[0]

    def _init_weights(self):
        """
        Setting the weight matrices for the cost function.
        Weight matrices are taken from the MATLAB code.
        """
        # Define normalization weight matrices
        self._Qn = np.diag([0.25, 1, 5.0])  # typical max values: d_v_x = 4 m/s, d_eta = 1.0 m, d_theta = 0.2 rad
        self._Rn = np.diag([0.2, 1.25])  # typical max values: a_x = 5 m/s, d_f = 0.4 rad
        self._Sn = np.diag([0.5, 20.0])  # typical max values: d_a_x = 2 m/s, d_d_f = 0.1 rad
        self._Kn = np.diag([1, 1, 1])   # errors of the last 10/30 sec are collected

        # Define weight matrix
        self._Q = np.diag([3, 1, 25])
        self._R = np.diag([0.1, 0.01])
        self._S = np.diag([2, 2])
        self._Ki = np.diag([5, 0, 0])
        self._C = 1

    def update_curvature(self, kr):
        """
        Updating the current curvature value of the reference trajectory.
        :param kr: curvature value
        :return:
        """
        self._kr = kr

    @property
    def state(self):
        """
        frenet state of the ego vehicle.
        :return: [X, Y, PSI, Velocity(m/s), Xi, Eta, Theta,  last_u1, last_u2, kappa]
        """
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        angle_wp = wp.get_wp_angle(self._wp_current, self._wp_next)
        angle_xy = wp.get_angle2wp_line(vehicle_transform, self._wp_current, self._wp_next)
        eta =  np.sign(angle_xy) * wp.get_distance2wp(vehicle_transform, self._wp_current, self._wp_next)
        kappa = wp.func_kappa(0, self._kr)

        if eta is np.NaN:
            eta = 0


        vehicle_heading = wrap2pi(np.pi - np.deg2rad(round(self._vehicle.get_transform().rotation.yaw, 3)))

        return np.array([-1 * vehicle_location.x,  # X
                         vehicle_location.y,  # Y
                         vehicle_heading,  # PSI [rad]
                         get_speed(self._vehicle) / 3.6,  # Velocity [m/s]
                         0,  # xi
                         eta,  # eta
                         wrap2pi(vehicle_heading - angle_wp),  # theta
                         self._last_control[0],  # last u0
                         self._last_control[1],  # last u1
                         kappa, # kappa,
                         sum(self._velocity_e_buffer) ,     # error in velocity over last T=10 timesteps
                         #sum(self._eta_e_buffer),    # error in eta over last T=10 timesteps
                         #sum(self._theta_e_buffer),  # error in theta over last T=10 timesteps
                         0,     # discrete timestep for time-varing constraints
                         ])

    @property
    def target(self):
        """
        Returning target state array.
        :return: [Velocity(m/s), 0, 0]
        """
        location = self._target.transform.location
        return np.array([self._target_speed / 3.6,  # Velocity [m/s]
                         0,
                         0,
                         ])

    def _init_controller(self, args_mpc_functions=None):
        super(CurvMPCController, self)._init_controller()

        # Defining lambda function for delta_x
        self.x_diff = lambda x: np.array([x[3], x[5], x[6]]) - self.target
        self.u_diff = lambda x, u: u - np.array([x[7], x[8]])
        # Defining lambda function for delta_xi and alpha
        self.dxi = lambda x, u: (1 / (1 - x[9] * x[5])) * x[3] * np.cos(x[6] + self.alpha(u))
        self.alpha = lambda u: np.arctan((self._lr / (self._lf + self._lr)) * np.tan(u[1]))
        self.kappa = lambda x: wp.func_kappa(x,self._kr)
        self.velocity_error = lambda x:  self.target[0] - x[3]
        self.i_error = lambda x: np.array([x[10], x[11], x[12]])

        if args_mpc_functions is None:
            args_mpc_functions = {'sys': lambda x, u: np.array(
                [self._dt * (x[3] * np.cos(x[2] + self.alpha(u))) + x[0],  # X
                 self._dt * (x[3] * np.sin(x[2] + self.alpha(u))) + x[1],  # Y
                 self._dt * ((x[3] / self._lr) * np.sin(self.alpha(u))) + x[2],  # psi
                 self._dt * (u[0]) + x[3],  # velocity
                 self._dt * self.dxi(x, u) + x[4],  # xi
                 self._dt * (x[3] * np.sin(x[6] + self.alpha(u))) + x[5],  # eta
                 self._dt * ((x[3] / self._lr) * np.sin(self.alpha(u)) - x[9] * self.dxi(x, u)) + x[6],  # theta
                 u[0],  # u0
                 u[1],  # u1
                 wp.func_kappa(x[4], self._kr), # kappa
                 # wp.func_kappa(0, self._kr),  # kappa
                 (x[10] + self.velocity_error(x) * self._dt),   # velocity error
                 #(x[11] + x[5] * self._dt),  # ETA error
                 #(x[12] + x[6] * self._dt),  # Theta error
                 x[11] + 1
                 ]),
                                  'lfunc': lambda x, u, s: mpc.mtimes(self.x_diff(x).T, self._Qn, self._Q, self.x_diff(x))
                                                        + mpc.mtimes(u.T, self._Rn, self._R, u) +
                                                        mpc.mtimes(self.u_diff(x, u).T, self._Sn, self._S,
                                                                   self.u_diff(x, u))
                                                        #+mpc.mtimes(self.i_error(x).T,self._Kn, self._Ki, self.i_error(x))
                                                        + 3 * x[10] ** 2
                                                        +  1.0 * mpc.mtimes(s.T, self._C, s)
                                                        }


        self._sys = args_mpc_functions.get('sys')
        self._lfunc = args_mpc_functions.get('lfunc')

    def mpc_control(self, wp_input, target_speed=None, solve_nmpc=True, manual=False, last_u=None, log=True, debug=True):

        self._wp_current = wp_input[0]
        self._wp_next = wp_input[1]
        self.update_curvature(wp_input[2])
        self.start_x = -1 * self._vehicle.get_location().x


        # Updating the velocity, eta and theta error buffer
        self._velocity_e_buffer.append((target_speed - get_speed(self._vehicle)) / 3.6 / self._fps)
        self._eta_e_buffer.append(self.state[5] / self._fps)
        self._theta_e_buffer.append(self.state[6] / self._fps)

        if debug:
            vehicle_loc = self._vehicle.get_location()
            ego_vehicle_loc = carla.Location(x=-1 * vehicle_loc.x, y=vehicle_loc.y, z=0)
            os.system('clear')
            print('============= DEBUG INFORMATION ==========')
            print('-----Frenet values -------')
            print('=== ETA = {} | dist-EV-WP = {} | XY_ANGLE = {}'.format(
                self.state[5], wp.get_distance2wp(self._vehicle.get_transform(), self._wp_current, self._wp_next),
                wp.get_angle2wp_line(self._vehicle.get_transform(), self._wp_current, self._wp_next)))
            print('=== Theta = {} | PSI = {} | WP_ANGLE = {}'.format(
                self.state[6], self.state[2], wp.get_wp_angle(self._wp_current, self._wp_next)))
            print('=== WP Current = {} | WP Next = {} | Curvature = {} === '.format(
                self._wp_current, self._wp_next, self._kr))
            print('=== EV Location x = {} | Y = {} | VXY = {} | VWP = {}=== '.format(
                self.state[0], self.state[1], wp.get_wp_vector(ego_vehicle_loc, self._wp_next), wp.get_wp_vector(self._wp_current, self._wp_next)))

        # if manual is True, apply as manual control fourth
        if manual:
            manual_control_input = wp_input[3]
            control = converting_mpc_u_to_control(manual_control_input)
            prediction =np.zeros((self._Nt, self._Nx))
            pred_u = np.zeros((2, 10))
            for i in range(self._Nt):
                if i == 0:
                    current_state = self.state
                else:
                    current_state = prediction[i-1, :]
                prediction[i] = self._sys(current_state, manual_control_input)
                pred_u[0, i] = manual_control_input[0]
                pred_u[1, i] = manual_control_input[1]

            return control, self.state, manual_control_input, prediction, pred_u, self._last_control


        if solve_nmpc:
            return super().mpc_control(wp_input, target_speed, solve_nmpc, manual, last_u, log, debug)
        else:

            prediction =np.zeros((self._Nt, self._Nx))
            for i in range(self._Nt):
                if i == 0:
                    current_state = self.state
                else:
                    current_state = prediction[i-1, :]
                prediction[i] = self._sys(current_state, self._last_control)

            return converting_mpc_u_to_control(self._last_control, debug), self.state, prediction, self._last_control
