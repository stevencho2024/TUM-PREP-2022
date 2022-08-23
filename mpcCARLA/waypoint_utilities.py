"""
    Utility functionalities for handling waypoint for agents classes.
"""
from collections import deque
from enum import Enum
from multiprocessing.dummy import Array
import random
import numpy as np
import math
import pdb

try:
    import carla
    from agents.navigation.controller import VehiclePIDController
    from agents.tools.misc import distance_vehicle, draw_waypoints, vector, is_within_distance_ahead, get_speed
except ImportError:
    print("Carla library is not installed !!")


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


def rotmat(angle_rad):
    """
    Computes a 2x2 rotation matrix
    :param angle_rad: Rotation angle in rad
    :return: 2x2 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, s],
                     [-s, c]])

def inv_rotmat(angle_rad):
    """
    Computes a 2x2 rotation matrix
    :param angle_rad: Rotation angle in rad
    :return: 2x2 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s],
                     [s, c]])


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if abs(diff_angle) < 5.0 or abs((180 - diff_angle)) < 5.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beginning of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def calculate_step_distance(vehicle_speed, dt=1.0 / 30.0, factor=1.5):
    """
    Calculate Distance traveled between the timestamps of the prediction based on the vehicle speed.
    :param: vehicle_speed Velocity of the ego vehicle in [km/h]
    :param dt time between two predictions in seconds
    :param factor Multiplication factor for the traveld distance between two timestamps
    :return: sampling_radius based on prediction frequency and vehicle speed.
    """
    #print("speed", vehicle_speed)
    #print("dt",  dt )
    #print("factor", factor / 3.6)
    return vehicle_speed * dt * factor / 3.6


def compute_next_waypoints(current_wp, d=2, k=100, stay_on_lane=True, active_lane_change=0):
    """
    Returning a trajectory queue of waypoint with the distance d to each other.
    If active_lane_change !=0 a lane change is desired, future wp will be set to the new lane (one to the right for 1
    and one to the left for -1).

    :param active_lane_change: [-1, 0, 1], if !=0 a lane change is desired, future wp will be set to the new lane.
    :param stay_on_lane: True, if the vehicle should stay on the lane,
                            in case of multiple options for the next wp.
    :param current_wp: Current waypoint of the ego vehicle.
    :param k: how many waypoints to compute
    :param d: distance of two waypoints

    :return: ordered queue of waypoints, starting from the nearest.
    """

    waypoints_queue = deque(maxlen=k)

    for i in range(k):
        if i < 0 and i != 0:
            next_waypoints = list(current_wp.next(3.0))
        else:
            try:
                next_waypoints = list(current_wp.next(d))
            except:
                print(i, " Last working waypoint for next: ", current_wp)


        if len(next_waypoints) == 1:
            # only one option available ==> lanefollowing
            next_waypoint = next_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(next_waypoints, current_wp)

            if stay_on_lane:
                road_option = RoadOption.STRAIGHT
                try :
                    next_waypoint = next_waypoints[road_options_list.index(road_option)]
                except:
                    # Use first entry of possible next waypoints, if road option STRAIGHT is not avaible.
                    next_waypoint = next_waypoints[0]
            else:
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

        waypoints_queue.append((next_waypoint, road_option))
        current_wp = next_waypoint

    return waypoints_queue


def compute_previous_waypoints(current_wp, d=2, k=100, stay_on_lane=True, active_lane_change=0):
    """
    Returning a trajectory queue of previous waypoint with the distance d to each other.

    :param stay_on_lane: True, if the vehicle should stay on the lane,
                            in case of multiple options for the next wp.
    :param current_wp: Current waypoint of the ego vehicle.
    :param k: how many waypoints to compute
    :param d: distance of two waypoints

    :return: ordered queue of waypoints, starting from the nearest.
    """

    waypoints_queue = deque(maxlen=k)

    for i in range(k):
        if i < 0:
            prev_waypoints = list(current_wp.previous(2.0))
        else:
            try:
                prev_waypoints = list(current_wp.previous(d))
            except:
                print(i, "Last working waypoint for previous: ", current_wp)


        if len(prev_waypoints) == 1:
            # only one option available ==> lanefollowing
            prev_waypoint = prev_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(
                prev_waypoints, current_wp)

            if stay_on_lane:
                road_option = RoadOption.STRAIGHT
                try:
                    prev_waypoint = prev_waypoints[road_options_list.index(
                        road_option)]
                except:
                    # Use first entry of possible previous waypoints, if road option STRAIGHT is not avaible.
                    prev_waypoint = prev_waypoints[-1]
            else:

                road_option = random.choice(road_options_list)
                prev_waypoint = prev_waypoints[road_options_list.index(
                    road_option)]

        waypoints_queue.append((prev_waypoint, road_option))
        current_wp = prev_waypoint

    return waypoints_queue


def get_localization_from_waypoint(wp: carla.Waypoint):
    """
    Returning the localization object from the waypoint object.
    :param wp: Carla waypoint object
    :return: carla.Location object of the waypoint element
    """
    loc = wp.transform.location

    return carla.Location(x= -1*loc.x, y=loc.y, z=0)


def get_localization_from_vehicle_transform(vehicle_transform: carla.Transform):
    """
    Returning the vehicle location in realation the right-handed coordinate system of the MPC.
    :param vehicle_transform: carla.Transform object of an actor in the scene
    :return: carla.Location object by considering the x-axis flip.
    """
    # Flipping x-axis of the vehicle location
    vehicle_loc = vehicle_transform.location
    return carla.Location(x=-1 * vehicle_loc.x, y=vehicle_loc.y, z=0)


def get_vehicle_velocity_vector(vehicle: carla.Vehicle, map_vehicle: carla.Map, velocity):
    """
    Function to return a velocity vector which points to the direction of the next waypoint.
    :param velocity: Desired vehicle velocity
    :param map_vehicle:  carla.Map
    :param vehicle: carla.Vehicle object
    :return: carla.Vector3D
    """

    # Getting current waypoint and next from vehicle
    current_wp = map_vehicle.get_waypoint(vehicle.get_location())
    next_wp = current_wp.next(1)[0]

    # Getting localization from the waypoints
    current_loc = get_localization_from_waypoint(current_wp)
    next_loc = get_localization_from_waypoint(next_wp)

    velocity_x = abs(next_loc.x - current_loc.x)
    velocity_y = abs(next_loc.y - current_loc.y)

    vector_vel0 = np.array([velocity_x, velocity_y, 0])
    vector_vel = (velocity / np.linalg.norm(vector_vel0)) * vector_vel0

    return carla.Vector3D(round(vector_vel[0], 3), round(vector_vel[1], 3), 0)


def euclidean_distance(loc1: carla.Location, loc2: carla.Location):
    """
    Calculating euclidean distance between two carla location points
    :param loc1: carla location 1
    :param loc2: carla location 2
    :return: distance
    """
    d = np.sqrt(np.square(loc1.x - loc2.x) + np.square(loc1.y - loc2.y))

    return d


def get_wp_vector(wp2: carla.Waypoint, wp1: carla.Waypoint):
    """
    Returns the unit vector from wp1 to wp2
    wp1, wp2:   carla.Waypoint objects
    """

    # Get location from wp if wp is a carla.Waypoint object
    location_1 = get_localization_from_waypoint(wp1) if isinstance(wp1, carla.Waypoint) else wp1
    location_2 = get_localization_from_waypoint(wp2) if isinstance(wp2, carla.Waypoint) else wp2

    x = location_2.x - location_1.x
    y = location_2.y - location_1.y

    return np.array([x, y, 0])


def get_wp_angle(wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Returning the orientation between two waypoints with respect to the x-axis.
    :param wp1: Carla Waypoint object 1 -->
    :param wp2: Carla Waypoint object 2
    :return: Angle [rad] between the both waypoints.
    """

    wp_vector = get_wp_vector(wp2, wp1)  # wp2 - wp1
    norm = np.linalg.norm(wp_vector) + np.finfo(float).eps

    # wp_angle = np.sign(wp_vector[1]) * np.arccos(np.dot(wp_vector, np.array([1, 0, 0]).T)/norm)
    wp_angle = np.arctan2(wp_vector[1], wp_vector[0])

    return wp_angle

def get_distance2wp(vehicle_transform, wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Calculating a norm vector form the vehicle position to the reference line (from wp1 to wp2) and
    returning the length from this vector.

    :param vehicle_transform: carla.Transform object of the location of the ego-vehicle in world
    :param wp1: Carla Waypoint object 1
    :param wp2: Carla Waypoint object 2
    :return: Returning distance of the vehicle to the reference line (from wp1 to wp2)
    """
    # Flipping x-axis of the vehicle location
    
    vehicle_loc = vehicle_transform.location
    #pdb.set_trace()
    ego_vehicle_loc = carla.Location(x=-1 * vehicle_loc.x, y=vehicle_loc.y, z=0)


    wp_vector = get_wp_vector(wp2, wp1)  # wp2 - wp1
    xy_vector = get_wp_vector(ego_vehicle_loc, wp2) # vehicle.loc - wp2

    norm = np.linalg.norm(wp_vector) + np.finfo(float).eps
    cross = np.cross(wp_vector, xy_vector) + np.finfo(float).eps

    return np.linalg.norm(cross/norm) + np.finfo(float).eps #normal distance from reference path of tv to ev (not refrence path of ev)


def get_angle2wp_line(vehicle_transform, wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Calculating a norm vector form the vehicle position to the reference line (from wp1 to wp2) and
    returning the length from this vector.

    :param vehicle_transform: carla.Transform object of the location of the ego-vehicle in world
    :param wp1: Carla Waypoint object 1
    :param wp2: Carla Waypoint object 2
    :return: Returning angle between 
    """
    # Flipping x-axis of the vehicle location
    
    vehicle_loc = vehicle_transform.location
    ego_vehicle_loc = carla.Location(x=-1 * vehicle_loc.x, y=vehicle_loc.y, z=0)


    wp_vector = get_wp_vector(wp2, wp1)  # wp2 - wp1
    xy_vector = get_wp_vector(ego_vehicle_loc, wp2) # vehicle.loc - wp2

    sign = np.sign(xy_vector[1] * wp_vector[0] - xy_vector[0] * wp_vector[1]) #pos or neg base on sign of the expression inside


    norm_wp = np.linalg.norm(wp_vector) + np.finfo(float).eps
    norm_xy = np.linalg.norm(xy_vector) + np.finfo(float).eps

    # angle_xy = sign * np.arccos(np.dot(xy_vector,
    #                         wp_vector) / (norm_xy * norm_wp))

    return sign

def draw_waypoints_debug(world, waypoints, z=0.5, color=None):
    """
    Draw a list of waypoints at a certain height given in z.

    :param color: RGB color of the waypoint marker
    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    if color is None:
        color = [255, 0, 0]
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, color=carla.Color(color[0], color[1], color[2]), life_time=0.2)


def draw_vehicle_bounding_box(world, location, heading, color=None):
    """
    Draw a list of waypoints at a certain height given in z.

    :param color: RGB color of the waypoint marker
    :param world: carla.world object
    :param location: carla.Location to place the center of the bounding box
    :param heading: heading of the bounding box element
    :return:
    """
    if color is None:
        color = [255, 165, 0]           # orange
    bbx = _create_bb(location)
    world.debug.draw_box(bbx, carla.Rotation(yaw=np.rad2deg(psi2carla(heading))), 0.1, carla.Color(255,165,0,0), life_time=0.2)

def draw_prediction_trajectory(world, location, color=None, thickness=0.25):
    """
    Drawing line following the location of the predictions steps.
    :param world: carla.world object
    :param location: list of carla.Location items of the prediction horizon of the MPC controller
    :param color: line color
    :return:
    """

    if color is None:
        color = [255, 165, 0]  # orange
    for step, loc in enumerate(location):
        if step == 0:
            continue
        world.debug.draw_line(location[step-1], loc, thickness=thickness, color=carla.Color(color[0], color[1], color[2]), life_time=0.2, )


def _create_bb(location: carla.Location, ):
  return carla.BoundingBox(location, carla.Vector3D(1.9, 0.8, 0.5))


def polynomial3(x, p0, p1, p2, p3):
    """
    3th-polynomial function for modeling a street trajectory.
        function: y = p3 * x³ + p2 * x² + p1 * x + p0
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return p3 * x ** 3 + p2 * x ** 2 + p1 * x + p0


def polynomial3_prime(x, p1, p2, p3):
    """
    First derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 3 * p3 * x² + 2 * p2 * x + p01
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 3 * p3 * x ** 2 + 2 * p2 * x + p1


def polynomial3_prime2(x, p2, p3):
    """
    Second derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 6 * p3 * x + 2 * p2 * x
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 6 * p3 * x + 2 * p2


def polynomial5(x, p0, p1, p2, p3, p4, p5):
    """
    3th-polynomial function for modeling a street trajectory.
        function: y = p3 * x³ + p2 * x² + p1 * x + p0
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return p5 * x **5 + p4 * x**4 + p3 * x ** 3 + p2 * x ** 2 + p1 * x + p0


def polynomial5_prime(x, p1, p2, p3, p4, p5):
    """
    First derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 3 * p3 * x² + 2 * p2 * x + p01
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 5 *p5 *x **4 + 4 * p4 * x **3 + 3 * p3 * x ** 2 + 2 * p2 * x + p1


def polynomial5_prime2(x, p2, p3, p4, p5):
    """
    Second derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 6 * p3 * x + 2 * p2 * x
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 20 *  p5 * x **3 + 12 * p4 * x **2 + 6 * p3 * x + 2 * p2


def func_kappa(x, p_arg):
    """
    Curvature function of a waypoint reference line.
    :param x: longitude value
    :param p_arg: np.array of fitted polynomial factors --> [p0, p1, p2, p3]
    :return: curvature of reference line at point x.
    """
    # if a constant is passes as polynomial arguments --> return p_arg as already calculated kappa
    if isinstance(p_arg, float) or isinstance(p_arg, int):
        return p_arg


    result = polynomial3_prime2(x, p_arg[2], p_arg[3])
    denominator = 1 + polynomial3_prime(x, p_arg[1], p_arg[2], p_arg[3]) ** 2
    denominator = denominator ** 1.5

    return result / denominator


def func_kappa2(x, p_arg):
    """
    Curvature function of a waypoint reference line for a n-th polynomial.
    :param x: longitude value
    :param p_arg: np.array of fitted polynomial factors --> [p0, p1, p2, p3]
    :return: curvature of reference line at point x.
    """
    # if a constant is passes as polynomial arguments --> return p_arg as already calculated kappa
    if isinstance(p_arg, float) or isinstance(p_arg, int):
        return p_arg

    p = np.poly1d(p_arg[-1::])
    p_prime = np.polyder(p)
    p_prime2 = np.polyder(p, 2)

    result = p_prime2(x)
    denominator = 1 + p_prime(x) ** 2
    denominator = denominator ** 1.5

    return result / denominator

def func_kappa5(x, p_arg):
    """
    Curvature function of a waypoint reference line for a n-th polynomial.
    :param x: longitude value
    :param p_arg: np.array of fitted polynomial factors --> [p0, p1, p2, p3]
    :return: curvature of reference line at point x.
    """
    # if a constant is passes as polynomial arguments --> return p_arg as already calculated kappa
    if isinstance(p_arg, float) or isinstance(p_arg, int):
        return p_arg

    result = polynomial5_prime2(x, p_arg[2], p_arg[3], p_arg[4], p_arg[5])
    denominator = 1 + polynomial5_prime(x, p_arg[1], p_arg[2], p_arg[3], p_arg[4], p_arg[5]) ** 2
    denominator = denominator ** 1.5

    return result / denominator


def psi2carla(psi):
    """
    Converting PSI (ego vehicle heading) in carla coordinate from NPMC coordinate system.
    :param psi: [rad] in NPMC coordinate system
    :return: psi: [rad] in in carla coordinate system.
    """
    return wrap2pi(np.pi - psi)


def psi2NMPC(psi):
    """
    Converting PSI (ego vehicle heading) in carla coordinate from NPMC coordinate system.
    :param psi: [rad] in NPMC coordinate system
    :return: psi: [rad] in in carla coordinate system.
    """
    return wrap2pi(np.pi - psi)

def get_xi_TV(tv_transform, wpList: carla.Waypoint, distance_wp: float, map, ev_wp_idx=5):
    """
    Retrieving the current xpof the MPC system.
    The Distance is approximated by determine the waypoints between the current WP of EV (W_C)
     and  nearest waypoint to the TV of the wpList W_nTV.
    Addionally the distance of  W_nTV and a new drawn nearest WP for TV is added by
        simple taking the euclidean distance .
    This is needed because the distance of the WPs in the wpList is ~2m.

    :param tv_transform: carla.Transform element of TV location + orientation
    :param wpList: list of carla.Waypoint for example the ones used
        to determine the road curvature kappa
    :return: XI in realtion to EV
    """

    # init wp index points to the nearest WP of TV in wpList.
    wp_idx = 0
    
    # Getting the nearest the wp to the target vehicle
    #tv_loc = get_localization_from_vehicle_transform(tv_transform) #gets xyz of car
    tv_wp = map.get_waypoint(tv_transform.location,project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    #pdb.set_trace()
    # Determine the last WP of wp_list which is behind tv_wp
    for i, wp in enumerate(wpList):
        if not is_within_distance_ahead(wp[0].transform, tv_wp.transform, distance_wp):
            continue
        
        wp_idx = i - 1
        
    if wp_idx == -1: #to fix the error when wp_idx = -1
        wp_idx = 0


    #if wp_idx - ev_wp_idx >= 0:
    euc_dis = euclidean_distance(wpList[wp_idx][0].transform.location, tv_wp.transform.location)
    #pdb.set_trace()
    return distance_wp * (wp_idx - ev_wp_idx) + euc_dis
    
    # if wp_idx - ev_wp_idx < 0:
    #     return distance_wp * (wp_idx - ev_wp_idx) - euclidean_distance(wpList[wp_idx][0].transform.location, tv_wp.transform.location)

def get_xi_TV_xyz(location, wpList: carla.Waypoint, distance_wp: float, map, ev_wp_idx=5):
    """
    Retrieving the current xpof the MPC system.
    The Distance is approximated by determine the waypoints between the current WP of EV (W_C)
     and  nearest waypoint to the TV of the wpList W_nTV.
    Addionally the distance of  W_nTV and a new drawn nearest WP for TV is added by
        simple taking the euclidean distance .
    This is needed because the distance of the WPs in the wpList is ~2m.

    :param tv_transform: carla.Transform element of TV location + orientation
    :param wpList: list of carla.Waypoint for example the ones used
        to determine the road curvature kappa
    :return: XI in realtion to EV
    """

    # init wp index points to the nearest WP of TV in wpList.
    wp_idx = 0

    # Getting the nearest the wp to the target vehicle
    #tv_loc = get_localization_from_vehicle_transform(tv_transform) #gets xyz of car
    tv_wp = map.get_waypoint(location,project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))

    # Determine the last WP of wp_list which is behind tv_wp
    for i, wp in enumerate(wpList):
        if not is_within_distance_ahead(wp[0].transform, tv_wp.transform, distance_wp):
            continue
        
        wp_idx = i - 1
        
    if wp_idx == -1: #to fix the error when wp_idx = -1
        wp_idx = 0

    return distance_wp * (wp_idx - ev_wp_idx) + euclidean_distance(wpList[wp_idx][0].transform.location, tv_wp.transform.location)
 

def xy2frenet_wp(vehicle: carla.Vehicle, map, wpList_kappa: carla.Waypoint, distance_wp: float):
    """
    Transforming the xy position of any vehicle in the scene into frenet state representation [xi, eta, phi]
    :param vehicle: carla.Vehicle has the current xy state included
    :param wpList_kappa: list of wp.Waypoint used to determine the road curvature value of the current scene
    :param distance_wp: distance between two wp in wpList
    :return: np.array([xi, eta, phi])
    """

    # Getting
    vehicle_transform = vehicle.get_transform()
    vehicle_loc = get_localization_from_vehicle_transform(vehicle_transform)
    wp_current = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    wp_next = wp_current.next(2)[0]

    # Calculating Xi
    xi = get_xi_TV(vehicle.get_transform(), wpList_kappa, distance_wp, map) #distance from

    # Determine ETA
    angle_xy = get_angle2wp_line(vehicle_transform, wp_current, wp_next) #angle between vehicle and waypoint path
    #print("I AM HERE")
    eta = np.sign(angle_xy) * get_distance2wp(vehicle_transform, wp_current, wp_next)

    # if EV and target are not on the same line add lane width to eta
    if wp_current.lane_id != wpList_kappa[5][0].lane_id:
        diff_lane = abs(wpList_kappa[5][0].lane_id) - abs(wp_current.lane_id)

        eta += np.sign(diff_lane) * abs(diff_lane) * wp_current.lane_width #differences in reference line for tv and ev

    # Determine Theta
    angle_wp = get_wp_angle(wp_current, wp_next) #gets angle of waypoints wrt x axis
    vehicle_heading = wrap2pi(np.pi - np.deg2rad(round(vehicle.get_transform().rotation.yaw, 3)))
    theta = wrap2pi(vehicle_heading - angle_wp) #angle offset of car from trajectory



    return np.array([vehicle_loc.x, vehicle_loc.y, vehicle_heading, get_speed(vehicle) / 3.6,xi, eta, theta,]) #x,y, , speed, xi,eta,theta

def xy2frenet_pnt_specific(vehicle, last_xi, time_stp ,tv_location, map, wpList_kappa: carla.Waypoint, distance_wp: float):
    """
    Transforming the xy position of any vehicle in the scene into frenet state representation [xi, eta, phi]
    :param vehicle: carla.Vehicle has the current xy state included
    :param wpList_kappa: list of wp.Waypoint used to determine the road curvature value of the current scene
    :param distance_wp: distance between two wp in wpList
    :param idx: define exact waypoint to be converted
    :return: np.array([xi, eta, phi])
    """
    
    tv_loc = carla.Location(x= tv_location[0], y= tv_location[1], z= tv_location[2])
    tv_trans = carla.Transform(location= tv_loc)
    #pdb.set_trace() #check location
    #getting waypoints
    wp_current = map.get_waypoint(tv_loc, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    wp_next = wp_current.next(2)[0]

    # Calculating Xi
    xi = get_xi_TV(tv_trans, wpList_kappa, distance_wp, map) #distance from
    #print('xi', xi)
    # Determine ETA
    angle_xy = get_angle2wp_line(tv_trans, wp_current, wp_next) 
    eta = np.sign(angle_xy) * get_distance2wp(tv_trans, wp_current, wp_next)

    # if EV and target are not on the same line add lane width to eta
    if wp_current.lane_id != wpList_kappa[5][0].lane_id:
        diff_lane = abs(wpList_kappa[5][0].lane_id) - abs(wp_current.lane_id)

        eta += np.sign(diff_lane) * abs(diff_lane) * wp_current.lane_width #differences in reference line for tv and ev
    # if eta < -4:
    #     pdb.set_trace()
    #print("eta", eta)
    #calculate velocity 
    a = np.sqrt(vehicle.get_acceleration().x**2 + vehicle.get_acceleration().y**2)


    t_delt = time_stp
    if last_xi == 0:
        last_xi = xi
    
    mag_vel = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
    tv_velocity = (xi - last_xi) + mag_vel - 0.5*a*t_delt
    #pdb.set_trace
    return [xi, tv_velocity, eta, 0] #speed, xi, eta

def predict_frenet_kinVehMod(x0: np.array, control: np.array, kappa: float,  delta: float=0.2):
    """
    Predict the future state (delta ahead) based on the kinematic bicycle model.
    :param x0: initial state [X, Y, PSI, Velocity, Xi, Eta, Theta]
    :param u: applied control [acceleration/braking, steering]
    :param delta: time passed till the desired return state, default: 0.2 seconds
    :return: future state
    """

    lr = 1.9
    lf = 1.9

    alpha = lambda u: np.arctan((lr / (lf + lr)) * np.tan(u[1]))
    dxi = lambda x, u: (1 / (1 - kappa * x[5])) * x[3] * np.cos(x[6] + alpha(u))

    kinVehModel = lambda x, u: np.array(
        [delta * (x[3] * np.cos(x[2] + alpha(u))) + x[0],  # X
         delta * (x[3] * np.sin(x[2] + alpha(u))) + x[1],  # Y
         delta * ((x[3] / lr) * np.sin(alpha(u))) + x[2],  # psi
         delta * (u[0]) + x[3],  # velocity
         delta * dxi(x, u) + x[4],  # xi
         delta * (x[3] * np.sin(x[6] + alpha(u))) + x[5],  # eta
         delta * ((x[3] / lr) * np.sin(alpha(u)) - kappa * dxi(x, u)) + x[6],  # theta
         ])

    return kinVehModel(x0, control)

