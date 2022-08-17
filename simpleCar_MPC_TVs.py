#!/usr/bin/env python

"""
    Simple MPC control example with some TV in front of the EV.
    Currently only simple case with considering one TV directly in front of EV is considered.
"""
import glob
import os
import sys

import argparse
import time
from termcolor import colored

import pandas as pd

try:
    # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #     sys.version_info.major,
    #     sys.version_info.minor,
    #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
except IndexError:
    raise RuntimeError("Unable to load carla !!")
try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from mpcCARLA.control_agent import *
from agents.tools.misc import *
from mpcCARLA.visualization_tools import plot_frenet_states
from mpcCARLA.local_planner_modified import *


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """
    #defining what is inside class
    def __init__(self, world, *sensors, **kwargs): #**kwangs takes a variable length dict
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30) #fps is the key and 30 is default val
        self._queues = []
        self._settings = None


    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings( #returns ID of frame
            no_rendering_mode=False,
            synchronous_mode=True, #waits for a client tick
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick) #each sensor is added in the queue
        for sensor in self.sensors:
            make_queue(sensor.listen) #calls to sensor every new measurement
        return self

    def tick(self, timeout):
        self.frame = self.world.tick() #server waits for client click before next frame
        data = [self._retrieve_data(q, timeout) for q in self._queues] #gets array data
        assert all(x.frame == self.frame for x in data)
        #pdb.set_trace()
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame: #sync check
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) #image data into 1D array
    array = np.reshape(array, (image.height, image.width, 4)) 
    array = array[:, :, :3] #keeps the first four of the third dimension
    array = array[:, :, ::-1] #reverse the third dimension
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def main():
    # ----------------------
    # Input Parser Handler
    # ----------------------
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host', #name or list of option strings
        metavar='H', #what to call in message
        default='127.0.0.1', #if none given
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-target-vehicles',
        metavar='N',
        default=5,
        type=int,
        help='number of vehicles (default: 5)')
    argparser.add_argument(
        '-f', '--filename-log',
        metavar='F',
        default=None,
        type=str,
        help='Filename of hdf5 file of the logging information')
    argparser.add_argument(
        '-s', '--sampling-waypoints',
        metavar='S',
        default=0,
        type=int,
        help='Define how much waypoints of the vehicle should get sampled during the drive.')
    argparser.add_argument(
        '-x', '--window-size-x',
        metavar='X',
        default=1920,
        type=int,
        help='Window width of the pygame window (default: 1920)')
    argparser.add_argument(
        '-y', '--window-size-y',
        metavar='Y',
        default=1080,
        type=int,
        help='Window height of the pygame window (default: 1080)')
    args = argparser.parse_args()

    actor_list = []
    target_vehicle_list = []  # list of all target vehicle objects
    required_tvs = args.number_of_target_vehicles
    flag_sampling_waypoints = True if args.sampling_waypoints > 0 else False

    # list locations of good spawn points
    spawn_dict = {
        # # ego vehicle spawn point OG
        # "EV": carla.Location(x=-13.298, y=-187, z=0),
        # ego vehicle spawn point
        "EV": carla.Location(x=-13, y=-145, z=0), #works

        ## same lane candidate (in front of EV) OG
        "TV1": carla.Location(x=-13.1351, y=-136.132, z=0.0),
        # same lane candidate (in front of EV)
        #"TV1": carla.Location(x=-13.1351, y=-205.132, z=0.0),
        # right lane candidate (behind EV)


        "TV2": carla.Location(x=-16.299, y=-177.89, z=0),
        # right lane candidate (in front of EV)
        "TV3": carla.Location(x=-16.299, y=-119.43228, z=0),
        # left lane candidate lane candidate (next to EV)
        "TV4": carla.Location(x=-9.798, y=-187.74, z=0),
        # left lane candidate (in front of EV)
        "TV5": carla.Location(x=-9.53, y=-119.432, z=0),
        # same lane candidate (behind of EV)
        "TV6": carla.Location(x=-9.395, y=-212.342, z=0.0),
        # same lane candidate (in front of EV)
        "TV7": carla.Location(x=-13.1351, y=-106.132, z=0.0)
    }

    # Init pygame display
    pygame.init()
    # display = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)
    display = pygame.display.set_mode(
        (args.window_size_x, args.window_size_y), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    # Init loging information
    data_log = pd.DataFrame()
    kappa_log = pd.DataFrame()
    filename = args.filename_log
    if filename is None:
        filename = 'frenet_mpc_speed_limit2.h5'





    try:
        # -----------------------------
        # Connect to the Carla server
        # -----------------------------

        # Connect to Carla server
        client = carla.Client(args.host, args.port)
        client.set_timeout(3.0)
        clock = pygame.time.Clock() #instantiate in-game clock

        # Loading the world of Town6
        world = client.get_world()
        # Town06 => Long Highwazs with many entrances and exits.
        # Town04 ==> Infinite Loop with a highway.
        world = client.load_world('Town04')
        map = world.get_map()

        # Getting blueprint for Vehicle --> BMW GrandTourer
        blueprint_library = world.get_blueprint_library() #returns a list of blueprints
        bp = blueprint_library.find('vehicle.bmw.grandtourer') #found the one we want



        # ------------------------------
        # Spawning the target vehicles
        # ------------------------------
        required_tvs = 2 #OG 2

        if required_tvs > 0:

            # Using manually observed starting location for it.
            tv_counter = 0
            target_vehicle_dict = {}
            # Change color of TV to red
            bp.set_attribute('color', '250,20,60') # Change color of TV to red
            for key, loc in spawn_dict.items():
                # Only spawning as much Target vehicle as required
                if tv_counter > required_tvs:
                    continue
                if key is not "EV":
                    print("Spawning ", key, " at ", loc, )
                    spawn_point = map.get_waypoint(spawn_dict.get(key), project_to_road=True, #############################################################
                                                   lane_type=(carla.LaneType.Driving))

                    tv_vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
                    print(type(tv_vehicle))
                    # Teleport the vehicle.
                    tv_vehicle.set_transform(spawn_point.transform)

                    if tv_vehicle:
                        print("Successfully spawned target vehicle !!")
                    target_vehicle_dict[key] = tv_vehicle
                tv_counter += 1             # COUNT UP TV COUNTER
            print(target_vehicle_dict)

            # Adding a local Planner to all target vehicles
            agent_dict = dict()
            for key, vehicle in target_vehicle_dict.items():
                if not vehicle:
                    print("Target Vehicle was not rightfully created.")
                    continue
                # Adjust vehicle speed to lane position of the vehicle
                if key in ["TV0", "TV1", "TV2"]:
                    velo = 60
                if key in ["TV3", "TV4", "TV5"]:
                    velo = 120
                if key in ["TV6"]:
                    velo = 100

                # Init local Planner parameters
                opt_dict = {
                    "dt": 1 / 30,
                    "target_speed": 50,
                    "sampling_radius": 0.5,
                }
                print("Target Vehicle ", key, " is set to a velocity of ", velo)
                agent_dict[key] = LocalPlanner(vehicle, opt_dict)


        # -------------------
        # Spawn Ego Vehicle
        # -------------------
        # Setting the car color to black
        bp.set_attribute('color', '172,216,230')
        # Now we need to give an initial transform for the ego vehicle.
        # We choose a transform location manually observed in the UE4 editor.
        spawn_point = map.get_waypoint(spawn_dict.get("EV"), project_to_road=True, 
                                       lane_type=(carla.LaneType.Driving))
        print("Ego vehicle spawn point: ",
              spawn_point.transform.location, spawn_dict.get("EV"))
        ego_vehicle = world.try_spawn_actor(
            bp, random.choice(map.get_spawn_points())) #spawn first then teleport the car?
        # Teleport the vehicle.
        ego_vehicle.set_transform(spawn_point.transform)

        #####
        print("given spawn point ", spawn_dict.get("EV"))
        print("teleported to ", spawn_point.transform.location)
        #####


        # Starting with a light offset from the ideal reference line
        light_offset_loc = spawn_point.transform.location
        #light_offset_loc.x += 0.8
        light_offset_rot = spawn_point.transform.rotation
        light_offset_rot.yaw -= 10 #why?
        ego_vehicle.set_transform(carla.Transform(light_offset_loc, light_offset_rot))


        # Camera sensor for ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.window_size_x))
        camera_bp.set_attribute('image_size_y', str(args.window_size_y))
        camera_transform = carla.Transform(carla.Location(x=-15, z=3))
        camera = world.spawn_actor(
            camera_bp, camera_transform, attach_to=ego_vehicle)

        actor_list.append(camera)

        #lidar sensor for ego vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('rotation_frequency', '30')
        lidar_bp.set_attribute('range', '30') #meters
        lidar_bp.set_attribute('lower_fov', '-22.5') #meters
        lidar_bp.set_attribute('upper_fov', '0') #meters
        #lidar_bp.set_attribute('points_per_second', '20000') #meters
        #lidar_bp.set_attribute('channels', '20') #meters
        lidar_transform = carla.Transform(carla.Location(x=0.4, z=1.6))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to = ego_vehicle)

        actor_list.append(lidar)


        # Set start velocity for ego vehicle
        # start_velocity = 100
        # vector_velocity = get_vehicle_velocity_vector(ego_vehicle, map, start_velocity / 3.6)
        # ego_vehicle.set_target_velocity(vector_velocity)


        # ----------------------------
        # Create MPC agent for the EV
        # ----------------------------
        # agent = VehicleClassicMPC(vehicle=vehicle, opt_dict=opt_dict)
        agent = VehicleCurvMPC(ego_vehicle, target_vehicle_dict) 

        # -------------------
        # Carla control Loop
        # -------------------
        fps = 30
        timestep_count = int(0)
        timestamp = timestep_count / fps

        x0 = agent.get_state()
        data0 = {'timestamp': timestamp, 'X': x0[0], 'Y':x0[1], 'PSI':x0[2], 'Velocity':x0[3], 'Xi':0, 'Eta':0, 'Theta':0, 'u_acceleration':0, 'u_steering_angle':0, 'pred_states':[agent.get_state()],
               'pred_control':[[0,0]], 'computation_time':0, "kappa": 0, "curvature_radius": 0}

        kappa0 = {
                'timestamp': timestamp,
                'wp_mat': [np.zeros([25, 2])],
                'wp_mat_0': [np.zeros([25, 2])],
                'refernce_point': [np.array([0,0])],
                'rotations_mat': [rotmat(0)],
                'angle_wp': 0,
                'p_opt': [np.array([0,0,0,0])], }

        data_log = pd.DataFrame(data0)
        kappa_log = pd.DataFrame(kappa0)

        # Start location of the EV
        start_loc = ego_vehicle.get_location()

        printer = False ###to prevent all the continual updates############################################################

        with CarlaSyncMode(world, camera, lidar, fps=30) as sync_mode:
            while True:
                frame_start = time.time()

                timestamp = timestep_count / fps
                #print("Timestamp: ", timestamp)
                snap_shot, image_rgb, lidar_data = sync_mode.tick(timeout=2.0)
                clock.tick()
                draw_image(display, image_rgb)
                pygame.display.flip()



                # printing the carla simulation time
                carla_time = time.time() - frame_start

                # get local planner control
                veh_control, frenet_data, kappa_data = agent.run_step(lidar_data, timestep_count, log=True, print=printer) #printer stops all repetitive printing , debug (in control agent stops clearing & more printing)

                veh_control.manual_gear_shift = False

                if printer:
                    # print mpc computation time
                    print("Timestamp: ", timestamp)
                    print(colored('Needed time for CARLA simulation: ' + str(carla_time), 'white', 'on_green'))
                    mpc_time = time.time() - frame_start - carla_time
                    print(colored('MPC computation time: ' + str(mpc_time), 'white', 'on_green'))

                # ------------------
                # logging data
                # ------------------
                frenet_data['timestamp'] = round(timestamp, 2)
                col = ['timestamp', 'X', 'Y', 'PSI', 'Velocity', 'Xi', 'Eta', 'Theta', 'u_acceleration',
                       'u_steering_angle', 'pred_states', 'pred_control', 'computation_time',
                       "kappa", "curvature_radius", 'velocity_error', 'kappa_state', 'target_velocity']
                df = pd.DataFrame(frenet_data, columns=col)
                data_log = data_log.append(df)

                kappa_data['timestamp'] = [round(timestamp, 2)]
                col2 = ['timestamp', 'wp_mat', 'wp_mat_0', 'refernce_point', 'rotations_mat', 'angle_wp', 'p_opt']
                df2 = pd.DataFrame(kappa_data, columns=col2)
                kappa_log = kappa_log.append(df2)

                # Applying the MPC control
                ego_vehicle.apply_control(veh_control)

                if printer:
                    # Print current vehicle state
                    print("Current speed: ", get_speed(ego_vehicle))
                    print("Current vehicle location: ", ego_vehicle.get_location(), np.deg2rad(ego_vehicle.get_transform().rotation.yaw))

                # Setting speed limit of right car to same as EV
                # agent_dict['TV2'].set_speed(get_speed(ego_vehicle)+2)
                current_speed_limit = ego_vehicle.get_speed_limit()
                if current_speed_limit <= 40:
                    current_speed_limit += 30

                agent_dict['TV1'].set_speed(current_speed_limit - 20)
                agent_dict['TV2'].set_speed(current_speed_limit - 20)

                # Applzing control on TVs
                if required_tvs > 1:
                    if printer:
                        print("Applying control to other TVs")
                    # Controls for Target Vehicle
                    for key, tv_agent in agent_dict.items():
                        # get local planner control
                        veh_control = tv_agent.run_step(debug=False)
                        veh_control.manual_gear_shift = False

                        # Applying the PID control
                        target_vehicle_dict[key].apply_control(veh_control)


                timestep_count += 1

                if printer:
                    # printing the carla simulation time
                    print(colored('MPC computation time with logging: ' + str(time.time() - frame_start - carla_time), 'white', 'on_green'))
                    print(colored('Time need for frame: ' + str(time.time() - frame_start), 'white', 'on_green'))

                

                 


                #if timestep_count / 30 > 10 and is_within_distance_ahead(start_loc, ego_vehicle.get_location(), ego_vehicle.get_transform().rotation.yaw, 10):
                if timestep_count / 30 > 20 and euclidean_distance(start_loc, ego_vehicle.get_location()) < 1:
                    print("Round is completed !!")
                    break




    finally:
        data_log = data_log.set_index('timestamp')
        data_log.to_hdf(filename, key='frenet_log', mode='w')
        kappa_log = kappa_log.set_index('timestamp')
        kappa_log.to_hdf(filename, key='kappa_log', mode='a')

        # Plotting frenet states (needs specific files for specific number of TVs?)
        #plot_frenet_states(filename)

        ego_vehicle.destroy()
        camera.destroy()
        lidar.destroy()


if __name__ == '__main__':
    main()
