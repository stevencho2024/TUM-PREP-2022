#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

import argparse
import time

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


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
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


def euclidean_distance(loc1: carla.Location, loc2: carla.Location):
    d = np.sqrt(np.square(loc1.x - loc2.x) + np.square(loc1.y - loc2.y))

    return d


def main():
    # ----------------------
    # Input Parser Handler
    # ----------------------
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
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
        # ego vehicle spawn point
        "EV": carla.Location(x=-13.298, y=-187, z=0),
        # right lane candidate (behind EV)
        "TV1": carla.Location(x=-16.299, y=-187.89, z=0),
        # right lane candidate (in front of EV)
        "TV2": carla.Location(x=-16.299, y=-119.43228, z=0),
        # left lane candidate lane candidate (next to EV)
        "TV3": carla.Location(x=-9.798, y=-187.74, z=0),
        # left lane candidate (in front of EV)
        "TV4": carla.Location(x=-9.53, y=-119.432, z=0),
        # same lane candidate (behind of EV)
        "TV5": carla.Location(x=-9.395, y=-212.342, z=0.0),
        # same lane candidate (in front of EV)
        "TV6": carla.Location(x=-13.1351, y=-146.132, z=0.0)
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
        clock = pygame.time.Clock()

        # Loading the world of Town6
        world = client.get_world()
        # Town06 => Long Highwazs with many entrances and exits.
        # Town04 ==> Infinite Loop with a highway.
        world = client.load_world('Town04')
        map = world.get_map()

        # Getting blueprint for Vehicle --> BMW GrandTourer
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.find('vehicle.bmw.grandtourer')
        # Setting the car color to black
        bp.set_attribute('color', '0,0,0')

        # -------------------
        # Spawn Ego Vehicle
        # -------------------

        # Now we need to give an initial transform for the ego vehicle.
        # We choose a transform location manually observed in the UE4 editor.
        spawn_point = map.get_waypoint(spawn_dict.get("EV"), project_to_road=True,
                                       lane_type=(carla.LaneType.Driving))
        print("Ego vehicle spawn point: ",
              spawn_point.transform.location, spawn_dict.get("EV"))
        ego_vehicle = world.try_spawn_actor(
            bp, random.choice(map.get_spawn_points()))
        # Teleport the vehicle.
        ego_vehicle.set_transform(spawn_point.transform)


        # Starting with a light offset from the ideal reference line
        light_offset_loc = spawn_point.transform.location
        #light_offset_loc.x += 0.8
        light_offset_rot = spawn_point.transform.rotation
        light_offset_rot.yaw -= 10
        ego_vehicle.set_transform(carla.Transform(light_offset_loc, light_offset_rot))


        # Camera sensor for ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.window_size_x))
        camera_bp.set_attribute('image_size_y', str(args.window_size_y))
        camera_transform = carla.Transform(carla.Location(x=-15, z=3))
        camera = world.spawn_actor(
            camera_bp, camera_transform, attach_to=ego_vehicle)

        actor_list.append(camera)

        # Set start velocity for ego vehicle
        # start_velocity = 100
        # vector_velocity = get_vehicle_velocity_vector(ego_vehicle, map, start_velocity / 3.6)
        # ego_vehicle.set_target_velocity(vector_velocity)

        # ----------------------------
        # Create MPC agent for the EV
        # ----------------------------
        # agent = VehicleClassicMPC(vehicle=vehicle, opt_dict=opt_dict)
        agent = VehicleCurvMPC(ego_vehicle)

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


        start_loc = ego_vehicle.get_location()

        with CarlaSyncMode(world, camera, fps=30) as sync_mode:
            while True:
                timestamp = timestep_count / fps
                print("Timestamp: ", timestamp)
                snapshot, image_rgb, = sync_mode.tick(timeout=2.0)
                clock.tick()
                draw_image(display, image_rgb)
                pygame.display.flip()

                # get local planner control
                veh_control, frenet_data, kappa_data = agent.run_step(timestep_count, log=True)

                veh_control.manual_gear_shift = False

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


                # Print current vehicle state
                print("Current speed: ", get_speed(ego_vehicle))
                print("Current vehicle location: ", ego_vehicle.get_location(), np.deg2rad(ego_vehicle.get_transform().rotation.yaw))

                timestep_count += 1

                #if timestep_count / 30 > 10 and is_within_distance_ahead(start_loc, ego_vehicle.get_location(), ego_vehicle.get_transform().rotation.yaw, 10):
                if timestep_count / 30 > 20 and euclidean_distance(start_loc, ego_vehicle.get_location()) < 1:
                    print("Round is completed !!")
                    break


    finally:
        data_log = data_log.set_index('timestamp')
        data_log.to_hdf(filename, key='frenet_log', mode='w')
        kappa_log = kappa_log.set_index('timestamp')
        kappa_log.to_hdf(filename, key='kappa_log', mode='a')

        # Plotting frenet states
        plot_frenet_states(filename)
        ego_vehicle.destroy()
        camera.destroy()


if __name__ == '__main__':
    main()
