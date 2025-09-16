# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane, VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics, transformations
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.storage.native import get_assets_root_path


ROBOT_POS = np.array([0.0, -1.5, 0.5])
ROBOT_ORI = euler_angles_to_quats([0, 0, 90])

class PackingScript:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._script_generator = None

        self._delivery = False
        self._delivery_argument = {}

    def load_objects(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """

        general_asset_path = os.path.join(
            os.path.dirname(__file__), "..", "assets"
        )

        # Robot Franka
        robot_prim_path = "/packing_robot"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = SingleArticulation(
            name="packing_robot",
            prim_path=robot_prim_path,
            position=ROBOT_POS,
            orientation=ROBOT_ORI
        )

        # shelve
        shelve_asset_path = os.path.join(general_asset_path, "shelve_large","shelve_large.usd")
        shelve_prim_path = "/World/Shelve"
        add_reference_to_stage(shelve_asset_path, shelve_prim_path)
        self._shelve = SingleXFormPrim(
            prim_path=shelve_prim_path,
            name="shelve",
            position=np.array([0, -2.4, 0.75]),
            orientation=euler_angles_to_quats([0, 0, np.deg2rad(90)]),
            scale=np.array([1.0, 1.0, 0.7]),
        )
        
        # stool
        # stool_asset_path = os.path.join(general_asset_path,"ikea_wood_stools", "ikea_wood_stools.usd")
        # stool_prim_path = "/World/ikea_stool"
        # add_reference_to_stage(stool_asset_path, stool_prim_path)
        # self._stool = SingleXFormPrim(
        #     prim_path=stool_prim_path,
        #     name="ikea_stool",
        #     position=np.array([0.0, 0.0, 0.15]),
        #     orientation=euler_angles_to_quats([0, 0, 0]),
        #     scale=np.array([1.0, 1.0, 1.0]),
        #     # visible=False
        # )

        self._table_obstacle = VisualCuboid(
            prim_path="/World/Obstacles/TableProxy",
            name="table_obstacle",
            position=np.array([0., -1.0, 0.58]),
            scale=np.array([1.0, 0.5, 0.3]),     
            visible=False,             
        )

        # self.Franka_support = VisualCuboid(
        #     name="f_sup",
        #     prim_path="/World/franka_support",
        #     scale=np.array([0.2, 0.2, 0.5]),
        #     position=np.array([0.0, -1.5, 0.4]),
        #     color=np.array([0.0, 0.0, 0.0]),
        # )

        self._blue_cubes = [
            DynamicCuboid(
            name="bc1",
            position=np.array([0., -2.2, 1.05]),
            scale=np.array([1.,1.,2.5]),
            prim_path="/World/blue_cube_1",
            size=0.05,
            color=np.array([1, 0, 0]),),
            DynamicCuboid(
            name="bc2",
            scale=np.array([1.,1.,2.5]),
            position=np.array([0.3, -2.2, 0.65]),
            prim_path="/World/blue_cube_2",
            size=0.05,
            color=np.array([1, 0, 0]),),
        ]

        self._ground_plane = GroundPlane("/World/Ground")

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._ground_plane, *self._blue_cubes, self._shelve, self._table_obstacle

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        # set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        # Loading RMPflow can be done quickly for supported robots
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)
        self._rmpflow.set_robot_base_pose(ROBOT_POS,ROBOT_ORI)
        # self._rmpflow.visualize_collision_spheres()

        self._rmpflow.add_obstacle(self._table_obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):

        while True:
            if self._delivery:
                target_pos = np.array([0., -1.0, 0.9])
                shelve_take_orientation = euler_angles_to_quats([0, np.pi/2, -np.pi/2])
                base_orientation = euler_angles_to_quats([np.pi,0,0])

                yield from self.open_gripper_franka(self._articulation)

                for cube in self._blue_cubes:
                    T_robot_cube = transformations.get_relative_transform(cube.prim, self._articulation.prim) # Pos du cube dans le repere robot
                    # cube_pos, cube_ori = transformations.pose_from_tf_matrix(T_robot_cube)
                    # post_take = transformations.get_translation_from_target(np.array([0.,0.,0.05]),cube.prim,self._articulation.prim)
                    # pre_take = transformations.get_translation_from_target(np.array([0.,0.3,0.0]),cube.prim,self._articulation.prim)

                    cube_pos, cube_ori = cube.get_world_pose()
                    post_take = cube_pos + np.array([0.,0.,0.1])
                    pre_take = cube_pos + np.array([0.,0.3,0.0])
                    
                    success = yield from self.goto_position(
                        pre_take , shelve_take_orientation, self._articulation, self._rmpflow
                    )
                    print("[PACKING] pre take : ", success)

                    success = yield from self.goto_position(
                        cube_pos , shelve_take_orientation, self._articulation, self._rmpflow
                    )
                    print("[PACKING] take : ", success)

                    yield from self.close_gripper_franka(self._articulation, close_position=np.array([0.015, 0.015]))

                    success = yield from self.goto_position(
                        post_take , shelve_take_orientation, self._articulation, self._rmpflow, timeout=200
                    )
                    print("[PACKING] post take : ", success)

                    success = yield from self.goto_position(
                        target_pos , base_orientation, self._articulation, self._rmpflow
                    )
                    print("[PACKING] deposit : ", success)

                    yield from self.open_gripper_franka(self._articulation)
                    self._delivery = False
                    
            yield from ()


    ################################### Functions


    def receive_deliver_order(
        self,
        arguments
    ):
        self._delivery = True
        self._delivery_argument = arguments

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=1.57,
        timeout=1000,
    ):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """
        rmpflow.update_world()
        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)

        for i in range(timeout):
            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_franka(self, articulation):
        open_gripper_action = ArticulationAction(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
            yield ()

        return True

    def close_gripper_franka(self, articulation, close_position=np.array([0, 0]), atol=0.05):
        # To close around the cube, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[7:], np.array(close_position), atol=atol):
            yield ()

        return True
