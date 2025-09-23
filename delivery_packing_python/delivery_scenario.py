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
import os, math

from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices, quats_to_euler_angles, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from . import global_variables

ROBOT_POS = np.array([0.,0.1,0])
ROBOT_ORI = euler_angles_to_quats(np.array([0.,0.,np.pi]))


bedroom_targets = [
    np.array([3,2]),
    np.array([1,7]),
    np.array([-1,5])
]

class DeliveringScript:
    def __init__(self):
        self._move_controller = None

        self._articulation = None

        self._script_generator = None

    def load_objects(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """

        self.general_asset_path = os.path.join(
            os.path.dirname(__file__), "..", "assets"
        )

        robot_prim_path = "/delivery_robot"
        path_to_robot_usd = os.path.join(self.general_asset_path, "a2d", "a2d_finger.usd")
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = SingleArticulation(
            name="delivery_robot",
            prim_path=robot_prim_path,
            position=ROBOT_POS,
            orientation=ROBOT_ORI
        )

        #tray
        tray_asset_path = os.path.join(self.general_asset_path,"ikea_tray","ikea_tray.usd")
        tray_prim_path = "/tray1"
        add_reference_to_stage(tray_asset_path,tray_prim_path)
        
        self._tray = SingleXFormPrim(
            name="tray1",
            prim_path=tray_prim_path,
            position=np.array([-1.1, 0.1, 1.05]),
            orientation=euler_angles_to_quats([0, 0, np.deg2rad(90)]),
            scale = np.array([1.5,1.5,1.5])
        )

        #tables
        table_asset_path = os.path.join(self.general_asset_path, "bedroom", "bedroom.usd")
        add_reference_to_stage(table_asset_path, "/bed1")
        add_reference_to_stage(table_asset_path, "/bed2")
        add_reference_to_stage(table_asset_path, "/bed3")
        self._tables = [
            SingleXFormPrim(
                name="bed1",
                prim_path="/bed1",
                position=np.array([4, 3, 0]),
                orientation=euler_angles_to_quats([0,0,-2*np.pi/3])
            ),
            SingleXFormPrim(
                name="bed2",
                prim_path="/bed2",
                position=np.array([1, 8, 0]),
                orientation=euler_angles_to_quats([0,0,-np.pi/2])
            ),
            SingleXFormPrim(
                name="bed3",
                prim_path="/bed3",
                position=np.array([-2, 6, 0]),
            ),
        ]



        self.support = [
            FixedCuboid(
                name="support1",
                prim_path="/support/o_central",
                position=np.array([-1.25,0.1,0.5]),
                scale = np.array([0.5,0.07,1])
            ),
            FixedCuboid(
                name="support2",
                prim_path="/support/o_right",
                position=np.array([-1.25,0.4,0.5]),
                scale = np.array([0.5,0.07,1])
            ),
            FixedCuboid(
                name="support3",
                prim_path="/support/o_left",
                position=np.array([-1.25,-0.2,0.5]),
                scale = np.array([0.5,0.07,1])
            )
        ]
        

        # Return assets that were added to the stage so that they can be registered with the core.World
        return *self._tables, self._articulation, self._tray

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """

        self._set_arms_position(0,1.2217,0,1.2217,np.pi/2,0.349)

        self._move_controller = AckermannController(
            "test_controller", wheel_base=0, track_width=0.56, front_wheel_radius=0.4
        )

        self._wheel_indices = [self._articulation.get_dof_index('left_wheel_f'),
        self._articulation.get_dof_index('right_wheel_f'),
        self._articulation.get_dof_index('left_wheel_d'),
        self._articulation.get_dof_index('right_wheel_d')]

        self._kinematics_solver = LulaKinematicsSolver(
            urdf_path=os.path.join(self.general_asset_path, "a2d","a2d.urdf"),
            robot_description_path=os.path.join(self.general_asset_path, "a2d","a2d_description.yaml"),
        )

        # stage = get_current_stage()
        # joint_prim = stage.DefinePrim("/simulated_grasp", "PhysicsRigidJoint")
        # joint_api = UsdPhysics.Joint(joint_prim)
        # joint_api.CreateJointTypeAttr("fixed")
        # joint_api.CreateBody0RelPathAttr("/World/robot/base_link")
        # joint_api.CreateBody1RelPathAttr("/tray1")


        # self._open_gripper()

        end_effector_name = "A2D_Link7_l"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)

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
        self._set_arms_position(0,1.2217,0,1.2217,np.pi/2,0.349)
        # self._open_gripper()
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

        yield from self.move_step(np.array([5,0,0,1,0,0,0]))


    ################################### Functions

    def receive_delvier_order(self, args):
        if "1" in args["room"]:
            self._room_id = 0
        elif "2" in args["room"]:
            self._room_id = 1
        else:
            self._room_id = 2

    def _set_body_position(self, body_lift : float, body_pitch : float, head_yaw : float, head_pitch : float):
        self._articulation.set_joint_positions(np.array([body_lift, body_pitch, head_yaw, head_pitch]),[0,5,6,9])

    def _set_arms_position(self, shoulder_pitch : float, shoulder_roll : float, shoulder_yaw : float, 
                           elbow_pitch : float, elbow_yaw : float, elbow_roll : float):
        self._articulation.set_joint_positions(np.array(
            [-shoulder_pitch, shoulder_pitch, shoulder_roll, -shoulder_roll, shoulder_yaw, -shoulder_yaw,
             -elbow_pitch, elbow_pitch, elbow_yaw, -elbow_yaw, elbow_roll, -elbow_roll]),
            [7,8,10,11,12,13,14,15,16,17,18,19])
        
    def _mirror_action_for_right_arm(self, action : ArticulationAction) -> ArticulationAction:
        d = action.get_dict()
        jp = d.get("joint_positions", None)
        if not jp:
            raise ValueError
        right_jp = np.array([-val for val in jp])
        return ArticulationAction(joint_positions=right_jp, joint_indices=[8,11,13,15,17,19,21])
        
        
    def _send_velocity_for_wheel(self, left_velocity : float, right_velocity : float):
        action = ArticulationAction(joint_velocities=[left_velocity, right_velocity, left_velocity, right_velocity],
                                    joint_indices=self._wheel_indices)
        self._articulation.apply_action(action)

    def move_step(self, goal_pose, dt=1/60.0, pos_tol=0.05, yaw_tol=0.05, timeout=10000):
        """
        Allow to move the robot to a goal pose N,7 pos + orientation.
        """
        
        # self._send_velocity_for_wheel(3,-3)

        print("[DELIVERING] Setting arm")

        robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        t = robot_base_translation + [-0.2,-0.1,1]
        
        self._kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        t = np.array([-0.65640712, -0.1 ,  0.8307782])
        l_action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(t)


        self._articulation.apply_action(l_action)
        r_action = self._mirror_action_for_right_arm(l_action)
        self._articulation.apply_action(r_action)

        # print(success, action)

        for i in range(1000):
            ee_trans, ee_ori = self._articulation_kinematics_solver.compute_end_effector_pose()
            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, t)
            if trans_dist <= 0.01:
                break
            yield ()

        t = np.array([-0.71, -0.1 ,  0.90])
        l_action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(t)


        self._articulation.apply_action(l_action)
        r_action = self._mirror_action_for_right_arm(l_action)
        self._articulation.apply_action(r_action)

        for i in range(1000):
            ee_trans, ee_ori = self._articulation_kinematics_solver.compute_end_effector_pose()
            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, t)
            if trans_dist <= 0.01:
                break
            yield ()

        print("[DELIVERING] Going to take the crate")

        self._send_velocity_for_wheel(1,1)

        for i in range(1000):
            r_trans, r_ori = self._articulation.get_world_pose()
            if r_trans[0] < -0.33:
                break
            yield ()

        self._articulation.apply_action(ArticulationAction(joint_positions=[0.4],joint_indices=[0]))

        print("[DELIVERING] Tray taken")

        self._send_velocity_for_wheel(-1,-1)

        for i in range(1000):
            r_trans, r_ori = self._articulation.get_world_pose()
            if r_trans[0] > 0.1:
                break
            yield ()

        self._articulation.apply_action(ArticulationAction(joint_positions=[0.2],joint_indices=[0]))

        global_variables.state_machine_id = 1

        self._articulation.apply_action(ArticulationAction(joint_positions=[-0.15],joint_indices=[0]))

        print("[DELIVERING] Going to packing zone")

        yield from self.go_to_goal([0,-0.5],[-0.1,-0.1],speed=2)

        print("[DELIVERING] Arrived to packing zone")

        while global_variables.state_machine_id != 2:
            yield ()

        print("[DELIVERING] Exiting packing zone")

        self._send_velocity_for_wheel(-1,-1)

        for i in range(1000):
            r_trans, r_ori = self._articulation.get_world_pose()
            if r_trans[1] > 0.2:
                break
            yield ()


        self._articulation.apply_action(ArticulationAction(joint_positions=[0.],joint_indices=[0]))

        print("[DELIVERING] GOing to bedroom")

        yield from self.go_to_goal(bedroom_targets[self._room_id], bedroom_targets[self._room_id],speed=2,trans_tolerance=0.8)

        print("[DELIVERING] Arrived to bedroom")

        return True
    
    def go_to_goal(self, ori_goal, pos_goal, speed : float = 1, trans_tolerance : float = 0.1, rot_tolerance : float = 0.1):

        def compute_rot_err():
            r_trans, r_ori = self._articulation.get_world_pose()
            yaw = quats_to_euler_angles(r_ori)[2]
            dx, dy = ori_goal[0] - r_trans[0], ori_goal[1] - r_trans[1]
            goal_angle = np.arctan2(dy, dx)
            return np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw))
        
        def compute_dist_err():
            r_trans, r_ori = self._articulation.get_world_pose()
            err = np.linalg.norm([pos_goal[0] - r_trans[0], pos_goal[1] - r_trans[1]])
            return err

        err = compute_rot_err()
        while abs(err) > rot_tolerance:  # 0.05 rad tolerance (~3°)
            if err > 0:
                self._send_velocity_for_wheel(-speed, speed) 
            else:
                self._send_velocity_for_wheel(speed, -speed)

            yield ()

            err = compute_rot_err()

        self._send_velocity_for_wheel(speed,speed)

        err = compute_dist_err() 
        while err > trans_tolerance: # 0.05 m
            linear = 2 * speed * abs(err-(trans_tolerance/2))
            linear = np.clip(linear,-speed,speed)
            self._send_velocity_for_wheel(linear,linear)
            yield ()
            err = compute_dist_err()

        self._send_velocity_for_wheel(0,0)
        
        return True
    

    # def move_with_controllers(self, xg, yg, theta):
    #     r_trans, r_ori = self._articulation.get_world_pose()
    #     ex = xg - r_trans[0]
    #     ey = yg - r_trans[1]

    #     k_rho = 1.0      # distance gain
    #     k_alpha = 1.5    # heading gain

    #     rho = math.sqrt(ex**2 + ey**2)
    #     goal_heading = math.atan2(ey, ex)
    #     alpha = goal_heading - theta

    #     # Forward velocity slows down when near target
    #     desired_forward_vel = min(2.0, k_rho * rho)

    #     # Steering proportional to heading error
    #     desired_steering_angle = max(-0.5, min(0.5, k_alpha * alpha))

    #     orientation_error = theta_g - theta