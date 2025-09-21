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

from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane, VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics, transformations
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices, quats_to_euler_angles, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from pxr import UsdPhysics, Gf

ROBOT_POS = np.array([0.,0.1,0])
ROBOT_ORI = euler_angles_to_quats(np.array([0.,0.,np.pi]))

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

        #conveyor
        conveyor_asset_path = get_assets_root_path() +  "/Isaac/Props/Conveyors/ConveyorBelt_A08.usd"
        conveyor_prim_path = "/World/Conveyor"
        add_reference_to_stage(conveyor_asset_path, conveyor_prim_path)
        self._conveyor =  SingleXFormPrim(
            prim_path=conveyor_prim_path,
            name="conveyor",
            position=np.array([4.0, 0, 0]),
            # orientation=euler_angles_to_quats([0, 0, np.deg2rad(90)]),
        )

        shelve_asset_path = os.path.join(self.general_asset_path, "shelve_large","shelve_large.usd")
        shelve_prim_path = "/World/Shelve2"
        add_reference_to_stage(shelve_asset_path, shelve_prim_path)
        self._shelve = SingleXFormPrim(
            prim_path=shelve_prim_path,
            name="shelve2",
            position=np.array([-1.2, 0, 0.75]),
            scale=np.array([1.0, 1.0, 0.7]),
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

        

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._conveyor, self._articulation, self._tray

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """

        self._set_arms_position(0,1.2217,0,1.2217,np.pi/2,0.349)

        self._move_controller = DifferentialController(name="diff_ctrl", wheel_radius=0.2, wheel_base=0.56)

        self._wheel_indices = [self._articulation.get_dof_index('left_wheel_f'),
        self._articulation.get_dof_index('right_wheel_f'),
        self._articulation.get_dof_index('left_wheel_d'),
        self._articulation.get_dof_index('right_wheel_d')]

        self._gripper_indice = [
            self._articulation.get_dof_index('left_Left_2_Joint'),
            self._articulation.get_dof_index('left_Right_2_Joint'),
            self._articulation.get_dof_index('right_Left_2_Joint'),
            self._articulation.get_dof_index('right_Right_2_Joint')
        ]

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


        self._open_gripper()

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
        self._open_gripper()
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

    def _open_gripper(self):
        action = ArticulationAction(joint_positions=[np.pi/4,-np.pi/4,np.pi/4,-np.pi/4],
                                    joint_indices=self._gripper_indice)
        self._articulation.apply_action(action)

    def _close_gripper(self):
        action = ArticulationAction(joint_positions=[0,0,0,0],
                                    joint_indices=self._gripper_indice)
        self._articulation.apply_action(action)


    def move_step(self, goal_pose, dt=1/60.0, pos_tol=0.05, yaw_tol=0.05, timeout=10000):
        """
        Allow to move the robot to a goal pose N,7 pos + orientation.
        """
        
        # self._send_velocity_for_wheel(3,-3)

        robot_base_translation,robot_base_orientation = self._articulation.get_world_pose()
        t = robot_base_translation + [-0.2,-0.1,1]
        
        self._kinematics_solver.set_robot_base_pose(robot_base_translation,robot_base_orientation)

        print(self._articulation_kinematics_solver.compute_end_effector_pose())


        t = np.array([-0.65640712, -0.1 ,  0.9307782])
        l_action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(t)


        self._articulation.apply_action(l_action)
        r_action = self._mirror_action_for_right_arm(l_action)
        self._articulation.apply_action(r_action)

        # print(success, action)

        for i in range(1000):
            ee_trans, ee_ori = self._articulation_kinematics_solver.compute_end_effector_pose()
            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, t)
            if trans_dist <= 0.01:
                print(i)
                break
            yield ()

        t = np.array([-0.71, -0.1 ,  0.95])
        l_action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(t)

        print(success, l_action)

        self._articulation.apply_action(l_action)
        r_action = self._mirror_action_for_right_arm(l_action)
        self._articulation.apply_action(r_action)

        for i in range(1000):
            ee_trans, ee_ori = self._articulation_kinematics_solver.compute_end_effector_pose()
            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, t)
            if trans_dist <= 0.01:
                print(i)
                break
            yield ()

        # self._close_gripper()

        # for i in range(1000):
        #     gripper_pos = self._articulation.get_joint_positions(joint_indices=self._gripper_indice)
        #     print(gripper_pos)
        #     if abs(abs(gripper_pos[0])-abs(gripper_pos[1])) <= 0.3 and abs(abs(gripper_pos[2])-abs(gripper_pos[3])) <= 0.3:
        #         print(i)
        #         break
        #     yield ()

        # # t = np.array([-0.71, -0.1 , 1.05])
        # # l_action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(t)

        # # print(success, l_action)

        # # self._articulation.apply_action(l_action)
        # # r_action = self._mirror_action_for_right_arm(l_action)
        # # self._articulation.apply_action(r_action)

        # # for i in range(1000):
        # #     ee_trans, ee_ori = self._articulation_kinematics_solver.compute_end_effector_pose()
        # #     trans_dist = distance_metrics.weighted_translational_distance(ee_trans, t)
        # #     if trans_dist <= 0.01:
        # #         print(i)
        # #         break
        # #     yield ()

        # self._send_velocity_for_wheel(-1,-1)


        return False
    

    # (array([-0.65640712, -0.1837069 ,  0.77307782]), 
    # 
    # array([[ 0.01631553, -0.39495108, -0.91855727],
    #    [ 0.99344906,  0.11032455, -0.02979036],
    #    [ 0.11310515, -0.9120538 ,  0.39416378]]))