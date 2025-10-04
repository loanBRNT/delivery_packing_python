# Author Loan BERNAT

import numpy as np
import os
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.storage.native import get_assets_root_path

from . import global_variables



ROBOT_POS = np.array([0.0, -1.5, 0.5])
ROBOT_ORI = euler_angles_to_quats([0, 0, 90])

bottle_offset = {
    "pre" : np.array([0,0.1,0.1]),
    "cur" : np.array([0,0,0.1]),
    "post" : np.array([0,0.1,0.05])
}
book_offset = {
    "pre" : np.array([0.03,0.01,0.1]),
    "cur" : np.array([0.03,0.01,0.04]),
    "post" : np.array([0.03,0,0.1])
}
tape_offset = {
    "pre" : np.array([0,0.1,0.05]),
    "cur" : np.array([0,0,0.01]),
    "post" : np.array([0,0.1,0.05])
}

pill_offset = {
    "pre" : np.array([0,0,0.1]),
    "cur" : np.array([0,0,0]),
    "post" : np.array([0,0,0.1])
}
 
close_position = {
    "bottle" : np.array([0.005,0.005]),
    "book" : np.array([0.001,0.001]),
    "tape" : np.array([0.0105,0.0105]),
    "pill" : np.array([0,0])
}

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
        robot_prim_path = "/robots/packing_robot"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = SingleArticulation(
            name="packing_robot",
            prim_path=robot_prim_path,
            position=ROBOT_POS,
            orientation=ROBOT_ORI
        )

        # side table
        side_table_prim_path = "/World/side_table"
        side_table_usd_path = get_assets_root_path() + "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd"
        add_reference_to_stage(side_table_usd_path,side_table_prim_path)
        self._side_table = SingleXFormPrim(
            name="side_table",
            prim_path=side_table_prim_path,
            position=np.array([0.7, -1.5, 0.4]),
            orientation=euler_angles_to_quats([0, 0, np.deg2rad(90)]),
            scale=np.array([1,1,0.6])
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
        
        # Isaac/5.0/Isaac/Environments/Hospital/Props/SM_BottleB SM_SHelf_06

        FixedCuboid(
            name = "packing_plateforme",
            prim_path="/World/packing_support",
            position=np.array([0.0, -1.4, 0.15]),
            scale=np.array([2.0,1.05,0.5])
        )

        self.obstacles = [
            VisualCuboid(
                prim_path="/World/Obstacles/TableProxy",
                name="obstacle1",
                position=np.array([0., -1.0, 0.58]),
                scale=np.array([1.0, 0.5, 0.3]),     
                visible=False,             
            ), 

            VisualCuboid(
                prim_path="/World/Obstacles/SideTableProxy",
                name="obstacle2",
                position=np.array([0.7, -1.5, 0.73]),
                scale=np.array([0.4, 1, 0.3]),     
                visible=False,             
            )

        ]
        
        # tape
        tape_asset_path = os.path.join(general_asset_path, "measuring_tape", "tape", "tape.usd")
        add_reference_to_stage(tape_asset_path,"/tapes/tape1")
        add_reference_to_stage(tape_asset_path,"/tapes/tape2")
        add_reference_to_stage(tape_asset_path,"/tapes/tape3")
        self._tapes = [
            SingleXFormPrim(
                name="tape_1",
                prim_path="/tapes/tape1",
                position=np.array([-0.15,-2.2,0.97]),
                orientation=euler_angles_to_quats([0, 0 ,np.pi/2])
            ),
            SingleXFormPrim(
                name="tape_2",
                prim_path="/tapes/tape2",
                position=np.array([0,-2.2,0.97]),
                orientation=euler_angles_to_quats([0, 0 ,np.pi/2])
            ),
            SingleXFormPrim(
                name="tape_3",
                prim_path="/tapes/tape3",
                position=np.array([-0.25,-2.2,1.35]),
                orientation=euler_angles_to_quats([0, 0 ,np.pi/2])
            )
            
        ]
        
        # book
        book_asset_path = os.path.join(general_asset_path, "book", "book.usd")
        add_reference_to_stage(book_asset_path, "/books/book1")
        add_reference_to_stage(book_asset_path, "/books/book2")
        self._book = [
            SingleXFormPrim(
                name="book1",
                prim_path="/books/book1",
                position=np.array([0.55,-1.4,0.90]),
                orientation=euler_angles_to_quats([np.pi/2, 0 ,np.pi/2])
            ),
            SingleXFormPrim(
                name="book2",
                prim_path="/books/book2",
                position=np.array([0.55,-1.6,0.90]),
                orientation=euler_angles_to_quats([np.pi/2, 0 ,np.pi/2])
            )
        ]

        bottle_asset_path = os.path.join(general_asset_path, "bottle", "bottle.usd")
        add_reference_to_stage(bottle_asset_path, "/bottles/bottle1")
        add_reference_to_stage(bottle_asset_path, "/bottles/bottle2")
        self._bottles = [
            SingleXFormPrim(
            name="bottle1",
            position=np.array([0., -2.15, 1.35]),
            prim_path="/bottles/bottle1",
            ),
            SingleXFormPrim(
            name="bottle2",
            position=np.array([0.3, -2.15, 0.95]),
            prim_path="/bottles/bottle2",
            )
        ]

        #pills
        pills_asset_path = os.path.join(general_asset_path, "pills", "pills.usd")
        add_reference_to_stage(pills_asset_path, "/pills/pill1")
        self._pills = [
            SingleXFormPrim(
                name="pill1",
                prim_path="/pills/pill1",
                position=np.array([0.57, -1.2, 0.90]),
                orientation=euler_angles_to_quats([np.pi/2, 0 ,0])
            )
        ]

        self._ground_plane = GroundPlane("/World/Ground")

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 25.0]),
            frequency=5,
            resolution=(256, 256),
            orientation=euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._ground_plane, *self._bottles, self._shelve, *self.obstacles, *self._book, self.camera, *self._tapes

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

        for obs in self.obstacles:
            self._rmpflow.add_obstacle(obs)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

        self.camera.initialize()
        self.camera.attach_annotator("distance_to_image_plane")
        self.camera.attach_annotator("pointcloud")

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
        global_variables.state_machine_id = 0
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like[0.00793253 0.0080996 ]
 way
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
        print(self.camera.get_pointcloud())
        while True:
            yield ()
            if self._delivery and global_variables.state_machine_id == 1:
                target_pos = np.array([0., -0.8, 1.05]) # A tuner
                base_orientation = euler_angles_to_quats([-np.pi/2,np.pi/2,0])
                for obj_name, val in self._delivery_argument.items():
                    if "book" in obj_name:
                        shelve_take_orientation = euler_angles_to_quats([np.pi,0,0])
                        offsets = book_offset
                        close_pos = close_position["book"]
                        obj_placeholder = self._book
                    elif "bottle" in obj_name:
                        shelve_take_orientation = euler_angles_to_quats([0, np.pi/2, -np.pi/2])
                        offsets = bottle_offset
                        close_pos = close_position["bottle"]
                        obj_placeholder = self._bottles
                    elif "tape" in obj_name:
                        shelve_take_orientation = euler_angles_to_quats([0, np.pi/2, -np.pi/2])
                        offsets = tape_offset
                        close_pos = close_position["tape"]
                        obj_placeholder = self._tapes
                    elif "pill" in obj_name:
                        shelve_take_orientation = euler_angles_to_quats([np.pi,0,0])
                        offsets = pill_offset
                        close_pos = close_position["pill"]
                        obj_placeholder = self._pills
                    else:
                        print("[PACKING] Unknown object : ", obj_name)
                        continue
                
                    yield from self.open_gripper_franka(self._articulation)  
                    if val > len(obj_placeholder):
                        val = len(obj_placeholder)
                    for i in range(val):
                        obj = obj_placeholder[i]
                        obj_pos, _ = obj.get_world_pose()
                        
                        success = yield from self.goto_position(
                            obj_pos + offsets["pre"] , shelve_take_orientation, self._articulation, self._rmpflow
                        )
                        print("[PACKING] pre take : ", success)

                        obj_pos, _ = obj.get_world_pose()

                        success = yield from self.goto_position(
                            obj_pos + offsets["cur"], shelve_take_orientation, self._articulation, self._rmpflow
                        )
                        print("[PACKING] take : ", success)

                        yield from self.close_gripper_franka(self._articulation, close_position=close_pos, atol=0.002)

                        print("[PACKING] closed")

                        success = yield from self.goto_position(
                            obj_pos + offsets["post"] , shelve_take_orientation, self._articulation, self._rmpflow, timeout=200
                        )
                        print("[PACKING] post take : ", success)

                        success = yield from self.goto_position(
                            target_pos , base_orientation, self._articulation, self._rmpflow
                        )
                        print("[PACKING] deposit : ", success)

                        yield from self.open_gripper_franka(self._articulation)
                self._delivery = False
                global_variables.state_machine_id = 2


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

    def close_gripper_franka(self, articulation, close_position=np.array([0, 0]), atol=0.01):
        # To close around the bottle, different values are passed in for close_position and atol
        open_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([7, 8]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        steps = 0
        history = []
        stable_frames = 15
        while steps < 1000:
            steps += 1
            current_pos = articulation.get_joint_positions()[7:]
            history.append(current_pos.copy())

            # Condition 1: reached target
            if np.allclose(current_pos, close_position, atol=atol):
                return True

            # Condition 2: stabilized (no movement for stable_frames)
            if len(history) >= stable_frames:
                diffs = np.ptp(np.stack(history), axis=0) 
                print(diffs)
                if np.all(diffs < 0.01):
                    return True
                history.pop(0)

            yield ()  # yield control back (async step)

        if steps >= 1000:
            return False

        return True
