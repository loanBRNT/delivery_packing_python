# Main code for running a server for the M2T2 method for grasp and place estimation
# Official github : https://github.com/NVlabs/M2T2
# Official website : https://m2-t2.github.io/
# Author : Loan BERNAT (l.bernat@sileane.com)

from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu

# ruff: noqa: E402
import json_numpy
import hydra
import torch

json_numpy.patch()
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np

from typing import Any, Dict, Optional, Union

from m2t2.meshcat_utils import (
    create_visualizer, visualize_grasp, visualize_pointcloud, make_frame)
import meshcat.geometry as g
import meshcat.transformations as tf

from scipy.spatial.transform import Rotation as R

import socket

def segment_to_color(segmented_array):
    # Initialize a color array of shape (N, 3)
    N = len(segmented_array)
    color_array = np.zeros((N, 3))  # All values initially set to [0, 0, 0] (black)
    
    # Map the segmented values to colors (example: non-zero values get a color)
    for i in range(N):
        if segmented_array[i] != 0:
            color_array[i] = [1, 0, 0]  # Red for non-zero values (example)
        else:
            color_array[i] = [0, 0, 0]  # Black for zero values (or another background color)
    
    return color_array

def normalize_point_cloud_rgb(rgb):
    """
    Normalize point cloud RGB data.

    Args:
        rgb (torch.Tensor): Tensor of shape [N, 3] representing RGB values for N points.
    
    Returns:
        torch.Tensor: Normalized RGB tensor of shape [N, 3].
    """

    # Mean and standard deviation for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Normalize RGB values
    normalized_rgb = (rgb - mean) / std

    return normalized_rgb

def draw_frame(vis, name, transform, h=0.05, o=1.0, radius=0.002):
    """
    Draws a coordinate frame at the given transform in MeshCat.

    - `vis`: the MeshCat Visualizer
    - `name`: name of the group
    - `transform`: 4x4 homogeneous transform
    - `length`: axis length
    - `radius`: cylinder radius
    """
    transform = np.array(transform)  # Ensure it's a NumPy array
    
    # X-axis (red)
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8),
    )
    x_tf = transform @ tf.translation_matrix([h / 2, 0, 0]) @ tf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    vis[name]["x"].set_transform(x_tf)

    # Y-axis (green)
    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8),
    )
    y_tf = transform @ tf.translation_matrix([0, h / 2, 0]) @ tf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    vis[name]["y"].set_transform(y_tf)

    # Z-axis (blue)
    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8),
    )
    z_tf = transform @ tf.translation_matrix([0, 0, h / 2])  # No rotation needed
    vis[name]["z"].set_transform(z_tf)

class M2T2Server:
    eval_checkpoint = "/m2t2.pth"

    extrinsic_param = None #Convention OpenGL

    distance_threshold = 0.08
    confidence_threshold = 0.8
    nb_grasp = 5
    bvis = False
    bfullvis = False

    # Init important parameter from config file
    def __init__(self,cfg):
        self.cfg = cfg

        self.model = M2T2.from_config(self.cfg.m2t2)
        ckpt = torch.load(self.eval_checkpoint)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.cuda().eval()

    def initialisation(self, payload: Dict[str, Any]) -> str:
        ex = np.array(payload['ex_param'])
        self.extrinsic_param = ex.squeeze(0)
        
        self.distance_threshold = payload['options']['max_distance']
        self.nb_grasp = payload['options']['min_nb_grasp']
        self.confidence_threshold = payload['options']['min_confidence']

        if payload['options']['visualize'] == "filtred":
            self.bvis = True
        elif payload['options']['visualize'] == "all":
            self.bfullvis = True
        
        return JSONResponse({})
    
    # Function called by Grasp Interface.
    # Responding with a dict of {"grasps":[],"conf":[],"contact":[]}
    def grasp(self, payload: Dict[str, Any]) -> str:
        
        xyz = torch.tensor(payload['xyz'])
        rgb = torch.tensor(payload['rgb'])
        seg = torch.tensor(payload['seg'])

        rgb = normalize_point_cloud_rgb(rgb)

        obj_xyz, obj_rgb = torch.tensor(payload['obj_xyz']), torch.tensor(payload['obj_rgb'])

        obj_xyz_grid = torch.unique(
            (obj_xyz[:, :2] / 0.01).round(), dim=0
        ) * 0.01
        bottom_center = obj_xyz.min(dim=0)[0]
        bottom_center[:2] = obj_xyz_grid.mean(dim=0)

        if "scene_bound" in payload:
            g_xyz, g_rgb, g_seg = xyz, rgb, seg
            bounds = payload['scene_bound']
            mask = (xyz[:, 0] > bounds[0]) & (xyz[:, 0] < bounds[3]) \
            & (xyz[:, 1] > bounds[1]) & (xyz[:, 1] < bounds[4]) \
            & (xyz[:, 2] > bounds[2]) & (xyz[:, 2] < bounds[5])
            xyz, rgb, seg = xyz[mask], rgb[mask], seg[mask]
        else:
           g_xyz, g_rgb, g_seg = None, None, None 
        
        data = {
            'inputs': torch.cat([xyz - xyz.mean(dim=0), rgb], dim=1),
            'points': xyz,
            'seg': seg,
            'cam_pose': torch.tensor(self.extrinsic_param),
            'task' : 'pick',
            'ee_pose': torch.tensor(payload['ee_pose']),
            'object_inputs': torch.cat([
                obj_xyz - obj_xyz.mean(dim=0), obj_rgb
            ], dim=1),
            'bottom_center': bottom_center,
            'object_center': obj_xyz.mean(dim=0)
        }

        inputs = data['inputs']

        obj_inputs = data['object_inputs']
  
        outputs = {
            'grasps': [],
            'grasp_confidence': [],
            'grasp_contacts': [],
            'placements': [],
            'placement_confidence': [],
            'placement_contacts': []
        }
        for _ in range(self.cfg.eval.num_runs):
            pt_idx = sample_points(xyz, self.cfg.data.num_points)
            data['inputs'] = inputs[pt_idx]
            data['points'] = xyz[pt_idx]
            data['seg'] = seg[pt_idx]
            pt_idx = sample_points(obj_inputs, self.cfg.data.num_object_points)
            data['object_inputs'] = obj_inputs[pt_idx]
            data_batch = collate([data])
            to_gpu(data_batch)

            with torch.no_grad():
                model_ouputs = self.model.infer(data_batch, self.cfg.eval)
            to_cpu(model_ouputs)
            for key in outputs:
                if 'place' in key and len(outputs[key]) > 0:
                    outputs[key] = [
                        torch.cat([prev, cur])
                        for prev, cur in zip(outputs[key], model_ouputs[key][0])
                    ]
                else:
                    outputs[key].extend(model_ouputs[key][0])

        if self.bfullvis:
            self.visualiseFull(outputs, xyz.numpy(), rgb.numpy())

        # Remove grasp too far or with not enought confidence
        out = self.simplify(outputs,obj_xyz)

        if self.bvis:
            self.visualiseFiltred(out, xyz.numpy(),rgb.numpy())

        l = len(out['conf'])
        if l == 0:
            return JSONResponse({'grasp':[]})
        elif l == 1:
            return JSONResponse({'grasp':out['grasps'][0].numpy().tolist()})
        
        rand_i = torch.randint(0, len(out['conf']), (1,)).item()

        # vis = create_visualizer() DEBUG FRAME

        # visualize_pointcloud(vis, 'scene', xyz.numpy(), rgb.numpy(), size=0.005)
        # visualize_grasp(
        #             vis, f"object/grasps/{1:03d}",
        #             np.array(out['grasps'][0]), [255,255,255], linewidth=0.5
        #         )
        # draw_frame(vis,'debugFrame',np.array(out['grasps'][0]))
        # make_frame(vis,'off_debugFrame',T=out['grasps'][0].double().numpy())
        # print(out['grasps'][0].double().numpy())
        # rot_matrix = out['grasps'][0].double().numpy()[:3, :3]
        # quat = R.from_matrix(rot_matrix).as_quat()
        # print(quat)

        response = {'grasp':out['grasps'][rand_i].numpy().tolist()} 

        return JSONResponse(response)

    # Make a general sort on all predicted grasp for the scene to keep only the best on desired object
    def simplify(self, outputs, obj : torch.Tensor):
        filtred_outputs = {"grasps":torch.Tensor([]),"conf":torch.Tensor([])}
        # Cette approche est cool car on pourrait parralleliser pour gagner du temps
        # + Cela assure une certaine diversité (e.g pas que des grasps sur une partie de l'objet)
        for i, (grasps, conf, contacts) in enumerate(zip(
                outputs['grasps'],
                outputs['grasp_confidence'],
                outputs['grasp_contacts']
            )):
            distances = torch.cdist(contacts, obj)  # [M, N]
            min_values, _ = torch.min(distances, dim=1)  # [M]

            # Get indices where the min is below the threshold
            indices = torch.nonzero(min_values < self.distance_threshold, as_tuple=True)[0]

            conf, grasps = conf[indices], grasps[indices]

            if len(conf) != 0:
                if self.nb_grasp < len(grasps):               
                    top_indices = torch.argsort(conf)[-self.nb_grasp:]
                    conf, grasps = conf[top_indices], grasps[top_indices]
                
                filtred_outputs['grasps'] = torch.cat((filtred_outputs['grasps'], grasps))
                filtred_outputs['conf'] = torch.cat((filtred_outputs['conf'], conf))
        
        # On garde que les grasp avec une conf superieure OU on prend le meilleur
        if len(filtred_outputs['conf']) != 0:
            indices = torch.nonzero(filtred_outputs['conf'] > self.confidence_threshold, as_tuple=True)[0]
            if indices := []:
                filtred_outputs['conf'], filtred_outputs['grasps'] = filtred_outputs['conf'][indices], filtred_outputs['grasps'][indices]
            '''else :
                ind = torch.argmax(filtred_outputs['conf'])
                filtred_outputs['conf'] = [filtred_outputs['conf'][ind]]
                filtred_outputs['grasps'] = [filtred_outputs['grasps'][ind]]'''

        print("Nb grasp > threshold distance + confidence :", len(filtred_outputs['grasps']))
        return filtred_outputs

    # Visualize the differents generated grasps
    def visualiseFiltred(self, outputs, xyz, rgb):
        vis = create_visualizer()

        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
        for j in range(len(outputs['grasps'])):
            grasp = outputs['grasps'][j]
            visualize_grasp(
                    vis, f"object/grasps/{j:03d}",
                    np.array(grasp), [255,255,255], linewidth=0.5
                )
            
     # Visualize the differents generated grasps
    def visualiseFull(self, outputs, xyz, rgb):
        vis = create_visualizer()

        visualize_pointcloud(vis, 'scene', xyz, rgb, size=0.005)
        for i, (grasps, confs) in enumerate(zip(
                outputs['grasps'],
                outputs['grasp_confidence']
            )):
            for j, (grasp, conf) in enumerate(zip(grasps,confs)):
                visualize_grasp(
                        vis, f"object{i:03d}/grasps/{j:03d}",
                        grasp.numpy(), value_to_color(conf.item()), linewidth=0.2
                    )
    
    def run(self, host: str = "MAGMA_grasp", port: int = 8001) -> None:
        self.app = FastAPI()
        self.app.post("/grasp")(self.grasp)
        self.app.post("/init")(self.initialisation)
        uvicorn.run(self.app, host=host, port=port)

def value_to_color(value: float) -> list:
    """Converts a value between 0 and 1 to a color between red and green."""
    value = max(0, min(1, value))  # Ensure value is within range [0,1]
    red = int(255 * (1 - value))
    green = int(255 * value)
    return [red, green, 0]

@hydra.main(config_path='/home/grasp/M2T2/', config_name='config', version_base='1.3')
def main(cfg):
    server = M2T2Server(cfg)
    server.run(host=socket.gethostname())

if __name__ == "__main__":
    main()