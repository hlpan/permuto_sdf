#!/usr/bin/env python3

#this scripts shows how to run PermutoSDF on your own custom dataset
#You would need to modify the function create_custom_dataset() to suit your needs. The current code is setup to read from the easypbr_render dataset (see README.md for the data) but you need to change it for your own data. The main points are that you need to provide an image, intrinsics and extrinsics for each your cameras. Afterwards you need to scale your scene so that your object of interest lies within the bounding sphere of radius 0.5 at the origin.

#CALL with ./permuto_sdf_py/experiments/run_custom_dataset/run_custom_dataset.py --exp_info test [--no_viewer]

import torch
import argparse
import os
import natsort
import numpy as np

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import TrainParams
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_mesh
from permuto_sdf_py.utils.permuto_sdf_utils import get_frames_cropped
from permuto_sdf_py.train_permuto_sdf import train
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes
from permuto_sdf import NGPGui

from dataloaders import *
from permuto_sdf  import Sphere
torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from PIL import Image
import torchvision.transforms.functional as TF
from colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

parser = argparse.ArgumentParser(description='Train sdf and color')
parser.add_argument('--dataset', default="custom", help='Dataset name which can also be custom in which case the user has to provide their own data')
parser.add_argument('--dataset_path', default="/media/rosu/Data/data/permuto_sdf_data/easy_pbr_renders/head/", help='Dataset path')
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
parser.add_argument('--scene_scale', default=1.0, type=float, help='Scale of the scene so that it fits inside the unit sphere')
parser.add_argument('--scene_translation', default=[0,0,0], type=float, nargs=3, help='Translation of the scene so that it fits inside the unit sphere')
parser.add_argument('--img_subsample', default=1.0, type=float, help="The higher the subsample value, the smaller the images are. Useful for low vram")
args = parser.parse_args()
with_viewer=not args.no_viewer


#MODIFY these for your dataset!
SCENE_SCALE=args.scene_scale
SCENE_TRANSLATION=args.scene_translation
SCENE_SCALE=0.25
SCENE_TRANSLATION=[0.11903904248331793,0.24058634782431088,0.19656414617581808]
IMG_SUBSAMPLE_FACTOR=args.img_subsample #subsample the image to lower resolution in case you are running on a low VRAM GPU. The higher this number, the smaller the images
DATASET_PATH=args.dataset_path #point this to wherever you downloaded the easypbr_data (see README.md for download link)
 
def create_custom_dataset():
    #CREATE CUSTOM DATASET---------------------------
    #We need to fill easypbr.Frame objects into a list. Each Frame object contains the image for a specific camera together with extrinsics and intrinsics
    #intrinsics and extrinsics
    assert os.path.exists(DATASET_PATH), "The dataset path does not exist. Please point to the path where you downloaded the EasyPBR renders"


    intrinics_extrinsics_file=os.path.join(DATASET_PATH,"poses_and_intrinsics.txt")
    with open(intrinics_extrinsics_file) as file:
        lines = [line.rstrip() for line in file]
    #remove comments
    lines = [item for item in lines if not item.startswith('#')]
    #images
    path_imgs=os.path.join(DATASET_PATH,"imgs_train") #modify this to wherever your 
    imgs_names_list=[img_name for img_name in os.listdir(path_imgs)]
    imgs_names_list=natsort.natsorted(imgs_names_list,reverse=False)

    #create list of frames for this scene
    frames=[]
    for idx, img_name in enumerate(imgs_names_list):
        #load img as single precision RGB
        print("img_name", img_name)
        frame=Frame()
        img=Mat(os.path.join(path_imgs,img_name))
        img=img.to_cv32f()
        #get rgb part and possibly the alpha as a mask
        if img.channels()==4:
            img_rgb=img.rgba2rgb()
        else:
            img_rgb=img
        #get the alpha a a mask if necessary
        if args.with_mask and img.channels()==4:
            img_mask=img.get_channel(3)
            frame.mask=img_mask
        if args.with_mask and not img.channels()==4:
            exit("You are requiring to use a foreground-background mask which should be in the alpha channel of the image. But the image does not have 4 channels")
        frame.rgb_32f=img_rgb 

        #img_size
        frame.width=img.cols
        frame.height=img.rows

        #intrinsics as fx, fy, cx, cy
        calib_line=lines[idx]
        calib_line_split=calib_line.split()
        K=np.identity(3)
        K[0][0]=calib_line_split[-4] #fx
        K[1][1]=calib_line_split[-3] #fy
        K[0][2]=calib_line_split[-2] #cx
        K[1][2]=calib_line_split[-1] #cy
        frame.K=K

        #extrinsics as a tf_cam_world (transformation that maps from world to camera coordiantes)
        translation_world_cam=calib_line_split[1:4] #translates from cam to world
        quaternion_world_cam=calib_line_split[4:8] #rotates from cam to world
        tf_world_cam=Affine3f()
        tf_world_cam.set_quat(quaternion_world_cam) #assumes the quaternion is expressed as [qx,qy,qz,qw]
        tf_world_cam.set_translation(translation_world_cam)
        tf_cam_world=tf_world_cam.inverse() #here we get the tf_cam_world that we need
        frame.tf_cam_world=tf_cam_world
        #ALTERNATIVELLY if you have already the extrinsics as a numpy matrix you can use the following line
        # frame.tf_cam_world.from_matrix(YOUR_4x4_TF_CAM_WORLD_NUMPY_MATRIX) 

        #scale scene so that the object of interest is within a sphere at the origin with radius 0.5
        tf_world_cam_rescaled = frame.tf_cam_world.inverse()
        translation=tf_world_cam_rescaled.translation().copy()
        translation*=SCENE_SCALE
        translation+=SCENE_TRANSLATION
        tf_world_cam_rescaled.set_translation(translation)
        frame.tf_cam_world=tf_world_cam_rescaled.inverse()

        #subsample the image to lower resolution in case you are running on a low VRAM GPU
        frame=frame.subsample(IMG_SUBSAMPLE_FACTOR)

        #append to the scene so the frustums are visualized if the viewer is enabled
        frustum_mesh=frame.create_frustum_mesh(scale_multiplier=0.06)
        Scene.show(frustum_mesh, "frustum_mesh_"+str(idx))

        #finish
        frames.append(frame)
    
    return frames

class ColmapData():

    def __init__(self, root_dir, img_downscale):
        "read colmap dataset"
        camdata = read_cameras_binary(os.path.join(root_dir, 'sparse/0/cameras.bin'))
        imdata = read_images_binary(os.path.join(root_dir, 'sparse/0/images.bin'))

        
        all_size, all_fxy, all_cxy = [], [], []
        all_R = []
        all_C = []
        all_images = []
        frames = []
        #camera to world coordinate
        #intrinsic, fx, fy, width, height
        cam_poses = torch.empty(size=(len(imdata.values()),3), dtype=torch.float32, device=torch.device('cpu'))
        for i, d in enumerate(imdata.values()):
            # if i > 0:
            #     continue
            R = d.qvec2rotmat()
            t = d.tvec.reshape(3, 1)

            cam_pos = -R.T@t
            cam_pos = cam_pos-np.array(SCENE_TRANSLATION).reshape(3,1)
            cam_pos = cam_pos*SCENE_SCALE
            #c2w = torch.from_numpy(np.concatenate([R.T, cam_pos], axis=1)).float()
            #c2w[:,1:2] *= -1. # COLMAP => OpenGL
            all_R.append(torch.tensor(R.T, dtype=torch.float32, device='cpu'))
            all_C.append(torch.tensor(cam_pos, dtype=torch.float32, device='cpu'))

            camera = camdata[d.camera_id]

            img_width = int(camera.width* img_downscale)
            img_height= int(camera.height* img_downscale)
            all_size.append(torch.tensor([[img_width], [img_height]], dtype=torch.int32, device='cpu'))
            

            # TODO scale
            if camera.model == 'SIMPLE_RADIAL':
                fx = fy = camera.params[0] * img_downscale
                cx = camera.params[1] * img_downscale
                cy = camera.params[2] * img_downscale
            elif camera.model in ['PINHOLE', 'OPENCV']:
                fx = camera.params[0] * img_downscale
                fy = camera.params[1] * img_downscale
                cx = camera.params[2] * img_downscale
                cy = camera.params[3] * img_downscale
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")

            all_fxy.append(torch.tensor([[fx],[fy]], dtype=torch.float32, device='cpu'))
          
            all_cxy.append(torch.tensor([[cx],[cy]], dtype=torch.float32, device='cpu'))

            img_path = os.path.join(root_dir, "images", d.name)
            img = Image.open(img_path)
            
            if img_downscale != 1:
                img = img.resize(size=(img_width, img_height))

            img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
            
        
            all_images.append(img)


            frame=Frame()

            #img_size
            frame.width=int(camera.width* img_downscale)
            frame.height=int(camera.height* img_downscale)

            #intrinsics as fx, fy, cx, cy
            K=np.identity(3)
            K[0][0]=fx
            K[1][1]=fy
            K[0][2]=cx
            K[1][2]=cy
            frame.K=K

            

            b = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device='cpu')
            c = torch.cat( (torch.tensor(np.concatenate([R.T, cam_pos], axis=1), dtype=torch.float32, device='cpu'), b), dim=0)
            frame.tf_cam_world.from_matrix(c)
            frame.tf_cam_world = frame.tf_cam_world.inverse()

            # img=Mat(os.path.join(root_dir, "images", d.name))

            #frame.rgb_32f.from_numpy(img.numpy())
            img=Mat(img_path)
            img=img.to_cv32f()
            #get rgb part and possibly the alpha as a mask
            if img.channels()==4:
                img_rgb=img.rgba2rgb()
            else:
                img_rgb=img
            #get the alpha a a mask if necessary
            if args.with_mask and img.channels()==4:
                img_mask=img.get_channel(3)
                frame.mask=img_mask
            if args.with_mask and not img.channels()==4:
                exit("You are requiring to use a foreground-background mask which should be in the alpha channel of the image. But the image does not have 4 channels")
            frame.rgb_32f=img_rgb 


            #extrinsics as a tf_cam_world (transformation that maps from world to camera coordiantes)
            #tf_cam_world=Affine3f()

            # in colmap the quaternion is expressed as [qw, qx,qy,qz]
            # in eigen the quaternion is expressed as [qx,qy,qz,qw]
            # qw, qx,qy,qz = d.qvec
            # tf_cam_world.set_quat(np.array([qx,qy,qz,qw]))
            # tf_cam_world.set_translation(t)
            # frame.tf_cam_world=tf_cam_world


            # ALTERNATIVELLY if you have already the extrinsics as a numpy matrix you can use the following line
            #frame.tf_cam_world.from_matrix(YOUR_4x4_TF_CAM_WORLD_NUMPY_MATRIX) 

            #scale scene so that the object of interest is within a sphere at the origin with radius 0.5
            # tf_world_cam_rescaled = frame.tf_cam_world.inverse()
            # translation=tf_world_cam_rescaled.translation().copy()
            # translation*=SCENE_SCALE
            # translation+=SCENE_TRANSLATION
            # tf_world_cam_rescaled.set_translation(translation)
            # frame.tf_cam_world=tf_world_cam_rescaled.inverse()

            # #subsample the image to lower resolution in case you are running on a low VRAM GPU
            # frame=frame.subsample(IMG_SUBSAMPLE_FACTOR)

            #append to the scene so the frustums are visualized if the viewer is enabled
            # frustum_mesh=frame.create_frustum_mesh(scale_multiplier=0.06)
            # Scene.show(frustum_mesh, "frustum_mesh_"+str(i))

            #finish
            frames.append(frame)

            #cam_pos[i] = c2w[..., -1:]

        self.frames = frames
        self.all_images = all_images

        #rotation from camera to world
        self.all_R = torch.stack(all_R, dim=0).to('cuda:0')   

        #camera pos
        self.all_C = torch.stack(all_C, dim=0).to('cuda:0')   

        self.all_size = torch.stack(all_size, dim=0).to('cuda:0')      
        self.all_fxy = torch.stack(all_fxy, dim=0).to('cuda:0')      
        self.all_cxy = torch.stack(all_cxy, dim=0).to('cuda:0')      

        pts3d = read_points3d_binary(os.path.join(root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])
        pts3d = pts3d - SCENE_TRANSLATION
        pts3d = pts3d*SCENE_SCALE

        points_mesh=Mesh()
        points_mesh.V=pts3d
        points_mesh.m_vis.m_show_points = True
        points_mesh.m_vis.m_point_color = [0.1, 0.9, 0.1]
        Scene.show(points_mesh, "colmap_points")

        pos_mesh=Mesh()
        pos_mesh.V=self.all_C.cpu().numpy().squeeze()
        pos_mesh.m_vis.m_show_points = True
        Scene.show(pos_mesh, "cam_points")

        #aabb=Sphere(2.0, [0.31,1.78,1.72])
        aabb=Sphere(0.5, [0,0,0])
        bb_mesh = create_bb_mesh(aabb) 
        bb_mesh.m_vis.m_line_color = [0.1, 0.9, 0.1]
        Scene.show(bb_mesh,"my_bb_mesh")

def run():

    config_file="train_permuto_sdf.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)
    train_params=TrainParams.create(config_path)
    hyperparams=HyperParamsPermutoSDF()


    #get the checkpoints path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints/custom_dataset")
    os.makedirs(checkpoint_path, exist_ok=True)

    
    train_params.set_with_tensorboard(True)
    train_params.set_save_checkpoint(True)
    print("checkpoint_path",checkpoint_path)
    print("with_viewer", with_viewer)

    experiment_name="custom"
    if args.exp_info:
        experiment_name+="_"+args.exp_info
    print("experiment name",experiment_name)


    #CREATE CUSTOM DATASET---------------------------
    #frames=create_custom_dataset() 

    # loader_train=DataLoaderColmap(config_path)
    # loader_train.set_mode_train()
    # loader_train.start()
    #print the scale of the scene which contains all the cameras.
    print("scene centroid", Scene.get_centroid()) #aproximate center of our scene which consists of all frustum of the cameras
    print("scene scale", Scene.get_scale()) #how big the scene is as a measure betwen the min and max of call cameras positions

    ##VISUALIZE
    # view=Viewer.create()
    # while True:
        # view.update()


    ####train
    #tensor_reel=MiscDataFuncs.frames2tensors(frames) #make an tensorreel and get rays from all the images at 

    #tensoreel
    # all_frame = []
    # for i in range(loader_train.nr_samples()):
    #     all_frame.append(loader_train.get_frame_at_idx(i))
    # tensor_reel=MiscDataFuncs.frames2tensors( all_frame ) #make an tensorreel and get rays from all the images at

    #colmap_dataset = ColmapData("/workspace/home/nerf-data/tanksandtemple/Truck/dense/", 1)
    #colmap_dataset = ColmapData("/workspace/home/nerf-data/germany/dense/", 0.1)
    colmap_dataset = ColmapData("/workspace/home/nerf-data/tanksandtemple/Ignatius2/dense/", 1)
    #tensor_reels=MiscDataFuncs.frames2tensors(colmap_dataset.frames)
    tensor_reels = None
    train(args, config_path, hyperparams, train_params, None, experiment_name, with_viewer, checkpoint_path, tensor_reel=tensor_reels, frames_train=colmap_dataset.frames, hardcoded_cam_init=False, colmap_data=colmap_dataset)



def main():
    run()



if __name__ == "__main__":
    # config_file="train_permuto_sdf.cfg"
    # config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)
    # view=Viewer.create(config_path)
    # ngp_gui=NGPGui.create(view)
    # colmap_dataset = ColmapData("/workspace/home/nerf-data/germany/dense/", 1)

    # while(True):
    #     view.update()
    # pass
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
