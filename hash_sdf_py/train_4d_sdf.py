#!/usr/bin/env python3

import torch
import torch.nn.functional as F

import sys
import os
import numpy as np
import time as time_module
import natsort 

import easypbr
from easypbr  import *

from hash_sdf  import TrainParams
from hash_sdf  import NGPGui
from hash_sdf  import OccupancyGrid
from hash_sdf  import Sphere
from hash_sdf_py.models.models import SDF
from hash_sdf_py.utils.sdf_utils import sdf_loss
from hash_sdf_py.utils.sdf_utils import sphere_trace
from hash_sdf_py.utils.sdf_utils import filter_unconverged_points
from hash_sdf_py.utils.nerf_utils import create_rays_from_frame
from hash_sdf_py.utils.common_utils import show_points
from hash_sdf_py.utils.common_utils import tex2img
from hash_sdf_py.utils.common_utils import colormap
from hash_sdf_py.utils.aabb import AABB

from hash_sdf_py.callbacks.callback import *
from hash_sdf_py.callbacks.viewer_callback import *
from hash_sdf_py.callbacks.visdom_callback import *
from hash_sdf_py.callbacks.tensorboard_callback import *
from hash_sdf_py.callbacks.state_callback import *
from hash_sdf_py.callbacks.phase import *


config_file="train_4d_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)


#loads a sequence of meshes that have the same topology and samples points temporally from the meshes
def load_mesh_sequence(folder):
    list_files=[]
    for file in os.listdir(folder):
        if file.endswith(".obj") or file.endswith(".py"): 
            list_files.append(file)
    list_files=natsort.natsorted(list_files,reverse=False)
    print("list_files",list_files)

    colormngr=ColorMngr()

    #load each mesh
    meshes_list=[]
    max_nr_meshes=min(12,len(list_files))
    for i in range(len(list_files)):
        if(i>max_nr_meshes):
            break
        file=list_files[i]
        full_path=os.path.join(folder,file)
        mesh=Mesh(full_path)
        #we normalize the first one and then all the rest with the same parameters
        if(i==0):
            scale=mesh.normalize_size()
            pos=mesh.normalize_position()
        else:
            mesh.scale_mesh(1.0/scale)
            mesh.translate_model_matrix(pos)
        mesh.apply_model_matrix_to_cpu(True)
        mesh.recalculate_normals()

        mesh.upsample(2,True)

        #remove hidden parts of the mesh
        if i==0:
            mesh.compute_embree_ao(100) 
            C=mesh.C
            too_occluded=C[:,0:1]<0.01
        too_occluded=too_occluded.flatten()
        mesh.remove_marked_vertices(too_occluded,False)
        mesh.remove_unreferenced_verts()
        mesh.sanity_check()

        meshes_list.append(mesh)

    #color the meshes by time
    for i in range(len(meshes_list)):
        mesh=meshes_list[i]
        #calculate time and color them by time
        time=i/(len(meshes_list)-1)
        print("time",time)
        time_array=np.empty(mesh.V.shape[0])
        time_array.fill(time)
        C=colormngr.eigen2color(time_array, "viridis")
        mesh.C=C
        mesh.m_vis.set_color_pervertcolor()
        #show each one of these
        if(i==0):
            Scene.show(mesh, "m_seq"+str(i))

    

    #multiply and interpolate in between pairs of meshes
    new_meshes_list=[]
    nr_multiplicity=20
    for i in range(len(meshes_list)-1):
        cur_mesh=meshes_list[i]
        next_mesh=meshes_list[i+1]
        for j in range(nr_multiplicity):
            interpol_val=j/nr_multiplicity
            mesh=cur_mesh.interpolate(next_mesh, interpol_val)
            mesh.recalculate_normals()
            new_meshes_list.append(mesh)
    meshes_list=new_meshes_list



    #sample points from the meshes and concatenate a time dimension
    samples_pos_list=[]
    samples_normal_list=[]
    for i in range(len(meshes_list)):
        mesh=meshes_list[i]
        time=i/(len(meshes_list)-1)
        print("time",time)
        gt_points=torch.from_numpy(mesh.V.copy()).cuda().float()
        gt_time=torch.empty((mesh.V.shape[0],1))
        gt_time.fill_(time)
        gt_points_time=torch.cat([gt_points,gt_time],1)
        gt_normals=torch.from_numpy(mesh.NV.copy()).cuda().float()
        #append
        samples_pos_list.append(gt_points_time)
        samples_normal_list.append(gt_normals)

    gt_points_time=torch.cat(samples_pos_list,0) 
    gt_normals=torch.cat(samples_normal_list,0) 
    
    return gt_points_time, gt_normals

        


def run():
    # #initialize the parameters used for training
    train_params=TrainParams.create(config_path)    
    if train_params.with_viewer():
        view=Viewer.create(config_path)
        ngp_gui=NGPGui.create(view)
        view.m_camera.from_string("-0.837286  0.360068  0.310824 -0.0496414    -0.5285  -0.030986 0.846901   0.11083  0.235897 -0.152857 60 0.0502494 5024.94")

    experiment_name="sdf_def"


    #create bounding box for the scene 
    aabb=AABB(bounding_box_sizes_xyz=[1.0, 1.0, 1.0], bounding_box_translation=[0.0, 0.0, 0.0])
    

    #load the sequences of points and normals annotated with time
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    package_root=os.path.join(cur_dir,"../")
    sequence_path=os.path.join(package_root,"./data/horse_gallop")
    assert os.path.exists(sequence_path), "The sequence of meshes path does not exists. Please unzip the corresponding folder from the ./data"
    gt_points_time, gt_normals=load_mesh_sequence(sequence_path)



    cb_list = []
    if(train_params.with_tensorboard()):
        cb_list.append(TensorboardCallback(experiment_name))
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)
    phases= [ Phase('train', None, grad=True) ] 
    phase=phases[0] #we usually switch between training and eval phases but here we only train 


    #model 
    model=SDF(in_channels=4, boundary_primitive=aabb, geom_feat_size_out=0, nr_iters_for_c2f=3000).to("cuda")
    model.train(True)

    #optimizer
    optimizer = torch.optim.AdamW (model.parameters(), amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=train_params.lr())

    first_time_getting_control=True

   
    while True:
        cb.before_forward_pass() #sets the appropriate sigma for the lattice



        #sample some random point on the surface and off the surface
        with torch.set_grad_enabled(False):
            multiplier=1
            rand_indices=torch.randint(gt_points_time.shape[0],(3000*multiplier,))
            surface_points_time=torch.index_select( gt_points_time, dim=0, index=rand_indices) 
            surface_normals=torch.index_select( gt_normals, dim=0, index=rand_indices) 
            offsurface_points=aabb.rand_points_inside(nr_points=3000*multiplier)
            rand_time=torch.rand((offsurface_points.shape[0],1))
            offsurface_points_time=torch.cat([offsurface_points,rand_time],1)
            points_time=torch.cat([surface_points_time, offsurface_points_time], 0)


        #run the points through the model
        sdf, sdf_gradients_time, geom_feat  = model.get_sdf_and_gradient(points_time, phase.iter_nr)
        sdf_gradients=sdf_gradients_time[:,0:3] #gradient is Nx4, remove the time part
        surface_sdf=sdf[:surface_points_time.shape[0]]
        surface_sdf_gradients=sdf_gradients[:surface_points_time.shape[0]]
        offsurface_sdf=sdf[-offsurface_points_time.shape[0]:]
        offsurface_sdf_gradients=sdf_gradients[-offsurface_points_time.shape[0]:]



        print("phase.iter_nr",  phase.iter_nr)
        sdf_loss_val=sdf_loss(surface_sdf, surface_sdf_gradients, offsurface_sdf, offsurface_sdf_gradients, surface_normals)
        loss=sdf_loss_val/30000 #reduce the loss so that the gradient doesn't become too large in the backward pass and it can still be represented with floating value
        print("loss", loss)


        cb.after_forward_pass(phase=phase, loss=loss.item(), lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction 

        #update gui
        if train_params.with_viewer():
            ngp_gui.m_c2f_progress=model.c2f.get_last_t()


        #backward
        optimizer.zero_grad()
        cb.before_backward_pass()
        loss.backward()


        cb.after_backward_pass()
        optimizer.step()


        #visualize the sdf by sphere tracing
        if phase.iter_nr%100==0 or phase.iter_nr==1 or ngp_gui.m_control_view:
            with torch.set_grad_enabled(False):
                # model.eval()


                vis_width=500
                vis_height=500
                if first_time_getting_control or ngp_gui.m_control_view:
                    first_time_getting_control=False
                    frame=Frame()
                    frame.from_camera(view.m_camera, vis_width, vis_height)
                    frustum_mesh=frame.create_frustum_mesh(0.1)
                    Scene.show(frustum_mesh,"frustum_mesh_vis")

                #forward all the pixels
                ray_origins, ray_dirs=create_rays_from_frame(frame, rand_indices=None) # ray origins and dirs as nr_pixels x 3
                

                #sphere trace those pixels
                ray_end, ray_end_sdf, ray_end_gradient, ray_end_t=sphere_trace(15, ray_origins, ray_dirs, model, return_gradients=True, sdf_multiplier=0.7, sdf_converged_tresh=0.005, time_val=ngp_gui.m_time_val)
                ray_end_converged, ray_end_gradient_converged, is_converged=filter_unconverged_points(ray_end, ray_end_sdf, ray_end_gradient) #leaves only the points that are converged
                ray_end_normal=F.normalize(ray_end_gradient, dim=1)
                ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                show_points(ray_end, "ray_end", color_per_vert=ray_end_normal_vis, normal_per_vert=ray_end_normal)
                ray_end_normal=F.normalize(ray_end_gradient_converged, dim=1)
                ray_end_normal_vis=(ray_end_normal+1.0)*0.5
                ray_end_normal_tex=ray_end_normal_vis.view(vis_width, vis_height, 3)
                ray_end_normal_img=tex2img(ray_end_normal_tex)
                Gui.show(tensor2mat(ray_end_normal_img), "ray_end_normal_img")


        # if phase.iter_nr%5000==0 or phase.iter_nr==1:
            # model.save(package_root, "4d_v2", phase.iter_nr)

        #finally just update the opengl viewer
        with torch.set_grad_enabled(False):
            if train_params.with_viewer():
                view.update()


def main():
    run()



if __name__ == "__main__":
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