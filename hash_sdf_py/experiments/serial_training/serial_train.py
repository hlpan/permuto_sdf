#!/usr/bin/env python3

#this scripts trains on a certain dataset but over all scenes
#useful for leaving it on a big machine training overnight or something
#CALL with ./hash_sdf_py/experiments/serial_training/serial_train.py --dataset dtu --with_mask --comp_name comp_1 --exp_info test

import torch
import argparse
import os

import easypbr
from easypbr  import *
from dataloaders import *

import hash_sdf
from hash_sdf  import TrainParams
from hash_sdf_py.utils.common_utils import create_dataloader
from hash_sdf_py.utils.hahsdf_utils import get_frames_cropped
from hash_sdf_py.train_hashsdf import train
from hash_sdf_py.train_hashsdf import HyperParamsHashSDF
import hash_sdf_py.paths.list_of_training_scenes as list_scenes



torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


parser = argparse.ArgumentParser(description='Train sdf and color')
parser.add_argument('--dataset', required=True, help='Dataset name from easypbr, dtu, bmvs, multiface')
parser.add_argument('--comp_name', required=True,  help='Tells which computer are we using which influences the paths for finding the data')
parser.add_argument('--low_res', action='store_true', help="Use_low res images for training for when you have little GPU memory")
parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
args = parser.parse_args()




def run():

    config_file="train_hashsdf.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)
    train_params=TrainParams.create(config_path)
    hyperparams=HyperParamsHashSDF()


    #get the checkpoints path which will be at the root of the hash_sdf package 
    hash_sdf_root=os.path.dirname(os.path.abspath(hash_sdf.__file__))
    checkpoint_path=os.path.join(hash_sdf_root, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    
    with_viewer=False
    train_params.set_with_tensorboard(True)
    train_params.set_save_checkpoint(True)
    dataset_name=args.dataset


    print("args.with_mask", args.with_mask)
    print("args.low_res", args.low_res)
    print("checkpoint_path",checkpoint_path)
    print("with_viewer", with_viewer)




    #get all the scene from a dataset
    scenes_list=list_scenes.datasets[dataset_name]

    #iterate though every scene, and create a dataloader for it
    for scene_name in scenes_list:
        print("READING scene_name", scene_name)

        training_config="with_mask_"+str(args.with_mask) 
        experiment_name="full_"+dataset_name+"_"+scene_name+"_"+training_config
        if args.exp_info:
            experiment_name+="_"+args.exp_info
        print("experiment name",experiment_name)


        loader_train, loader_test= create_dataloader(config_path, args.dataset, scene_name, args.low_res, args.comp_name, args.with_mask)
        

        #tensoreel
        if isinstance(loader_train, DataLoaderPhenorobCP1):
            aabb = create_bb_for_dataset(args.dataset)
            tensor_reel=MiscDataFuncs.frames2tensors( get_frames_cropped(loader_train, aabb) ) #make an tensorreel and get rays from all the images at
        else:
            tensor_reel=MiscDataFuncs.frames2tensors(loader_train.get_all_frames()) #make an tensorreel and get rays from all the images at 



        #start training 
        train(args, config_path, hyperparams, train_params, loader_train, experiment_name, with_viewer, checkpoint_path, tensor_reel)



        # if use_home:
        #     if model_type=="hashsdf":
        #         checkpoint_path="/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/batch_training_v7_NoLipshitzJustWD/"+dataset_name+"/"+training_config
        #     elif model_type=="ingp":
        #         checkpoint_path="/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/ingp/"+dataset_name+"/"+training_config
        #     elif model_type=="neus":
        #         checkpoint_path="/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/neus/"+dataset_name+"/"+training_config
                
        # else:
        #     if model_type=="hashsdf":
        #         checkpoint_path="/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/batch_training_v7_NoLipshitzJustWD/"+dataset_name+"/"+training_config
        #     elif model_type=="ingp":
        #         checkpoint_path="/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/ingp/"+dataset_name+"/"+training_config
        #     elif model_type=="neus":
        #         checkpoint_path="/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/neus/"+dataset_name+"/"+training_config



        # #start training
        # print("start training")
        # if model_type=="hashsdf":
        #     train(args, config_path, loader_train, frames_train, experiment_name, dataset_name,  with_viewer, with_tensorboard, save_checkpoint, checkpoint_path, tensor_reel, 1e-3, 50000 ,[100000,150000,180000,190000], 200000)
        # elif model_type=="ingp":
        #     train_ingp(args, config_path, loader_train, experiment_name, dataset_name,  with_viewer, with_tensorboard, save_checkpoint, checkpoint_path, tensor_reel, 1e-2, [60000,70000,80000,90000], 100000)
        # elif model_type=="neus":
        #     # train_neus(args, config_path, loader_train, experiment_name, dataset_name,  with_viewer, with_tensorboard, save_checkpoint, checkpoint_path, tensor_reel, 1e-2, [60000,70000,80000,90000], 100000)
        #     train_neus(args, config_path, loader_train, experiment_name, dataset_name,  with_viewer, with_tensorboard, save_checkpoint, checkpoint_path, tensor_reel, 1e-2, 300000) #the lr doesnt actually matter here
        # else:
        #     print("Now known model type", model_type)
        #     exit(1)















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
