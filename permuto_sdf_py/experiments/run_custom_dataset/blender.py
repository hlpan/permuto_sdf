#
import os
import json
import math
import numpy as np
from PIL import Image

import torch

class BlenderDatasetBase():
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split


    def load(self):
        # self.has_mask = True
        # self.apply_mask = True
        with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        #self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)
            if self.split in ['train', 'val']:
                img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}")
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                if self.has_mask:
                    mask_paths = [os.path.join(mask_dir, d.name), os.path.join(mask_dir, d.name[3:])]
                    mask_paths = list(filter(os.path.exists, mask_paths))
                    assert len(mask_paths) == 1
                    mask = Image.open(mask_paths[0]).convert('L') # (H, W, 1)
                    mask = mask.resize(self.img_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask)[0]
                else:
                    mask = torch.ones_like(img[...,0], device=img.device)
                self.all_fg_masks.append(mask) # (h, w)
                self.all_images.append(img)

        self.all_c2w = torch.stack(self.all_c2w, dim=0)   
        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()

        sphere_center = torch.from_numpy(np.array(meta['sphere_center']))
        self.all_c2w = normalize_poses(self.all_c2w, sphere_center) 
        # self.all_c2w, self.all_images, self.all_fg_masks = \
        #     torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
        #     torch.stack(self.all_images, dim=0).float().to(self.rank), \
        #     torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)
        self.all_c2w = self.all_c2w.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)