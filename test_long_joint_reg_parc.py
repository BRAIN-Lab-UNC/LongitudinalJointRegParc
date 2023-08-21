#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  19 17:13:04 2022

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""
import argparse
import numpy as np
import glob
import torch
import time
import os
import math

from s3pipe.models.models import LongJointRegAndParc
from s3pipe.utils.interp_torch import convert2DTo3D, diffeomorp_torch, bilinearResampleSphereSurf_torch, resampleSphereSurf_torch
from s3pipe.surface.atlas import LongitudinalRegAndParcSpheres
from s3pipe.surface.s3reg import readRegConfig, createRegConfig
from s3pipe.surface.parc import compute_dice
from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.utils.utils import get_par_fs_lookup_table

 
lookup_table_vec, lookup_table_scalar, lookup_table_name = get_par_fs_lookup_table()
  
abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Perform longitudinally joint registration and parcellation of cortical surfaces, '+\
                                     'leading to longituidnally more consistent results., ' +\
                                     'Longitudinal cortical surfaces with sulc and curv features spherically mapped and resampled with 163,842 vertices in vtk are required.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--files_pattern', default=None, help="pattern of the filename of one subject's longitudinal surfaces, "+\
                                                              "e.g. '/path/to/subject-0001/ses-*/*.SphereSurf.Resp163842.vtk'. Note the pattern needs to be quoted for python otherwise it will be parsed by the shell by default.")
    parser.add_argument('--hemi', help="hemisphere, lh or rh", choices=['lh', 'rh'], required=True)
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')
    parser.add_argument('--model_path', default=None, help="the folder containing pre-trained models, if not given, it will be ./pretrained_models ")
    args = parser.parse_args()
    resp_inner_surf = args.resp_inner_surf
    files_pattern = args.files_pattern
    hemi = args.hemi
    device = args.device
    model_path = args.model_path
    print('\n------------------------------------------------------------------')
    print('Longitudinal joint registration and parcellation of cortical surfaces...')
    print('files_pattern:', files_pattern)
    print('hemi:', hemi)
    if model_path is None:
        model_path = abspath + '/pretrained_models'
    print('model_path:', model_path)
    
    # check device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    
    files = sorted(glob.glob(files_pattern))
    print('files:', files)
    print('Number of longitudinal scans:', len(files))
        
        
    ########################################################################
    config = readRegConfig(abspath + '/regConfig_163842_sucu.yaml')
    config['device'] = device
    config['atlas_file'] = abspath + '/fs_atlas/72_lh.SphereSurf.vtk'
    config = createRegConfig(config)
     
    batch_size = 1
    NUM_ROIS = 36 # although 36, but OASIS only has 35 ROIs, No. 5 corpus callosum is combined with unlabeled subcortical region in OASIS
    num_long_scans = 5
    complex_chs = 16
    
    model_name = 'OASIS_long_joint_reg_parc'  # joint_wSharedEnc_wParcSim
     
    test_files = files
    val_dataset = LongitudinalRegAndParcSpheres(test_files, n_vertex=config['n_vertexs'][-1], 
                                                num_long_scans=num_long_scans)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, pin_memory=True)
    
    
    model = LongJointRegAndParc(in_ch=2, out_parc_ch=NUM_ROIS, level=config['levels'][-1], 
                                n_res=5, rotated=0, complex_chs=complex_chs, num_long_scans=num_long_scans)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.to(device)
    model.load_state_dict(torch.load(model_path+'/'+ model_name +'.mdl'))
    model.eval()
    
    ###############################################################################
    """  load predefined parameter from config   """
    
    fixed_sucu_0 = torch.cat((config['atlas']['sulc'].unsqueeze(1), config['atlas']['curv'].unsqueeze(1)), dim=1)
    fixed_sucu_0 = fixed_sucu_0.unsqueeze(0).permute(0, 2, 1)
    fixed_lbl_target = config['atlas']['par_fs_0_35']
    fixed_lbl_mask = torch.zeros((fixed_lbl_target.shape[0], NUM_ROIS), dtype=torch.float32, device=device)
    tmp0 = np.asarray(list(range(len(fixed_lbl_target))))
    tmp1 = fixed_lbl_target.detach().cpu().numpy().astype(np.int32)
    fixed_lbl_mask[tmp0, tmp1] = 1.0
    neigh_orders = config['neigh_orders'][-1]
    n_vertex = config['n_vertexs'][-1]
    bi_inter = config['bi_inter_0s'][-1]
    En_0 = config['Ens'][-1][0]
    fixed_xyz_0 = config['fixed_xyz_0'][0:n_vertex, :]
    
    
     
    # dataiter = iter(train_dataloader)
    # moving_sucu_0, moving_lbl_target, running_files = next(dataiter)


    deform_max = 0
    for batch_idx, (moving_sucu_0, moving_lbl_target, running_files) in enumerate(val_dataloader):
        print('----------------------------------------------------------------------------')
        print("Longitudinal consistent registration and parcellation for:",
              running_files[0][0].split('/')[-2])
        
        # if os.path.exists(running_files[-1][0].replace('sphere.Resp163842.RigidAlignToAtlas.Resp163842.AlignToUNCAtlas.Resp163842.npy', 
        #                                                'sphere.Resp163842.RigidAlignToAtlas.Resp163842.AlignToUNCAtlas.Resp163842.'+ model_name +'.moved.vtk')):
        #     continue
        
        
        model.eval()
        
        moving_sucu_0 = moving_sucu_0.to(device)
        moving_sucu_0 = moving_sucu_0.permute(1, 0, 3, 2)  # input shape should be num_long_scans * 1 * input_channel * num_vertices 
        moving_lbl_target = moving_lbl_target.squeeze(0).squeeze(-1).to(device) # label shape should be num_long_scans * num_vertices 
        
        with torch.no_grad():
            moving_lbl_pred, fixed_lbl_pred, displacement_2d = model(moving_sucu_0, fixed_sucu_0)
            # displacement_2d = model(moving_sucu_0, fixed_sucu_0)
        for i in range(num_long_scans):
            moving_lbl_pred[i] = moving_lbl_pred[i].squeeze().permute(1,0)
        displacement_2d = displacement_2d.squeeze().permute(1,0)
        velocity_2d = displacement_2d/30.0    # pytorch scale is too large for deformation
        
        fixed_lbl_pred = fixed_lbl_pred.squeeze().permute(1,0).max(1)[1]
        print("fixed_lbl_pred dice:", compute_dice(fixed_lbl_pred, fixed_lbl_target))
        
        n_running_files = len(running_files)
        for i in range(num_long_scans):   
            print(i,'-th scan:')
            # obtain parcellation result
            moving_lbl_pred_i = moving_lbl_pred[i].max(1)[1]
            if i < n_running_files:
                dice_i = compute_dice(moving_lbl_pred_i, moving_lbl_target[i])
                print("moving_lbl_pred dice:", dice_i)
            
            # obtain reg result
            displacement_3d = convert2DTo3D(velocity_2d[:, i*2:i*2+2], En_0, device)
            # print(torch.norm(displacement_3d,dim=1).max().item())
            velocity_3d = displacement_3d/math.pow(2, 6)
            total_deform = diffeomorp_torch(fixed_xyz_0, velocity_3d, 
                                            num_composition=6, bi=True, 
                                            bi_inter=bi_inter, 
                                            device=device)
                       
            # check longitudinal repeated scan's reproducibility
            if i >= n_running_files-1:
                if i == n_running_files-1:
                    prev_deform = total_deform
                else:
                    deform_max_tmp = torch.max(torch.abs(total_deform*100 - prev_deform*100))
                    if deform_max_tmp.item() > deform_max:
                        deform_max = deform_max_tmp.item()
                    prev_deform = total_deform
                    assert (torch.abs(moving_lbl_pred[i] - moving_lbl_pred[i-1]) ==0).sum() == 163842*36, 'parcellation not consistent'

            # save resutls
            if i < n_running_files:
                orig_surf = read_vtk(running_files[i][0].replace('.SphereSurf.RigidAlignToAtlas.Resp163842.npy', 
                                                                     '.SphereSurf.RigidAlignToAtlas.Resp163842.vtk'))
                surf = {'vertices': config['atlas']['vertices'],
                        'faces': config['atlas']['faces'],
                        'sulc': orig_surf['sulc'][0:n_vertex],
                        'curv': orig_surf['curv'][0:n_vertex],
                        'deform': displacement_3d.cpu().numpy() * 100.0,
                        'par_fs_vec_long_joint_reg_parc': lookup_table_vec[moving_lbl_pred_i.cpu().numpy()]}
                        # 'par_fs_vec': orig_surf['par_fs_vec'][0:n_vertex,:],
                        
                
                write_vtk(surf, running_files[i][0].replace('.SphereSurf.RigidAlignToAtlas.Resp163842.npy', 
                                                                '.SphereSurf.RigidAlignToAtlas.Resp163842.'+ model_name +'.deform.vtk'))
                    
                moved_surf = {'vertices': total_deform.detach().cpu().numpy()*100.,
                              'faces': config['atlas']['faces'],
                              'sulc': orig_surf['sulc'][0:n_vertex],
                              'curv': orig_surf['curv'][0:n_vertex],
                              'par_fs_vec_long_joint_reg_parc': lookup_table_vec[moving_lbl_pred_i.cpu().numpy()]}
                write_vtk(moved_surf, running_files[i][0].replace('.SphereSurf.RigidAlignToAtlas.Resp163842.npy', 
                                                                      '.SphereSurf.RigidAlignToAtlas.Resp163842.' + model_name +'.moved.vtk'))
                print("save total deform done for", running_files[i][0])
            
    print('deform_max:', deform_max)
    print('done!')

                 