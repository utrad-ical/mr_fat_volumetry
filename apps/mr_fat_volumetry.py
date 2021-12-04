"""
Created on Sat Feb 13 15:00:00 2021
@author: ynomura
"""
from __future__ import print_function

import argparse
import json
import numpy as np

from operator import itemgetter
from PIL import Image
from scipy import ndimage
from skimage import filters

import torch
import torch.utils.data

import measure_boundary_length
import readmhd
import thigh_slice_selection as slice_selection

from fc_resnet import FC_RESNET


def _load_volume_data(in_path):
    
    for n in range(4):
        
        # Load volume data (0:in, 1:oop, 2:fat, 3:water)
        in_file_name = "{}/{}.mhd".format(args.in_path, n)

        volume = readmhd.read(in_file_name)

        if n == 0:
            org_volume = volume.vol[None]
            voxel_size_mm = np.array(volume.voxelsize) # [x,y,z]
            
            # get ogirinal slice number from 0.txt
            dcm_dump_file_name = in_file_name = "{}/{}.json".format(args.in_path, n)
            
            with open(dcm_dump_file_name) as fp:
                dcm_dump_data = json.load(fp)
            
            org_slice_num_list = []
            
            for n in range(org_volume.shape[1]):
                org_slice_num_list.append(dcm_dump_data["unique"][n]["0020,0013"])
            
        else:
            org_volume = np.concatenate((org_volume, volume.vol[None]), axis=0)

    return org_volume, voxel_size_mm, org_slice_num_list
    

def _preprocessing(org_volume, voxel_size_mm, out_path):

    for n in range(4):
        
        tmp_volume = np.copy(org_volume[n])
        
        # Decide slice range (220 slices, using in-phgase)
        if n == 0:
            lowest_slice_num = slice_selection.main(tmp_volume)
            highest_slice_num = lowest_slice_num - 220
            print("highest_slice:%d, lowest_slice:%d"\
                  % (highest_slice_num+1, lowest_slice_num+1))
            org_slice_range = list(range(highest_slice_num + 1, lowest_slice_num + 1,1))         
                       
        # crop volume 
        tmp_volume = tmp_volume[:lowest_slice_num]
        tmp_volume = tmp_volume[-1 * 220:]
        
        # normarization
        tmp_volume = tmp_volume.astype(np.float32)
        tmp_volume /= tmp_volume.std()

        # extract fat volume (original size, normalized)
        if n == 2:
            fat_volume = np.copy(tmp_volume)
                
        # zoom
        tmp_volume = ndimage.zoom(tmp_volume, (0.5, 0.5, 0.5), order=1)

        if n == 0:
            dl_input_volume = tmp_volume[None]
        else:
            dl_input_volume = np.concatenate((dl_input_volume, tmp_volume[None]), axis=0)    
    
    return dl_input_volume, fat_volume, org_slice_range


def _extract_body_trunk_and_cavity(in_volume, model_file_name, parameter_n, gpu_id=0):
    
    device = f'cuda:{gpu_id}' 
    
    # Load model
    model = FC_RESNET(4, 2, parameter_n)
    model.load_state_dict(torch.load(model_file_name))
    model = model.to(device)
    model.eval()
         
    #extraction of body trunk & abdomibnal cavity
    with torch.no_grad():
        in_volume = torch.from_numpy(in_volume[np.newaxis,:,:,:,:]).to(device)
        output = model(in_volume).cpu().numpy()

        # abdominal cavity
        cavity = output[0,0,:,:,:]
        cavity = ndimage.zoom(cavity, 2.0, order=1)
        cavity = (cavity >= 0.5).astype(np.uint8)
            
        # body trunk
        body_trunk = output[0,1,:,:,:]
        body_trunk = ndimage.zoom(body_trunk, 2.0, order=1)
        body_trunk = (body_trunk >= 0.5).astype(np.uint8)
    
    return body_trunk, cavity
    
    
def _export_png_images(out_path, slice_num, org_img, vat_mask, sat_mask,
                       boundary_img, window_level=350, window_width=750):
    
    # adjustment of contrast using WL/WW
    lowest_value = window_level - (window_width // 2)
    convert_ratio = 255.0 / float(window_width)
    org_img = np.floor((org_img.astype(np.float32) - float(lowest_value)) * convert_ratio + 0.5)
    org_img[org_img < 0] = 0
    org_img[org_img > 255] = 255
    gray_img = org_img.astype(np.uint8)

    # Save original image
    org_file_name = f"{out_path}/mr{slice_num:03d}.png"
    Image.fromarray(gray_img).save(org_file_name)
    
    # Save overlay image
    alpha = 0.9
    vat_img = np.floor(alpha * 255 * vat_mask.astype(np.float32)\
                       + (1 - alpha) * org_img + 0.5).astype(np.uint8)
    vat_img = np.where(vat_mask == 1, vat_img, gray_img)
    vat_img[boundary_img == 1] = 0
    sat_img = np.floor(alpha * 255 * sat_mask.astype(np.float32)\
                       + (1 - alpha) * org_img + 0.5).astype(np.uint8)
    sat_img = np.where(sat_mask == 1, sat_img, gray_img)
    sat_img[boundary_img == 1] = 0

    gray_img[boundary_img == 1] = 255
        
    color_file_name = f"{out_path}/result{slice_num:03d}.png"
    color_img = np.array([vat_img, gray_img, sat_img], dtype=np.uint8)
    Image.fromarray(color_img.transpose(1,2,0)).save(color_file_name)


def _create_result_data(in_phase_volume, body_trunk, vat_data, sat_data, voxel_size_mm,
                        out_path, org_slice_range,
                        org_slice_num_list, fat_volume_range):
    
    voxel_size_cm = voxel_size_mm * 0.1
    json_results = {}
    
    # volume results
    body_volume = np.sum(body_trunk) * voxel_size_cm.prod()
    vat_volume = np.sum(vat_data) * voxel_size_cm.prod()
    sat_volume = np.sum(sat_data) * voxel_size_cm.prod() 
    json_results["results"] = { "bodyTrunkVolume": body_volume,
                                "vatVolume": vat_volume,
                                "satVolume": sat_volume,
                                "volRatio": vat_volume / (sat_volume + 0.00001)} 
        
    # slice results
    slice_results = []
    #for original_slice_num, slice_num in zip(org_slice_range, range(vat_data.shape[0])):
    for n in range(fat_volume_range[0]-1, fat_volume_range[1]+2):
        
        idx = n - (fat_volume_range[0] - 1)

        body_area = np.sum(body_trunk[n,:,:]) * voxel_size_cm[0:2].prod()
        vat_area = np.sum(vat_data[n,:,:]) * voxel_size_cm[0:2].prod()
        sat_area = np.sum(sat_data[n,:,:]) * voxel_size_cm[0:2].prod()
        
        boundary_length, boundary_img = measure_boundary_length.main(body_trunk[n], voxel_size_mm)

        slice_items = { "rank": int(idx + 1),
                        "sliceNum": org_slice_num_list[n],
                        "volSliceIdx": org_slice_range[n],
                        "areaBody": body_area,
                        "areaVAT": vat_area,
                        "areaSAT": sat_area,
                        "areaRatio": vat_area / (sat_area + 0.00001),
                        "bodyContourLength": boundary_length * 0.1 }
        slice_results.append(slice_items)

        _export_png_images(out_path,
                           idx,
                           in_phase_volume[org_slice_range[n],:,:],
                           vat_data[n,:,:],
                           sat_data[n,:,:],
                           boundary_img)
         
    json_results["sliceResults"] = slice_results
     
    json_file_name = f"{out_path}/results.json"
    with open(json_file_name, mode="w") as fp:
        fp.write(json.dumps(json_results))
        
         
def mr_fat_volumetry(in_path, out_path, model_file_name, parameter_n, gpu_id):    

    org_volume, voxel_size_mm, org_slice_num_list = _load_volume_data(in_path)

    dl_input_volume, fat_volume, org_slice_range = _preprocessing(org_volume,
                                                                  voxel_size_mm,
                                                                  out_path)
    
    cropped_slice_num_list = itemgetter(*org_slice_range)(org_slice_num_list)
   
    body_trunk, cavity = _extract_body_trunk_and_cavity(dl_input_volume,
                                                        model_file_name,
                                                        parameter_n,
                                                        gpu_id)
    
    # extraction of VAT and SAT
    fat_threshold = filters.threshold_otsu(fat_volume)
    fat_label = (fat_volume >= fat_threshold).astype(np.uint8)
   
    vat_data = np.zeros(fat_label.shape, dtype=np.uint8)
    sat_data = np.zeros(fat_label.shape, dtype=np.uint8)

    cavity_pos = np.where(cavity != 0)
    fat_volume_range = [np.min(cavity_pos[0]), np.max(cavity_pos[0])]
    print(fat_volume_range)
            
    vat_data = fat_label * body_trunk * cavity
 
    sat_data = fat_label * body_trunk * np.abs(1 - cavity)
    sat_data[:fat_volume_range[0],:,:] = 0
    sat_data[fat_volume_range[1]+1:,:,:] = 0
    
    body_trunk[:fat_volume_range[0],:,:] = 0
    body_trunk[fat_volume_range[1]+1:,:,:] = 0
    body_pos = np.where(body_trunk != 0)
        
    crop_range = [ max(0, np.min(body_pos[1] - 10)),\
                   min(body_trunk.shape[1], np.max(body_pos[1] + 11)),\
                   max(0, np.min(body_pos[2] - 10)),\
                   min(body_trunk.shape[2], np.max(body_pos[2] + 11))]
                                
    in_phase_volume = org_volume[0,:,crop_range[0]:crop_range[1],crop_range[2]:crop_range[3]]                        
    body_trunk = body_trunk[:,crop_range[0]:crop_range[1],crop_range[2]:crop_range[3]]                        
    vat_data = vat_data[:,crop_range[0]:crop_range[1],crop_range[2]:crop_range[3]]                        
    sat_data = sat_data[:,crop_range[0]:crop_range[1],crop_range[2]:crop_range[3]]                        

    _create_result_data(in_phase_volume, body_trunk, vat_data, sat_data, voxel_size_mm,
                        out_path, org_slice_range, cropped_slice_num_list, fat_volume_range)
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(    
                description='Fat volumetory in whole body dixon MR images (for CIRCUS plug-in)',
                add_help=True)
    parser.add_argument('-i', '--in_path',\
                        default="C:/Users/ynomura/Desktop/dixon/circus")
    parser.add_argument('-o', '--out_path',\
                        help='Output path name',
                        default="C:/Users/ynomura/Desktop/dixon/circus")
    parser.add_argument('-m', '--model_file_name',\
                        help='File name of trained model (.pth)',
                        default="model_random_search_best.pth")
    parser.add_argument('-n', '--parameter_n',\
                        help='Parameter n of FC-ResNet',
                        type=int, default=2)
    parser.add_argument('-g', '--gpu_id', type=str, default='0')    
    
    args = parser.parse_args()
        
    #model test & fat caluculation
    mr_fat_volumetry(args.in_path,
                     args.out_path, 
                     args.model_file_name,
                     args.parameter_n,
                     args.gpu_id)
        