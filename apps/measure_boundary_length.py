"""
Created on Sat Feb 13 21:19:52 2021
@author: ynomura
"""

import numpy as np
from scipy import ndimage

import readmhd as readmhd

def _set_array_dist_to_xy_arounds(arounds, voxel_size_mm):
    
    distance = np.zeros(len(arounds))

    for i in range(len(arounds)):
        dx = voxel_size_mm[0] * arounds[i][0]
        dy = voxel_size_mm[1] * arounds[i][1]
        distance[i] = np.sqrt(dx * dx + dy * dy)

    return distance


def main(body_trunk, voxel_size_mm):
    
    around_num = 8
    arounds = [ [-1, -1], [0, -1], [1, -1], [1, 0], 
                 [1, 1], [0, 1], [-1, 1], [-1, 0]]
    back_idx = [4, 5, 6, 7, 0, 1, 2, 3]
    distance_xy = _set_array_dist_to_xy_arounds(arounds, voxel_size_mm)
    
    structure = np.ones([3,3])

    tmp_img = np.copy(body_trunk)

    # preprocessing
    mask_img = ndimage.binary_dilation(body_trunk, structure)
    
    # cavity filling
    tmp_img = np.abs(1 - mask_img.astype(np.int8)).astype(np.int8)
    labeled_img, label_num = ndimage.label(tmp_img)
    sizes = ndimage.measurements.sum(labeled_img, tmp_img, range(1, label_num + 1))
    mask_img = (1 - (labeled_img == (np.argmax(sizes) + 1)).astype(np.uint8))
    
    # erosinon (2 times)
    mask_img = ndimage.binary_erosion(mask_img, structure)
    mask_img = ndimage.binary_erosion(mask_img, structure)
    
    # dilation
    mask_img = ndimage.binary_dilation(mask_img, structure).astype(np.uint8)

    # measurement 
    length = 0.0
    
    x_start = -1
    y_start = -1
    contour_num = 0

    for j in range(mask_img.shape[0]):
        for i in range(mask_img.shape[1]):
           b = 0
           if mask_img[j][i] > 0:
                if (mask_img[j][i - 1] == 0 or mask_img[j][i + 1] == 0\
                    or mask_img[j - 1][i] == 0 or mask_img[j + 1][i] == 0):
                    b = 1  

                if b > 0:
                    mask_img[j][i] = 2
                    contour_num += 1
                    if x_start < 0:
                        #print(j,i)
                        x_start = i
                        y_start = j
                else:        
                   mask_img[j][i] = 1
    
    x_current = x_start
    y_current = y_start
    prev_idx = 0

    for n in range(around_num):
        xa = x_current + arounds[n][0]
        ya = y_current + arounds[n][1]
        
        if mask_img[ya][xa] > 1:
            length += distance_xy[n]
            x_current = xa
            y_current = ya
            prev_idx = n
            mask_img[y_current][x_current] = 3
            break

    track_cnt = 0
    track_num = contour_num * 4 // 3

    for n in range(track_num):
        
        track_cnt = n
        
        idx_start_search = back_idx[prev_idx]
        
        for idx_add in range(1, around_num):
            idx_around = idx_start_search + idx_add

            if idx_around >= around_num:
                idx_around = idx_around - around_num
                
            xa = x_current + arounds[idx_around][0]
            ya = y_current + arounds[idx_around][1]

            if mask_img[ya][xa] > 1:
                length += distance_xy[idx_around]
                x_current = xa
                y_current = ya
                prev_idx = idx_around
                mask_img[y_current][x_current] = 3
                break

        if idx_add == around_num:
            length = 0.0
            mask_img[:,:] = 0
            break

        if x_current == x_start and y_current == y_start:
            break

    if track_cnt == track_num:
        length = 0.0
        mask_img[:,:] = 0

    mask_img = (mask_img == 3).astype(np.uint8)
    
    return length, mask_img

    
if __name__ == '__main__':


   in_file_name = "body_trunk.mhd"

   volume = readmhd.read(in_file_name)
   voxel_size_mm = np.array(volume.voxelsize) # [x,y,z]
     
   img = volume.vol[88]
   
   length, mask_img = main(img, voxel_size_mm)
   
   print(length)