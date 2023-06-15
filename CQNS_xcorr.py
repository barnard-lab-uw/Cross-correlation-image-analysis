import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from scipy import interpolate
import scipy as sp
from scipy import ndimage,signal

import rawpy
import imageio
import os
import png

from PIL import Image
from astropy.io import fits
from skimage.registration import phase_cross_correlation
from skimage.feature import  canny
import cv2
import time


def get_image_file_names(path_in):
    '''
    finds all image frames in a folder to be used for analysis
    '''
    fnames=os.listdir(path_in)
    fnames2=[]

    for _fname in fnames:
        if len(_fname.split('.txt'))==1:
            fnames2.append(_fname)
    fnames=fnames2.copy()

    return fnames
    
    
def load_reference_range(fnames,first_frame,last_frame,path_in='./'):
    '''
    Used to generate a reference image by averaging over a range of frames
    '''
    n_frames = 1+last_frame-first_frame
    
    _r = range(first_frame,last_frame+1)

    for i,fname in enumerate(fnames[first_frame:(last_frame+1)]):

    
        path = path_in+fname
        with Image.open(path) as _image:
            if i==0:
                ref_image=np.array(_image).astype(float)
            else:      
                _image=np.array(_image).astype(float)
                ref_image+=_image

        print(path_in,_r[i])
        display.clear_output(wait=True)

    ref_image/=n_frames
    return ref_image
    
def plot_ROI_box(ROI,color_in='b'):
    '''
    adds ROIs to matplotlib plots
    '''
    x1,y1,x2,y2 = ROI
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],color_in)

def image_ROI_edges(image_in,ROI_coords,pad,gauss_blur,threshold):
    x1,y1,x2,y2 = ROI_coords
    
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                       [-10+0j, 0+ 0j, +10 +0j],
                       [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    
    image_out = np.array(image_in).astype(float)[y1:y2,x1:x2]
    n,m = np.shape(image_out)
    image_out = sp.ndimage.gaussian_filter(image_out,gauss_blur)
    image_out = np.abs(sp.signal.convolve2d(image_out, scharr, boundary='symm',
                                            mode='same'))[pad:(n-pad),pad:(m-pad)]
    image_out[image_out< threshold]=threshold
    image_out-=threshold
    
    return image_out

def view_image_ROI_edges(image_in,ROI_coords,pad,gauss_blur,threshold):
    '''
    Function for viewing the output of image_ROI_edges to optimize parameters
    before carrying out cross correlation analysis
    '''
    dx = ROI_coords[2]-ROI_coords[0]
    dy = ROI_coords[3]-ROI_coords[1]
    
    fig=plt.figure(figsize=(dx/dy*10,10))

    _image = image_ROI_edges(image_in,ROI_coords,pad, gauss_blur,threshold)

    plt.pcolormesh(_image,cmap='cool')

    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()
    
def offsets_from_xcorr(fnames, ref_image, pad, ROI_coords, gauss_blur,threshold=0,path_in='./'):
    '''
    Function for determining offsets between images in a list to a reference image
    based on edge detection in a chosen ROI.
    '''
    offsets=[]
    ref_image_cross = image_ROI_edges(ref_image,ROI_coords,pad,gauss_blur,threshold)
    
    for i,fname in enumerate(fnames):

        path = path_in + fname
        with Image.open(path) as _image:
            _image=image_ROI_edges(_image,ROI_coords,pad,gauss_blur,threshold)

        offset,_,_ = phase_cross_correlation(_image,ref_image_cross,upsample_factor=300,normalization ='phase')
        offsets.append(offset)

        print(path_in,i)
        display.clear_output(wait=True)

    return np.array(offsets)


def offsets_from_xcorr_image_array(image_array, ref_image, pad, ROI_coords, gauss_blur,threshold=0):
    '''
    Function for determining offsets between images in a list to a reference image
    based on edge detection in a chosen ROI.
    '''
    offsets=[]
    ref_image_cross = image_ROI_edges(ref_image,ROI_coords,pad,gauss_blur,threshold)
    
    for i,_image in enumerate(image_array):

        _image=image_ROI_edges(_image,ROI_coords,pad,gauss_blur,threshold)            
        offset,_,_ = phase_cross_correlation(-_image,-ref_image_cross,upsample_factor=300,normalization ='phase')
        offsets.append(offset)

        print(i)
        display.clear_output(wait=True)
        
    return np.array(offsets)

def stack_shifted_reduced_images(fnames,offsets,red_N,path_in='./'):
    '''
    Applies x,y shifts from offsets and bins red_N frames in to produce lower noise out
    '''
    shifted_images=[]
    l = len(fnames)
    l_red = int(np.floor((l-1)/red_N))

    for i,fname in enumerate(fnames):

        if np.mod(i,red_N)==0:
            lumpped_images=[]
 
        path = path_in+fname
        with Image.open(path) as _image:
            _image=np.array(_image).astype(float)
        
        _image_2=sp.ndimage.shift(_image,-offsets[i],order=5) 
        lumpped_images.append(_image_2)
        
        if np.mod(i,red_N)==red_N-1:

            _data_array=np.array(lumpped_images)
            shifted_images.append(np.mean(_data_array,axis=0))

        print(i)    
        display.clear_output(wait=True)
    
    return shifted_images

def stack_reduced_images(fnames,red_N,path_in='./'):
    '''
    Bins red_N frames in to produce lower noise out
    '''   
    shifted_images=[]
    l = len(fnames)
    l_red = int(np.floor((l-1)/red_N))

    for i,fname in enumerate(fnames):

        if np.mod(i,red_N)==0:
            lumpped_images=[]
 
        path = path_in+fname
        with Image.open(path) as _image:
            _image=np.array(_image).astype(float)
        
        lumpped_images.append(_image)
        
        if np.mod(i,red_N)==red_N-1:

            _data_array=np.array(lumpped_images)
            shifted_images.append(np.mean(_data_array,axis=0))

        print(i)    
        display.clear_output(wait=True)
    
    return shifted_images

def plot_offsets_comparison(base_offsets,comparison_offsets=None):
    '''
    Plots dx and dy where base_offsets are offsets frame-by-frame while 
    comparison_offsets are potentially the result of binning frames
    '''
    fig=plt.figure(figsize=(15,5))
    
    plt.subplot(121)
    plt.plot(np.array(base_offsets).T[1,:])
    plt.xlabel('# frames',fontsize=14)
    plt.ylabel(r'$\Delta$ x',fontsize=14)
    plt.subplot(122)
    plt.plot(np.array(base_offsets).T[0,:])
    plt.xlabel('# frames',fontsize=14)
    plt.ylabel(r'$\Delta$ y',fontsize=14)
    
    if comparison_offsets is not None:
        if isinstance(base_offsets,tuple):
            n_base = len(base_offsets[0])
        else:
            n_base = len(base_offsets)
            
        if isinstance(comparison_offsets,tuple):
            n_comparison = len(comparison_offsets[0])
        else:
            n_comparison = len(comparison_offsets)
            
        
        x_post_shift = np.linspace(0,n_base,n_comparison)

        plt.subplot(121)
        plt.plot(x_post_shift,np.array(comparison_offsets).T[1,:])
        plt.subplot(122)
        plt.plot(x_post_shift,np.array(comparison_offsets).T[0,:])
        
    plt.show()
    
def generate_stacked_frames(fname, shifted_images,ref_image ,color_range, 
                            xpad=3,ypad=3,gauss_blur=1,ROI_out = None, path_out=None,cmap_out='bwr'):
    '''
    Generates series of frames where the output is plotted relative to a reference image.
    Also generates a "_base" series of frames for creating un-analyzed output frames.
    '''
    if path_out is not None:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
            
        fname = path_out+fname
    
    _n,_m = np.shape(ref_image)
    x1,x2 = xpad, _m-xpad
    y1,y2 = ypad, _n-ypad
    if ROI_out is not None:
        x1,y1,x2,y2 = ROI_out

    l=len(shifted_images)
    for i in range(l):

        fig=plt.figure(figsize=(15,10))

        _image_no_ref = shifted_images[i][y1:y2,x1:x2]
        _image = _image_no_ref/ref_image[y1:y2,x1:x2]

        _plot_data = sp.ndimage.gaussian_filter(_image-np.median(_image),gauss_blur)
        _plot_data_no_ref = sp.ndimage.gaussian_filter(_image_no_ref-np.median(_image_no_ref),gauss_blur)
        plt.pcolormesh(_plot_data,cmap='RdBu_r',vmin=-color_range,vmax=color_range)

        plt.imsave(fname+'_%04d'%i+'.png',_plot_data
                   ,cmap=cmap_out,vmin=-color_range,vmax=color_range )

        plt.imsave(fname+'_base_%04d'%i+'.png',_plot_data_no_ref
                   ,cmap='gray')

        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.show()
        display.clear_output(wait=True)

def shift_image_stack(stack_in,offsets):
    '''
    takes an image stack and applies (sub-pixel) x,y shifts
    '''
    n_stack = np.shape(stack_in)[0]
    
    stack_out = stack_in
    for i in range(n_stack):
        stack_out[i]=sp.ndimage.shift(stack_in[i],-offsets[i],order=5)
        print(i)
        display.clear_output(wait=True)
        
    return stack_out



def iterate_xcorr_shift(stack_in,ref_image,pad,ROI_coord,gauss_blur,threshold,N_iterate=0):
    '''
    iteratively calculates image registration offsets and applies shifts to converge
    to a well-aligned image stack. N_iterate seems to have to be greater than 2 or 3.
    '''
    stack_out = stack_in.copy()
    for i in range(N_iterate):
        _offsets = offsets_from_xcorr_image_array(stack_out,ref_image,pad,ROI_coord,gauss_blur,threshold)
        stack_out = shift_image_stack(stack_out,_offsets)        
    offsets = offsets_from_xcorr_image_array(stack_out,ref_image,pad,ROI_coord,gauss_blur,threshold)

    return stack_out, offsets
    
    
    

