from typing import Tuple, Optional, Union, List
from typing_extensions import Literal
import json
import sys
sys.path.insert(1, '/Users/bangun/pyptychostem')
from libertem.common import Shape 
from scipy.ndimage import center_of_mass 
import numpy as np
from STEM4D import Data4D 

def cut_det(idp:np.ndarray,
            com:Tuple,
            det_size:Tuple):
    """
    A function to cut the detector size measured from the center of mass

    Parameters
    ----------
    idp
        Diffraction patterns can be four dimensional or only two dimensional (without scanning)
    com
        Center of mass
    
    det_size
        Pre-defined size of detector's cut

    Returns
    -------
    idp
        Resize detector size of diffraction patterns
    """
    cy = round(com[0])
    cx = round(com[1])

    top = cy - det_size[0] // 2
    bottom = cy + det_size[0] // 2
    if top < 0:
        bottom += abs(top)

    left = cx - det_size[1] // 2
    right = cx + det_size[1] // 2
    if left < 0:
        right += abs(left)

    if len(idp.shape) > 2:

        idp = idp[:, :, max(top, 0): bottom,
                  max(left, 0): right]
    else:
        idp = idp[max(top, 0): bottom,
                  max(left, 0): right]
    return idp

def padding_detector(idp, ori_crop):
    """
    A function to add zero padding to make detector's dimension larger

    Parameters
    ----------
    idp
        Pre-defined memorymap data so we can store efficiently
    ori_crop
        Original data after croping the detector
    
    Returns
    -------
    idp
        Resize detector size of diffraction patterns
    """

    pad_dimension = (np.array(idp.shape[2::])-np.array(ori_crop.shape[2::]))//2
       
    for row in range(idp.shape[0]):
        for col in range(idp.shape[1]):
            idp[row,col] = np.pad(ori_crop[row,col], tuple(pad_dimension), constant_values=0)[:]
            
    return idp

def replicate_scan(ori_crop:np.ndarray, 
                   ds_shape:Shape):
    """
    A function to add scanning points by replicate the data

    Parameters
    ----------
    ori_crop
        Original data after croping the detector
    ds_shape
        Tuple contains dimensions of datasets
    
    Returns
    -------
    idp
        Resize scanning size of diffraction patterns
    """
    idp = np.zeros((ds_shape.nav[0], ds_shape.nav[1], ori_crop.shape[2], ori_crop.shape[3]))
    #scan_dimension = (np.array(ds_shape.nav)/np.array(ori_crop.shape[0:2]))
    step_row = np.arange(0,ds_shape.nav[0],ori_crop.shape[0])
    step_col = np.arange(0,ds_shape.nav[1],ori_crop.shape[1])

    for row in range(len(step_row)):
        for col in range(len(step_col)):
            idp[step_row[row]:step_row[row] + ori_crop.shape[0],
                step_col[col]:step_col[col] + ori_crop.shape[1]] = ori_crop[:]
    return idp

 

def increase_detector(dim:Tuple, 
                      parfile:str):
    
    """
    A function to increase the detector dimension directly from pyptychostem data

    Parameters
    ----------
    parfile
        Path for our datasets
    dim
        Tuple contains dimensions of datasets
    
    Returns
    -------
    path_data
        Path to the new datasets
    path_json
        Path to the json file containing parameters
    """
     
    # Create Dictionary
    par_dictionary = {}
    file = open(parfile)

    for line in file:
        if line.startswith('##'):
            continue
        split_line = line.rstrip().split('\t')


        if len(split_line)!=2:
            continue
        key, value = split_line
        par_dictionary[key] = value

    # Load Data
    data_4D = Data4D(parfile)
    data_4D.estimate_aperture_size()

    # Adapting dimension
    ds_shape = Shape(dim, sig_dims=2)

    ori = data_4D.data_4D[:,0:64,:,:]
    pacbed = np.sum(ori,(0,1))

    com = center_of_mass(pacbed)

    ori_crop = cut_det(ori,com,ds_shape.sig)
    print('Original crop ', ori_crop.shape)
    path_data = "/Local/erc-1/bangun/LiveWDD_Data/idp.npy"

    idp = np.lib.format.open_memmap(path_data, dtype='float32', mode='w+', shape=dim)
    print('Allocation ', idp.shape)
    # Increasing detector 
    idp = padding_detector(idp, replicate_scan(ori_crop, ds_shape))

    # Calculate semiconv_pix and com
    data_4D.data_4D = idp
    data_4D.estimate_aperture_size()

    # Create json
    par_dictionary['com'] = (data_4D.center_y,data_4D.center_x)
    par_dictionary['rad'] = data_4D.aperture_radius
    par_dictionary['complex_dtype'] = 'complex64'
    par_dictionary['order'] = 16
    par_dictionary['dim'] = dim

    path_json = "/Local/erc-1/bangun/LiveWDD_Data/params.json"
    
    with open(path_json, 'w') as f:
         json.dump(par_dictionary, f)

    return path_data, path_json
    
def increase_scan(dim, parfile):


    """
    A function to increase the scanning points dimension directly from pyptychostem data

    Parameters
    ----------
    parfile
        Path for our datasets
    dim
        Tuple contains dimensions of datasets
    
    Returns
    -------
    path_data
        Path to the new datasets
    path_json
        Path to the json file containing parameters
    """
        
    # Create Dictionary

    par_dictionary = {}
    file = open(parfile)

    for line in file:
        if line.startswith('##'):
            continue
        split_line = line.rstrip().split('\t')


        if len(split_line)!=2:
            continue
        key, value = split_line
        par_dictionary[key] = value

    # Load Data
    data_4D = Data4D(parfile)
    data_4D.estimate_aperture_size()

 
    # Adapting dimension 
    ds_shape = Shape(dim, sig_dims=2)

    ori = data_4D.data_4D[:,0:64,:,:]
    pacbed = np.sum(ori,(0,1))

    com = center_of_mass(pacbed)

    ori_crop = cut_det(ori,com,ds_shape.sig)

    path_data = "/Local/erc-1/bangun/LiveWDD_Data/idp.npy"

    idp = np.lib.format.open_memmap(path_data, dtype='float32', mode='w+', shape=dim)

    # Increasing scanning points
    idp[:] = replicate_scan(ori_crop, ds_shape)


   # Calculate semiconv_pix and com
    data_4D.data_4D = idp
    data_4D.estimate_aperture_size()

    # Create json
    par_dictionary['com'] = (data_4D.center_y,data_4D.center_x)
    par_dictionary['rad'] = data_4D.aperture_radius
    par_dictionary['complex_dtype'] = 'complex64'
    par_dictionary['order'] = 16
    par_dictionary['dim'] = dim

    path_json = "/Local/erc-1/bangun/LiveWDD_Data/params.json"
    
    with open(path_json, 'w') as f:
         json.dump(par_dictionary, f)
         
    return path_data, path_json

def setup_data(type_increase:str, 
               type_eval:str):
    """
    A function to generate increasing dimension of datasets
    
    Parameters
    ----------
    type_increase
        Option to increase dimension of scanning points or dimension of detectors
    type_eval
        Option to evaluate computation time or memory allocation

    Returns
    -------
    MC
        Monte carlo parameters or number of trials
    path_store
        The path to store the data for scanning points or detectors
    list_dim
        List of dimensions we want to evaluate
    set_scan_det
        Dictionary of increasing scanning points or detectors
    
    """
    
    if type_eval == 'Time':
        MC = 11
    else:
        MC = 1
    
    set_scan_det = {'scan':increase_scan,
                    'detector':increase_detector}
    
    if type_increase == 'scan':
        print('Increase the scanning points')
        path_store = '/Users/bangun/LiveWDD/Result/' + type_eval +'/Scan'
        # List scanning points
        list_dim = [(128,128,128,128)]
    elif type_increase == 'detector':
        print('Increasing the detector')
        path_store = '/Users/bangun/LiveWDD/Result/' + type_eval + '/Detector'
        list_dim = [(128,128,128,128)]
    else:
        raise RuntimeError('Choose increasing dimension!')
     
    return MC, path_store,list_dim, set_scan_det[type_increase]

 