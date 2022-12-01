
import os
import json
from typing import Tuple
from live_wdd.prepare_resize_data import setup_data
import subprocess
import sys
sys.path.insert(1, '/Users/bangun/pyptychostem')
from STEM4D import WDD, Data4D
from live_wdd.live_wdd import prepare_livewdd
from live_wdd.wdd_udf import WDDUDF
from live_wdd.prepare_resize_data import setup_data
import time
import numpy as np 
import os
from libertem.io.dataset.base import DataSet
from libertem.api import Context
import click
from libertem.common import Shape 
from perf_utils import timer

def run_pyptychostem_wdd(parfile_new):

    """
    A function to run conventional Wigner Distribution Deconvolution
    with pyptychostem

    Parameters
    ----------
    
    parfile_new
        Path of datasets

    Returns
    -------
    wdd
        Instance from reconstruction contains phase, amplitude, etc
    """
    # Create Dictionary
    par_dictionary = {}
    file = open(parfile_new)

    for line in file:
        if line.startswith('##'):
            continue
        split_line = line.rstrip().split('\t')


        if len(split_line)!=2:
            continue
        key, value = split_line
        par_dictionary[key] = value

    # Load data
    data_4D = Data4D(parfile_new)
    data_4D.estimate_aperture_size() 

    # Dose 
    if int(par_dictionary.get('dose',-1))  >0:
        print('Adding Dose..')
        data_4D.apply_dose(int(par_dictionary.get('dose',-1)))


    expansion_ratio = float(par_dictionary.get('CBED/BF',-1))
    if  expansion_ratio<1:
        expansion_ratio = None
    data_4D.truncate_ronchigram(expansion_ratio=expansion_ratio) # crops ronchigram to area of interest
    
    
    # Spatial frequency
    data_4D.apply_FT()

    # Run wdd
    wdd = WDD(data_4D)
    wdd.run()

    
    return wdd
    

def run_livewdd(ds: DataSet, 
                ctx: Context,
                wiener_filter_compressed:np.ndarray,
                coeff:Tuple,
                scan_idx:np.ndarray,
                row_exp:np.ndarray,
                col_exp:np.ndarray,
                complex_dtype:str):

    """
    A function to run live Wigner Distribution Deconvolution

    Parameters
    ----------
    ds
        DataSet from liberTEM
    ctx
        Context to run the UDF
    scan_idx
        Non zero index for overlapping region
    wiener_filter_compressed
        Wiener filter after dimensionality reduction for deonvolutoin process
    row_exp
        Fourier matrix applied on the row space
    col_exp
        Fourier matrix applied on the column space
    coeff
        Matrix for dimensionality reduction
    complex_dtype
        Number of floating points    
    Returns
    -------
    result
        Dictionary results contains phase reconstruction and computation time
    """
    
     
    # Run live wdd
    
    live_wdd = ctx.run_udf(dataset=ds, roi = None,
                           udf= WDDUDF(wiener_filter_compressed, 
                                       scan_idx, coeff, 
                                       row_exp, col_exp, complex_dtype))

    otf = live_wdd['reconstructed']
  

    return otf 

def compute_pyptycho(MC, parfile_new):
    """
    Calculate pyptychoSTEM

    Parameters
    ----------
    parfile_new
        Path to the datasets
    MC
        Number of trials
        
    Returns
    -------
    result
        Dictionary results contains phase reconstruction and computation time
    """
    
    
    # Run PyPtychoSTEM
    print('Run PyPtychoSTEM....')
 
    # Pre-allocation time
    time_all = {'perf': [],
                'thread': [],
                'process': []}

    conv_wdd_mc = []

    for it in range(MC):
        print('Trials :', it)
    

        # Start calculate time
        start_perf = time.perf_counter()
        start_thread = time.thread_time()
        start_process = time.process_time()

        # Run PyptychoSTEM
        conv_wdd = run_pyptychostem_wdd(parfile_new)

        conv_wdd_mc.append(conv_wdd.phase.tolist())

        # Stop calculate time
        end_perf = time.perf_counter()
        end_thread = time.thread_time()
        end_process = time.process_time()
    
        # Calculate time

        time_all['perf'].append(end_perf - start_perf)
        time_all['thread'].append(end_thread - start_thread)
        time_all['process'].append(end_process - start_process)


    result = {'recon_phase':conv_wdd_mc,
              'run_time': time_all}

    return result

def compute_liveproc(path:str, 
                     path_json:str, 
                     MC:int):
    """
    Calculate live Wigner Distribution Deconvolution

    Parameters
    ----------
    path
        Path to the datasets
    path_json
        Path to the parameters datasets
    MC
        Number of trials

    Returns
    -------
    result
        Dictionary results contains phase reconstruction and computation time
    """
    
    # Create context
    ctx = Context()
    
    # Pre-allocation time
    time_all = {'perf': [],
                'thread': [],
                'process': []}

    
    otf_mc = []
    for it in range(MC):           
        print('Trials :', it)
    

        # Start calculate time
        start_perf = time.perf_counter()
        start_thread = time.thread_time()
        start_process = time.process_time()

        # Prepare data for live processing
        f = open(path_json)
        
        # returns JSON object as 
        # a dictionary
        par_dictionary = json.load(f)
        dim = par_dictionary['dim']
        order = par_dictionary['order']
        complex_dtype = np.dtype(par_dictionary['complex_dtype'])
        ds_shape = Shape(dim, sig_dims=2)

        
        ds = ctx.load("npy", path=path, nav_shape=ds_shape.nav, sig_shape=ds_shape.sig)  
        
        ds.set_num_cores(4*18)

        acc = float(par_dictionary['voltage'])# in kV
        scan_real = float(par_dictionary['stepsize'])*1e-1 # in nm
        semiconv=float(par_dictionary['aperture'])*1e3 # In mrad
 
    
        print('Pre Computed for Live Processing...')
        
        scan_idx, wiener_filter_compressed, row_exp, col_exp,coeff = prepare_livewdd(ds_shape, acc, scan_real, 
                                                                                     semiconv, par_dictionary['rad'], 
                                                                                     par_dictionary['com'], order, 
                                                                                     complex_dtype,
                                                                                     ctx, ds)
 
    
        # Run Live WDD
        print('Run Live Processing....')
        with timer("Time for live processing"):    
            otf = run_livewdd(ds, ctx,
                            wiener_filter_compressed,
                            coeff,scan_idx,
                            row_exp,col_exp,
                            complex_dtype)
            otf_mc.append(np.angle(otf).tolist())
     
        # Stop calculate time
        end_perf = time.perf_counter()
        end_thread = time.thread_time()
        end_process = time.process_time()
    
        # Calculate time

        time_all['perf'].append(end_perf - start_perf)
        time_all['thread'].append(end_thread - start_thread)
        time_all['process'].append(end_process - start_process)

    result = {'recon_phase':otf_mc,
              'run_time': time_all}

    ctx.close()
    return result

def main(solver:str, 
         MC:int, 
         parfile_new:str, 
         path_data:str, 
         path_json:str):
    """
    Run the solver independently between pyptychostem and livewdd

    Parameters
    ----------
    solver
        Choose between pyptychostem and livewdd solver
    MC
        Number of trials for Monte Carlo
    
    parfile_new
        Path for new dimension data for pyptychostem

    path_data
        Path for dimension ata for livewdd

    path_json
        Path for parameters of datasets

    Returns
    -------
    result
        Dictionary of result contains computation time and phase reconstruction

    """

    if solver == 'pyptychostem':
        result = compute_pyptycho(MC, parfile_new)
    elif solver == 'livewdd':
        result = compute_liveproc(path_data, path_json, MC)
    else:
        raise RuntimeError('No Solver!')

    return result


if __name__ == '__main__':
    
    type_increase = 'detector'
    type_eval = 'Time'
    MC, path_store, list_dim, set_scan_det = setup_data(type_increase, type_eval)
    
    # Choose Solver
    solver = 'livewdd'
    formt = '.json'
    file_name = os.path.join(path_store, solver + formt)
    
     
    # Path file
    parfile ='/Users/bangun/pyptychostem/parameters.txt'
    total_result = []
    for idx in range(len(list_dim)):
        print('Processing dimension ', str(list_dim[idx]))
        path_data, path_json = set_scan_det(list_dim[idx],parfile)

        # Load Data and run reconstruction
        parfile_new = '/Local/erc-1/bangun/LiveWDD_Data/parameters.txt'
        
        # Run algorithm
        result = main(solver, MC, parfile_new, path_data, path_json)

        total_result.append({'dimension': list_dim[idx],
                             'result': result})

        
        #with open(file_name, 'w') as f:
        #    json.dump(total_result, f)
