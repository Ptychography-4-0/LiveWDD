import os
import json 
import subprocess
import sys 
from live_wdd.live_wdd import prepare_livewdd
from live_wdd.wdd_udf import WDDUDF
from live_wdd.prepare_resize_data import setup_data
import time
import numpy as np 
import os
path_current = os.path.abspath(os.getcwd())
from libertem.api import Context
import click
from libertem.common import Shape 
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.api import Context
from typing import Tuple
from libertem.io.dataset.base import DataSet
 
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

def compute_liveproc(path, path_json, MC):
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
    list_cpu = [1,2,4,8,16,32]
    time_core = []
    for idx_cpu in list_cpu:
        print('Number of core ', idx_cpu)
        no_cpu = range(idx_cpu)
        cs = cluster_spec(no_cpu,[],False)
        exc = DaskJobExecutor.make_local(cs)
	
        ctx = Context(exc)
    
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
                                                                                        complex_dtype, 6.0)

        
        
            # Run Live WDD
            print('Run Live Processing....')

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
        time_core.append(result)
        ctx.close()
    
    return time_core

def main( MC, path_data, path_json):

    """
    Run the solver in this case we only evaluate liveWDD

    Parameters
    ----------
    
    MC
        Number of trials for Monte Carlo
    
    
    path_data
        Path for dimension ata for livewdd

    path_json
        Path for parameters of datasets

    Returns
    -------
    result
        Dictionary of result contains computation time and phase reconstruction

    """

    result = compute_liveproc(path_data, path_json, MC)
    return result


if __name__ == '__main__':
    
    type_increase = 'scan'
    type_eval = 'Time'
    MC, path_store, list_dim, set_scan_det = setup_data(type_increase, type_eval)
    # Create directory
    os.makedirs(path_store, exist_ok = True)

 
    # Choose Solver
    solver = 'livewdd_core'
    formt = '.json'
    file_name = os.path.join(path_store, solver + formt)
    
     
    # Path file
    parfile ='/Users/bangun/pyptychostem-master/parameters.txt'
    total_result = []
    for idx in range(len(list_dim)):
        print('Processing dimension ', str(list_dim[idx]))
        path_data, path_json = set_scan_det(list_dim[idx],parfile)

        # Load Data and run reconstruction
        parfile_new = os.path.join(path_current, 'LiveWDD_Data/parameters_new.txt')
        
        # Run algorithm
        result = main(MC, path_data, path_json)

        total_result.append({'dimension': list_dim[idx],
                             'result': result})

        
        with open(file_name, 'w') as f:
            json.dump(total_result, f)
