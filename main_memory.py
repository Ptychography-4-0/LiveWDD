  
import os
import json
from live_wdd.prepare_resize_data import setup_data
import subprocess
import sys
# Set path for pyptychostem
sys.path.insert(1, '/Users/bangun/pyptychostem-master')
from STEM4D import WDD, Data4D 
from live_wdd.live_wdd import prepare_livewdd
from live_wdd.wdd_udf import WDDUDF
from live_wdd.prepare_resize_data import setup_data
import time
import numpy as np 
import os
from libertem.api import Context
import click
from libertem.common import Shape 

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

def run_livewdd(ds, ctx,
                wiener_filter_compressed,
                coeff,scan_idx,
                row_exp,col_exp,
                complex_dtype):
    
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
 

    conv_wdd_mc = []

    for it in range(MC):
        print('Trials :', it)
        conv_wdd = run_pyptychostem_wdd(parfile_new)

        conv_wdd_mc.append(conv_wdd.phase.tolist())

   
    return conv_wdd_mc 

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
        
    # Prepare data for live processing
    
    f = open(path_json)
    
    # returns JSON object as 
    # a dictionary
    par_dictionary = json.load(f)
    dim = par_dictionary['dim']
    order = par_dictionary['order']
    complex_dtype = np.dtype(par_dictionary['complex_dtype'])
    ds_shape = Shape(dim, sig_dims=2)

    # Create context
    ctx = Context()
    
    ds = ctx.load("npy", path=path, nav_shape=ds_shape.nav, sig_shape=ds_shape.sig)  
    
    ds.set_num_cores(4*18)

    acc = float(par_dictionary['voltage'])# in kV
    scan_real = float(par_dictionary['stepsize'])*1e-1 # in nm
    semiconv=float(par_dictionary['aperture'])*1e3 # In mrad
 
    
    otf_mc = []
    for it in range(MC):
        print('Pre Computed for Live Processing...')
        print('Trials :', it)
        scan_idx, wiener_filter_compressed, row_exp, col_exp,coeff = prepare_livewdd(ds_shape, acc, scan_real, 
                                                                                     semiconv, par_dictionary['rad'], 
                                                                                     par_dictionary['com'], order, 
                                                                                     complex_dtype,6.0)

    
    
        # Run Live WDD
        print('Run Live Processing....')

        otf = run_livewdd(ds, ctx,
                          wiener_filter_compressed,
                          coeff,scan_idx,
                          row_exp,col_exp,
                          complex_dtype)
        otf_mc.append(np.angle(otf).tolist())
     
     
    ctx.close()
    return otf_mc

@click.command()
@click.option('-s', '--solver', type=str)
@click.option('-p', '--parfile-new', type=str)
@click.option('-d', '--path-data', type=str)
@click.option('-j', '--path-json', type=str)
@click.option('-m', '--mc', type=int)
def main(solver, parfile_new, path_data, path_json, mc):
    """
    Run the solver independently between pyptychostem and livewdd

    Parameters
    ----------
    solver
        Choose between pyptychostem and livewdd solver
    mc
        Number of trials for Monte Carlo
    
    parfile_new
        Path for new dimension data for pyptychostem

    path_data
        Path for dimension data for livewdd

    path_json
        Path for parameters of datasets

    Returns
    -------
    recon_wdd
        Dictionary of result contains computation time and phase reconstruction

    """
   
    if solver == 'pyptychostem':
        recon_wdd = compute_pyptycho(mc, parfile_new)
    else:
        recon_wdd = compute_liveproc(path_data, path_json, mc)
    return recon_wdd
    
if __name__ == '__main__':
    main()
   
