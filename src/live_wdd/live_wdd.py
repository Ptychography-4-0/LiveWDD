from typing import Tuple, Optional, Union, List
from typing_extensions import Literal
from live_wdd.dim_reduct import compress, get_sampled_basis
from libertem.corrections.coordinates import identity
from live_wdd.parameters import physical_coordinate, est_scale
from perf_utils import timer
import typing
import numpy as np
import numba
import time
if typing.TYPE_CHECKING:
	import numpy.typing as nt

    
def prepare_livewdd(ds_shape: Tuple,
                    acc: int, 
                    scan_real: float, 
                    semiconv: float,
                    rad: float, 
                    com:Tuple, 
                    order: int, 
                    complex_dtype: str,
                    ctx, ds,
                    transformation=None):
    
    """
    Prepare for live WDD incuding calculating physical coordinate,
    Construct the basis function for dimensionality reduction,
    Fourier matrices for per scanning point transformation,
    Initial probe on the reciprocal space, i.e., circular aperture,
    Wiener fulter
    
    Parameters
    -----------------
    ds_shape
        dimension of dataset navigation shape (Ny,Nx) and detector shape (Sy,Sx)
    acc
        Acceleration voltage in kV
    scan_real
        Step size in nm
    semiconv
        Semiconvergence angle in mrad
    rad
        Semiconvergence angle in pixel
    com 
        Center of mass
    order
        Bandlimited number of dimensionality reduction, i.e., L = 16
    complex_dtype
        Predefined dtype of our data
    ctx
        Contex to run libertem
    ds
        dataset and its parameters
        
    Return
    ------
    scan_idx
        Non zero index for overlapping region
    wiener_filter_compressed
        Wiener filter after dimensionality reduction
    row_exp
        Fourier matrix applied on the row space
    col_exp
        Fourier matrix applied on the column space
    coeff
        Matrix for dimensionality reduction
    
    """
     
    if np.dtype(complex_dtype) == np.complex64:
        float_dtype = np.float32
    elif np.dtype(complex_dtype) == np.complex128:
        float_dtype = np.float64
    else:
        raise RuntimeError(f"unknown complex dtype: {complex_dtype}")

    params = physical_coordinate(ds_shape=ds_shape,
                                 acc=acc,
                                 scan_real=scan_real,
                                 semiconv=semiconv,
                                 semiconv_pix=rad,
                                 com=com)
   

    # Get Matrix from Basis functions
    scale = est_scale(order, ds_shape,com,rad, complex_dtype, ctx, ds)
 
    coeff = get_sampled_basis(order=order, 
                          ds_shape=ds_shape,
                          cy=com[0], cx=com[1],
                          scale = scale,
                          semiconv_pix=rad, 
                          float_dtype = float_dtype)

    # Fourier Transform
    row_exp, col_exp = f2d_matrix_replacement(ds_shape=ds_shape, complex_dtype=complex_dtype)                       
    
    # Filter center
    filter_center = probe_initial(com=com,
                                  ds_shape=ds_shape,
                                  semiconv_pix=rad,
                                  float_dtype=float_dtype,
                                  norm=True,
                              )
    
    # Avoid zero divisiion
    epsilon = 1e-2
    
    # Pre-computed Wiener filter
    with timer("pre computing Wiener"):    
        wiener_filter_compressed = pre_computed_Wiener(scan_dim=tuple(ds_shape.nav),
                                                       order=order,
                                                       probe_center=filter_center,
                                                       params=params,
                                                       coeff=coeff,
                                                       epsilon=epsilon,
                                                       complex_dtype=complex_dtype,
                                                       transformation=transformation)
    scan_idx = roi_wiener_filter(wiener_filter_compressed)
 
    
    return scan_idx, wiener_filter_compressed,row_exp, col_exp, coeff


def probe_initial(
    com:Tuple,
    ds_shape:Tuple,
    semiconv_pix: float,
    float_dtype: 'nt.DTypeLike',
    norm: Optional[bool] = False,
    ):
    """
    A function to generate circular aperture, since we are working on the Fourier space the probe is circular aperture

    Parameters
    ----------
    com
        Center of mass (cy,cx)
    ds_shape
        Dimension of four-dimensional data
    semiconv_pix
        Radius on the pixel

    Return
    ------
    filter center
        Circular aperture at the center
    """
    y, x = np.ogrid[0:ds_shape.sig[0], 0:ds_shape.sig[1]]
    filter_center = (y -com[0] )**2 + (x - com[1])**2 < semiconv_pix**2
    
    if norm:
        filter_center = filter_center/np.linalg.norm(filter_center)
    return filter_center.astype(float_dtype)

 


compress_jit = numba.njit(compress)
 

def pre_computed_Wiener(
    scan_dim:Tuple, 
    order:int,
    probe_center:np.ndarray,
    params:dict,
    coeff:Tuple[np.ndarray,np.ndarray],
    epsilon:float,
    complex_dtype: "nt.DTypeLike" = np.complex64,
    transformation: Optional[np.ndarray] = None,
    ):
    
    """
    A wrapper function to choose if we use numba or not for calculating Wiener filter
    
    Parameters
    ----------
    scan_dim 
        Dimension of scaning points
    
    order
        Maximum degree of polynomials
    probe_center
        Circular aperture
    
    params
        Dictionary related to physical coordinate
    coeff 
        Matrix from sampled Hermite-Gauss polynomials for both x and y direction
    epsilon
        Small factor to avoid zero division
    
    """
    
    
    if transformation is None:
        transformation = identity()
    
    y = params['y']
    x = params['x']
    cy = params['cy']
    cx = params['cx']
    d_Kf = params['d_Kf']
    d_Qp = params['d_Qp']
    semiconv_pix = params['semiconv_pix']
    x,y = np.meshgrid(y,x)
    wiener_filter_compressed =  pre_computed_Wiener_jit(scan_dim, order, probe_center,
                                                        y, x, cy, cx, d_Kf, d_Qp, semiconv_pix,
                                                        coeff, epsilon, complex_dtype, transformation)
  
    return wiener_filter_compressed

def pre_computed_Wiener_wojit(
    scan_dim:Tuple, 
    order:int,
    probe_center:np.ndarray,
    params:dict,
    coeff:Tuple[np.ndarray,np.ndarray],
    epsilon:float,
    complex_dtype: "nt.DTypeLike" = np.complex64,
    transformation: Optional[np.ndarray] = None,
    ):
    """
    A function to calculate compressed Wiener filter the result in compressed space, shape (q, p, order, order)
    It should be noted here we only implement Wiener filter with respect to initial probe, to esitmate the probe
    the shift direction should be reversed
    
    Parameters
    ----------
    scan_dim 
        Dimension of scaning points
    
    order
        Maximum degree of polynomials
    probe_center
        Circular aperture
    
    params
        Dictionary related to physical coordinate
    coeff 
        Matrix from sampled Hermite-Gauss polynomials for both x and y direction
    epsilon
        Small factor to avoid zero division
    
    """
    
    # Allocation for result
    wiener_filter_compressed = np.zeros(scan_dim + (order, order), dtype=complex_dtype)

    if transformation is None:
        transformation = identity()
    
    # Array of scanning dimension, should we avoid np.array?
    scan_dim = np.array(scan_dim)
    for q in range(scan_dim[0]):
        for p in range(scan_dim[1]):

            # Get physical coordinate
            qp = np.array((q, p))
            flip = qp > scan_dim / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - scan_dim[flip]

            # Shift of diffraction order relative to zero order
            # without rotation in physical coordinates
            real_sy_phys, real_sx_phys = real_qp * params['d_Qp']
            # We apply the transformation backwards to go
            # from physical orientation to detector orientation,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            # Afterwards, we transform from physical detector coordinates
            # to pixel coordinates
            sy, sx = ((real_sy_phys, real_sx_phys) @ transformation) / params['d_Kf']

            # Shift in physical coordinate
            # sy, sx = real_qp * params['d_Qp']/params['d_Kf']
            
            # Shift probe
            probe_shift = (params['y'] - params['cy'] + sy)**2 + (params['x'] - params['cx'] + sx)**2 < params['semiconv_pix']**2

            # Autocorrelation
            product_conjugate = np.conj(probe_center)*probe_shift
            
            ## Get the inverse Wigner function
            ## Below is wrong it should be compressed space, since we change ifft2 to compressed basis
            #wigner_func_inv =  np.fft.ifft2((product_conjugate)).astype(np.complex64)
            

            wigner_func_inv = compress(product_conjugate, coeff).astype(np.complex64)
            
            # Wiener filter
           
         
            wiener_filter = np.conj(wigner_func_inv)/(np.abs(wigner_func_inv)**2 + epsilon)
            
           
            wiener_filter_compressed[q,p] = wiener_filter
            
    return wiener_filter_compressed

@numba.njit(parallel=True, cache=True)
def pre_computed_Wiener_jit(
    scan_dim:Tuple, 
    order:int,
    probe_center:np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    cy: float,
    cx: float,
    d_Kf: float,
    d_Qp: float,
    semiconv_pix: float,
    coeff:Tuple[np.ndarray,np.ndarray],
    epsilon:float,
    complex_dtype: 'nt.DTypeLike',
    transformation: np.ndarray,
    ):
    """
    A function to calculate compressed Wiener filter the result in compressed space, shape (q, p, order, order)
    It should be noted here we only implement Wiener filter with respect to initial probe, to esitmate the probe
    the shift direction should be reversed. In this case we use numba for faster implementation
    
    Parameters
    ----------

    scan_dim 
        Dimension of scaning points
    
    order
        Maximum degree of polynomials

    probe_center
        Circular aperture
    y
        Grid on the y dimensional axes
    x
        Grid on the x dimensional axes
    cy
        Center of mass coordinate on y axes
    cx
        Center of mass coordinate on x axes
    d_Kf
        TODO: what is this?
    d_Qp
        TODO: what is this?

    semiconv_pix
        Radius of diffraction pattern in pixel
        
    coeff 
        Matrix from sampled Hermite-Gauss polynomials for both x and y direction

    epsilon
        Small factor to avoid zero division
    """
    

    # Allocation for result
    wiener_filter_compressed = np.zeros(scan_dim + (order, order), dtype=complex_dtype)
    scan_dim = np.array(scan_dim)
    for q in numba.prange(scan_dim[0]):
        for p in range(scan_dim[1]):

            # Get physical coordinate
            qp = np.array((q, p))
            flip = qp > scan_dim / 2
            real_qp = qp.copy()
            real_qp[flip] = qp[flip] - scan_dim[flip]
             
            # Shift of diffraction order relative to zero order
            # without rotation in physical coordinates
            real_sy_phys, real_sx_phys = real_qp * d_Qp
            # We apply the transformation backwards to go
            # from physical orientation to detector orientation,
            # while the forward direction in center of mass analysis
            # goes from detector coordinates to physical coordinates
            # Afterwards, we transform from physical detector coordinates
            # to pixel coordinates
            sy, sx = (np.array((real_sy_phys, real_sx_phys)) @ transformation) /d_Kf
            
             # Shift probe
            probe_shift = (y - cy + sy)**2 + (x - cx + sx)**2 < semiconv_pix**2
            
            # Autocorrelation
            product_conjugate = probe_center*np.conj(probe_shift)
            
            ## Get the inverse Wigner function
            ## Below is wrong it should be compressed space, since we change ifft2 to compressed basis
            #wigner_func_inv =  np.fft.ifft2((product_conjugate)).astype(np.complex64)
            

            wigner_func_inv = compress_jit(product_conjugate, coeff) 
            
            # Wiener filter
            
            
            wiener_filter_compressed[q,p] = np.conj(wigner_func_inv)/(np.abs(wigner_func_inv)**2 + epsilon)

    return wiener_filter_compressed

def roi_wiener_filter(
    wiener_filter_compressed:np.ndarray
    ):

    """
    A function to calculate which scanning points has zero Wiener filter

    Parameters
    ----------
    wiener_filter_compressed
        Four-dimensional wiener filter
    
    Returns
    -------
        The non-zero scanning position
    """
    check_zero = np.sum(np.abs(wiener_filter_compressed), axis = (2,3))
    
    return np.argwhere(~np.isclose(np.abs(check_zero),0))

def f2d_matrix_replacement(
    ds_shape:Tuple,
    complex_dtype: 'nt.DTypeLike'
    ):

    """
    A Function to generate sampled Fourier basis, since we process the data
    per frame, it is better to use sampled Fourier basis not fft

    Parameters
    ----------
    ds_shape
        Dimension of datasets
    complex_dtype
        Pre defined data type of the elements

    Returns
    -------
    row_exp
        Sampled Fourier basis in terms of row dimension
    col_exp
        Sampled Fourier basis in terms of column dimension

    """
    reconstruct_shape = tuple(ds_shape.nav)
    row_steps = -2j*np.pi*np.linspace(0, 1, reconstruct_shape[0], endpoint=False)
    col_steps = -2j*np.pi*np.linspace(0, 1, reconstruct_shape[1], endpoint=False)
     
    full_y = reconstruct_shape[0]
    full_x = reconstruct_shape[1]

    # This creates a 2D array of row x spatial frequency
    row_exp = np.exp(
        row_steps[:,np.newaxis]
        * np.arange(full_y)[np.newaxis,:]
    )
    # This creates a 2D array of col x spatial frequency
    col_exp = np.exp(
        col_steps[:, np.newaxis]
        *np.arange(full_x)[np.newaxis, :]
    )
    # Fourier matrix has normalization
    return (1/np.sqrt(reconstruct_shape[0]))*row_exp.astype(complex_dtype), ((1/np.sqrt(reconstruct_shape[1]))*col_exp).astype(complex_dtype)


@numba.njit(fastmath=True, parallel=False)
def get_frame_contribution_to_cut_rowcol_exp(
    row_exp:np.ndarray, 
    col_exp:np.ndarray, 
    frame_compressed:np.ndarray, 
    y:int, 
    x:int,
    nav_shape:Tuple, 
    wiener_filter_compressed:np.ndarray, 
    scan_idx:np.ndarray,
    complex_dtype: 'nt.DTypeLike',
    ):

    """
    A Function to process data per frame and calculate the spatial frequency

    Parameters
    ----------
    row_exp
        Sampled Fourier basis on the row dimension
    col_exp
        Sampled Fourier basis on the column dimension
    frame_compressed
        Frame diffraction patterns after dimensionality reduction
    y  
        Certain scanning point on y dimension
    x
        Certain scanning point on x dimension
    nav_shape
        Dimension of scanning points
    wiener_filter
        Four-dimensional Wiener filter after dimensionality reduction
    scan_idx
        Non zero contribution on the scanning points
    complex_dtype
        Pre defined complex dtype

    Returns
    -------
    cut
        Reconsruction of the specimen transfer function on Fourier space
        per scanning point

    """

    cut = np.zeros(nav_shape, dtype=complex_dtype)
    for nn_idx in range(len(scan_idx)):
        q = scan_idx[nn_idx][0]
        p = scan_idx[nn_idx][1]
        

        # assuming we have isotropic sampling, np.allclose(row_exp, col_exp) holds:
         # If scan dim is not rectangular we cant use row_exp*row_exp!
        exp_factor = row_exp[y, q] * col_exp[x, p]
        acc = np.zeros((1), dtype=complex_dtype)[0]
        wiener_qp = wiener_filter_compressed[q, p]
        # NOTE: we still can cut this in ~half, as `frame_compressed` should be truncated to
        # zero in the lower-right triangle 
        for yy in range(frame_compressed.shape[0]):
            for xx in range(frame_compressed.shape[1]):
                acc += frame_compressed[yy, xx] * wiener_qp[yy, xx]
        cut[q, p] = (acc * exp_factor)
    return cut


def wdd_per_frame_combined(
    idp:np.ndarray, 
    coeff:Tuple[np.ndarray,np.ndarray],  
    wiener_filter_compressed:np.ndarray, 
    scan_idx:np.ndarray,
    row_exp:np.ndarray,
    col_exp:np.ndarray,
    complex_dtype: 'nt.DTypeLike',
    ):
    """
    WDD in harmonic compressed space and process the data per frame

    Parameters
    ----------
    idp
        Intensity of diffraction patterns of 4D-STEM
    
    wiener_filter_compressed
        Four-dimensional compressed Wiener filter 
    
    scan_idx
        Non-zero index position
    
    row_exp
        Sampled Fourier basis on the row dimension
    col_exp
        Sampled Fourier basis on the column dimension
 
    coeff 
        Matrix from sampled Hermite-Gauss polynomials for both x and y direction
    complex_dtype
        Pre defined complex dtype
    """
    nav_shape = idp.shape[:2]
    cut = np.zeros((nav_shape[0], nav_shape[1]), dtype = complex_dtype)
    # idp = np.zeros((128, 64, 256, 256), dtype=np.float32)  # put experimental data here
    #idp_compressed = np.zeros(nav_shape + (order, order), dtype=np.complex64)
    for y in range(nav_shape[0]):
        for x in range(nav_shape[1]):
            idp_compressed = compress(idp[y, x], coeff)
            cut += get_frame_contribution_to_cut_rowcol_exp(
                row_exp, col_exp, idp_compressed, y, x, nav_shape, 
                wiener_filter_compressed, scan_idx,complex_dtype,
            )
    real_cut = np.fft.ifft2((cut)).astype(complex_dtype)
    real_cut = real_cut/np.max(np.abs(real_cut))
    return  real_cut.conj()