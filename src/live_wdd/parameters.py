from typing import Tuple, Optional, Union, List
from typing_extensions import Literal
import scipy.constants as const
import math
import numpy as np
from libertem.udf.sum import SumUDF
from live_wdd.dim_reduct import get_sampled_basis

 
def wavelength(U:float):

    """
    Function to calculate the relativistic electron wavelength in meters
    we use unit parameter in scipy

    Parameters
    ----------
    
    U
        acceleration voltage in kV
     

    Return
    ------
    electron wavelength

    """
    
    e = const.elementary_charge  # Elementary charge  !!! 1.602176634×10−19
    h = const.Planck  # Planck constant    !!! 6.62607004 × 10-34
    c = const.speed_of_light  # Speed of light
    m_0 = const.electron_mass  # Electron rest mass

    T = e*U*1000
    lambda_e = h*c/(math.sqrt(T**2+2*T*m_0*(c**2)))
    return lambda_e


def est_scale(order, ds_shape,com,rad, float_dtype, pacbed):
    """
    Function to estimate optimal radius for Hermite-Gauss polynomials
    
    Parameters
    ----------
    
    order
        Maximum degree of polynomials
    ds_shape
        Dimension of four-dimensional datasets, contains nav for scanning points and sig for frame dimension
    com
        Center of mass coordinate (cy, cx)
    rad
        Radius of diffraction pattern in pixel
    float_dtype
        Data type that should be used for array

    Return
    ------
    Scaling radius for envelope of Hermite polynomials

    """
    pacbed = np.sum(pacbed, 0)
    
    err = []
    for sc in np.arange(1.0, 11.0):
        coeff = get_sampled_basis(order=order, 
                                  ds_shape=ds_shape,
                                  cy=com[0], cx=com[1],
                                  scale = sc,
                                  semiconv_pix=rad,
                                  float_dtype=float_dtype)

        err.append(np.linalg.norm((pacbed/np.max(pacbed))[:,np.newaxis] -
                                  np.abs(np.sum(coeff,0))/np.max(np.abs(np.sum(coeff,0)))))
        
    return np.argmin(err) + 1


def physical_coordinate(
    ds_shape:Tuple,
    acc:int,
    scan_real:float,
    semiconv:float,
    semiconv_pix:float,
    com:Tuple
    ):
    """
    Function to store important information related to the physical coordinates
    
    Parameters
    ----------
    ds_shape
        Dimension of four-dimensional datasets, contains nav for scanning points and sig for frame dimension
    acc
        Acceleration voltage in kV
    scan_real
        Distance between scanning points in nm
    semiconv
        Semiconvergence angle in radian
    semiconv_pix
        Radius of diffraction pattern in pixel
    com
        Center of mass coordinate (cy,cx)

    
    Return
    ------
    Dictionary that contains information for physical coordinates


    """
    scan_dim = np.array(ds_shape.nav)
    det_dim = np.array(ds_shape.sig)

    lamb = wavelength(acc)
    dpix = (scan_real)*1e-9
    semiconv = semiconv*1e-3
    d_Kf = np.sin(semiconv)/lamb/semiconv_pix
    d_Qp = 1/dpix/scan_dim
    y, x = np.ogrid[0:det_dim[0], 0:det_dim[1]]
     
    return {'d_Kf':d_Kf,    
            'd_Qp':d_Qp,
            'lamb':lamb,
            'x':x,
            'y':y,
            'cy':com[0],
            'cx':com[1],
            'semiconv_pix':semiconv_pix
             }
