import numpy as np
from typing import Tuple
from libertem.udf import UDF
from live_wdd import dim_reduct
from live_wdd import live_wdd
import typing
if typing.TYPE_CHECKING:
	import numpy.typing as nt


class WDDUDF(UDF):
    """
    Class that use UDF for live processing method, the implementation uses LiberTEM UDF

    Parameters
    ----------

    wiener_filter
        Pre computed Wiener filter for deconvolution process
    wiener_roi
        Region of interest of the scanning points
    row_exp
        Construction of row element of Fourier matrix
    col_exp
        Construction of column element of Fourier matrix
    complex_dtype
        Complex dtype for array

    """

    def __init__(self, wiener_filter: np.ndarray, 
                 wiener_roi, coeff:Tuple[np.ndarray, np.ndarray], 
                 row_exp: np.ndarray, 
                 col_exp:np.ndarray, 
                 complex_dtype:"nt.DTypeLike"):

        super().__init__(
            wiener_filter=wiener_filter,
            wiener_roi=wiener_roi,
            coeff=coeff,
            row_exp=row_exp,
            col_exp=col_exp,
            complex_dtype=complex_dtype
        )

    def get_result_buffers(self):
        """
        Method for preparation of variable output
        
        """
        return {
            'cut': self.buffer(
                kind='single',
                extra_shape=self.meta.dataset_shape.nav,
                dtype=self.params.complex_dtype
            ),
            'reconstructed': self.buffer(
                kind='nav',
                use='result_only',
                dtype=self.params.complex_dtype
            ),
        }

    def process_frame(self, frame):

        """
        Method for processing frame per scan, since modification of WDD
        we can model the Fourier transformation into summation
        """
        y, x = self.meta.coordinates[0]
        frame_compressed = dim_reduct.compress(frame, self.params.coeff)
        self.results.cut[:] += live_wdd.get_frame_contribution_to_cut_rowcol_exp(
            self.params.row_exp,
            self.params.col_exp,
            frame_compressed,
            y,
            x,
            tuple(self.meta.dataset_shape.nav),
            self.params.wiener_filter,
            self.params.wiener_roi,
            self.params.complex_dtype
        )

    def merge(self, dest, src):
        """
        Merge result 
        """
        dest.cut[:] += src.cut

    def get_results(self):
        """
        Inverse Fourier transform of the result in order to get into real space
        
        """
        real_cut = np.fft.ifft2((self.results.cut)).astype(self.params.complex_dtype)
        real_cut = real_cut/np.max(np.abs(real_cut))
        return {
            'cut': self.results.cut,
            'reconstructed': real_cut.conj() ,
        }
