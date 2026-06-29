from jax import config
config.update("jax_enable_x64", True)
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax.numpy.fft as fft
from .grid import FourierGrid3D
from .turbulence import KolmogorovPowerSpectrum
import astropy.units as u
from .emissivity import XrayEmissivity

class EmissivityCube(object):
    """
    Generate an emissivity cube
    """
    
    def __init__(self,
                 spatial_grid,
                 exposure = 125e3):
        """
        Initialize

        Parameters:
            spatial_grid (class): 3D Spatial grid
            exposure (float): Exposure time in seconds

        """ 
        self.power_spectrum = KolmogorovPowerSpectrum()
        self.spatial_grid = spatial_grid
        self.exposure = exposure #exposure in seconds
        pixsize_cm = self.spatial_grid.pixsize * u.kiloparsec.to(u.cm)
        
        model = XrayEmissivity()
        self.lam = model(self.spatial_grid.R) * pixsize_cm**3 * self.exposure

        
        
    def __call__(self):
        """
        Returns:
            field_spatial (jnp.array): Random realization of emissivity cube

        """
        field_spatial = np.random.poisson(
                                       lam  = self.lam, 
                                       size = self.spatial_grid.shape)
        return field_spatial



class FluctuationCube(object):
    """
    Generate a fluctuation cube as seen from https://garrettgoon.com/gaussian-fields/
    """
    

    def __init__(self, spatial_grid):
        """
        Initialize. The power spectrum is implicitly defined from the Kolmogorov power spectrum

        Parameters:
            spatial_grid (class): 3D Spatial grid
            exposure (float): Exposure time in seconds

        """ 

        self.power_spectrum = KolmogorovPowerSpectrum()
        self.spatial_grid = spatial_grid
        self.fourier_grid = FourierGrid3D(self.spatial_grid)
        self.K = self.fourier_grid.K.astype(np.float64)
        self.k_max = jnp.max(self.fourier_grid.K.flatten()[1:])
        
    def __call__(self,
                 rng_key : jax.random.PRNGKey,
                 sigma: float,
                 log_inj: float,
                 log_dis: float,
                 alpha: float
                 ):
        """
        Returns:
            field_spatial (jnp.array): Random realization of GRF

        """
        rng_key, sub_key = jax.random.split(rng_key)

        PS, corr_factor = self.power_spectrum(self.K, sigma, log_inj, log_dis, alpha, self.k_max / jnp.sqrt(2))
        
        #Dont mind the rfft, it is here to gain memory
        field_spatial = random.normal(rng_key, shape=self.spatial_grid.shape)
        field_fourier = fft.rfftn(field_spatial)*jnp.sqrt(PS/self.spatial_grid.pixsize**3)
        field_spatial = fft.irfftn(field_fourier, s=self.spatial_grid.shape)

        # Add correction factor
        field_spatial += random.normal(sub_key, shape=self.spatial_grid.shape) * jnp.sqrt(corr_factor)
        
        return field_spatial
