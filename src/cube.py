from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax.numpy.fft as fft
from grid import SpatialGrid3D, FourierGrid3D
from turbulence import KolmogorovPowerSpectrum
from beta_model import BetaModel, XBetaModel, FlatModel, VikhlininModel
import astropy.units as u
from emissivity import XrayEmissivity

import matplotlib.pyplot as plt

class EmissivityCube(hk.Module):
    """
    Generate an emissivity cube
    """
    
    def __init__(self, spatial_grid, exposure = 125e3):
        super(EmissivityCube, self).__init__()
        self.power_spectrum = KolmogorovPowerSpectrum()
        self.spatial_grid = spatial_grid
        self.exposure = exposure #exposure in seconds
        pixsize_cm = self.spatial_grid.pixsize * u.kiloparsec.to(u.cm)
        
        model = XrayEmissivity()
        self.lam = model(self.spatial_grid.R) * pixsize_cm**3 * self.exposure

        
        
    def __call__(self):
        
        key = hk.next_rng_key()


        field_spatial = np.random.poisson(
                                       lam  = self.lam, 
                                       size = self.spatial_grid.shape)
        return field_spatial



class FluctuationCube(hk.Module):
    """
    Generate a fluctuation cube as seen from https://garrettgoon.com/gaussian-fields/
    """
    

    def __init__(self, spatial_grid):
        super(FluctuationCube, self).__init__()
        self.power_spectrum = KolmogorovPowerSpectrum()
        self.spatial_grid = spatial_grid
        self.fourier_grid = FourierGrid3D(self.spatial_grid)
        self.K = self.fourier_grid.K.astype(np.float64)
        
    def __call__(self):
        
        key = hk.next_rng_key()
        
        #Dont mind the rfft, it is here to gain memory
        field_spatial = random.normal(key, shape=self.spatial_grid.shape)
        field_fourier = fft.rfftn(field_spatial)*jnp.sqrt(self.power_spectrum(self.K)/self.spatial_grid.pixsize**3)
        field_spatial = fft.irfftn(field_fourier, s=self.spatial_grid.shape)
        
        return field_spatial
