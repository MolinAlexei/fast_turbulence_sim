""" Python module containing the interpolation of the APEC emissivity in temperature and abundance."""
from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import jax

class APECEmissivity(hk.Module):
    """
    Universal temperature profile as defined in Ghirardini 2018+ in the X-COP cluster sample
    """

    def __init__(self, TZ_grid_to_interp_from = '/xifu/home/mola/SBI_Turbulence/data/flux_table_APEC_oldXIFU.npy'):
        super(APECEmissivity, self).__init__()

        """
        OLD Version 
        #Load APEC flux computed for a regular grid of temperatures and abundances
        #flux_table = np.load('/xifu/home/mola/SBI_Turbulence/data/flux_table_z0.1.npy')
        #TZ_table = np.load('/xifu/home/mola/SBI_Turbulence/data/flux_interp_z0.1.npy',allow_pickle=True)
        #(Z_table, T_table, _,_)= TZ_table
        

        #Sum all photons in spectrum
        #flux_table_photons = np.sum(flux_table, axis = -1, weights )

        #Interpolating function
        #self.interp_function  = RegularGridInterpolator((Z_table, T_table), flux_table_photons)
        """

        """
        OLD Version 2 

        Npts = 100
        flux_table = np.reshape(np.load('data/flux_table_APEC_oldXIFU.npy'), (100,100)).T

        T_table = np.linspace(0.1, 10, Npts)
        Z_table = np.linspace(0.01, 1, Npts)

        self.interp_function = RegularGridInterpolator((T_table, Z_table), flux_table)
        """
        self.flux_table = jnp.reshape(jnp.load(TZ_grid_to_interp_from), (100,100)).T


    def __call__(self, Z, T):
        """
        Compute the temperature function for a given radius.

        Parameters:
            Z (float): Abundance 
			T (float) : Temperature (keV)
        Returns:
            (float): Interpolated flux in photons, for norm = 1 and exposure = 1s
        """

        ### OLD Version 2 : return self.interp_function((T, Z))

        # JAX Version

        Npts = 100
        T_table = jnp.linspace(0.1, 10, Npts)
        Z_table = jnp.linspace(0.01, 1, Npts)

        idxT = jnp.searchsorted(T_table, T, 'right')
        idxZ = jnp.searchsorted(Z_table, Z, 'right')

        T_coord = (T - T_table[idxT])/(T_table[idxT+1] - T_table[idxT]) + idxT
        Z_coord = (Z - Z_table[idxZ])/(Z_table[idxZ+1] - Z_table[idxZ]) + idxZ

        coordinates = jnp.array([T_coord, Z_coord])

        interp = jax.scipy.ndimage.map_coordinates(self.flux_table, coordinates, order = 1, cval=0.0)

        return interp
