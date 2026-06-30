""" Python module containing the interpolation of the APEC emissivity in temperature and abundance."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax


class APECEmissivity(object):
    """
    Universal temperature profile as defined in Ghirardini 2018+ in the X-COP cluster sample
    """

    def __init__(self, TZ_grid_to_interp_from = '/xifu/home/mola/SBI_Turbulence/data/flux_table_APEC_oldXIFU.npy'):
        """
        Init
        Parameters:
            TZ_grid_to_interp_from (string): path to temperature and abundance table
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
