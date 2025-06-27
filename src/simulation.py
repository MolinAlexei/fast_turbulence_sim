from jax import config
config.update("jax_enable_x64", True)
import haiku as hk 
import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
import jax.random as random
import haiku as hk
from haiku.initializers import Constant
from astropy.io import fits
 
from cube import FluctuationCube

def rng_key():

    return random.PRNGKey(np.random.randint(0, int(1e6)))


class Simulation(hk.Module):

    def __init__(self, 
                spatial_grid,
                structure_function,
                binning,
                projection,
                radial_bins_mes_errors,
                censhift_offsets,
                censhift_errors,
                broad_offsets,
                broad_errors,
                ):
        """
        Initialize simulation model

        Parameters:
            spatial_grid (hk.Module): 3D spatial grid
            structure_function (hk.Module): Structure Function
            binning (hk.Module): Binning
            projection (hk.Module) : Projection
            radial_bins_mes_errors (jnp.array): Bounds of the radial bins defininf the measurement error
            censhift_offsets (jnp.array): Means of censhift measurement error
            censhift_errors (jnp.array): stds of censhift measurement error
            broad_offsets (jnp.array): Means of broadening measurement error
            broad_errors (jnp.array): std of censhift measurement error
        """

        super(Simulation, self).__init__()
        

        # Initialize 
        self.fluctuation_generator = FluctuationCube(spatial_grid)
        #self.pars = self.fluctuation_generator.init(rng_key())
        
        # Binning
        self.binning = binning

        # Structure Function function
        self.StructureFunction = structure_function

        # Projection
        self.projection = projection
        
        # Half size of the grid
        half_grid_size = spatial_grid.shape[0]/2
        
        # Distances of bins to center of grid
        rBar_bins = jnp.sqrt((binning.xBar_bins - half_grid_size)**2 + (binning.yBar_bins - half_grid_size)**2)
        
        # Spread in centroid shift measurement error in each bin
        self.offsets_v = censhift_offsets[jnp.searchsorted(radial_bins_mes_errors, rBar_bins)-1]
        self.errors_v = censhift_errors[jnp.searchsorted(radial_bins_mes_errors, rBar_bins)-1]
        
        # Spread in broadening measurement error in each bin
        self.offsets_std = broad_offsets[jnp.searchsorted(radial_bins_mes_errors, rBar_bins)-1]
        self.errors_std = broad_errors[jnp.searchsorted(radial_bins_mes_errors, rBar_bins)-1]
        
    def __call__(self):
        """
        Creates a realization of a GRF for the speed along the los
        Projects the speed with em-weighting in binned maps
        Adds measurement error
        Returns the structure function

        Returns:
            dist (jn.array): Vector of separations at which the SF is computed
            sf (jnp.array): SF of centroid shift
            sf_std (jnp.array): SF of broadening
            v_vec (jnp.array): Vector of centroid shifts
            std_vec (jnp.array): Vector of broadenings
        """

        key = hk.next_rng_key()

        v_cube = self.fluctuation_generator()

        _,_,v_vec, std_vec = self.projection(v_cube)

        #Add measurement error on centroid shift
        err_v = random.multivariate_normal(key = key,
                                            mean = self.offsets_v, 
                                            cov = jnp.diag(self.errors_v**2))

        v_vec += jnp.where(jnp.invert(jnp.isnan(err_v)), err_v, 0)
        #v_vec = v_vec.at[jnp.invert(jnp.isnan(err_v))].add(err_v)


        # Add measurement error on broadening 
        # with a Gamma distribution, so that for high broadening values, 
        # the errors are distributed normally around the expected vector
        # and for low broadening values, the errors are always positive
        
        mu = std_vec + self.offsets_std
        a = mu**2 / self.errors_std**2
        scale = self.errors_std**2 / mu
        std_vec = random.gamma(key = rng_key(),a = a) * scale


        #SF of velocity map
        dist, sf = self.StructureFunction(v_vec , 
                                          self.binning.xBar_bins, 
                                          self.binning.yBar_bins)
        dist, sf_std = self.StructureFunction(std_vec, 
                                              self.binning.xBar_bins, 
                                              self.binning.yBar_bins)
        
        return dist, sf, sf_std, v_vec, std_vec
