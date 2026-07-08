""" Python module containing models for the abundance profile of the ICM."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


class XCOPAbundance(object):
    r"""Universal iron abundance profile as defined in Ghirardini 2018+ in the X-COP cluster sample

    $\mathrm{Fe}(x) = 0.21 (x + 0.021)^{-0.48} - 6.54\times \exp(- \frac{(r + 0.0816)^2}{0.0027})$
    """

    def __init__(self,
                 R500 = 1309.):
        """
        Parameters:
            R500 (jnp.array): Radius R500 of the cluster
        """

        self.R500 = R500

    def __call__(self, r):
        """
        Compute the abundance function for a given radius, following Mernier et al 2017.

        Parameters:
            r (jnp.array): Radius to compute the temperature in kpc

        Returns:
            (jnp.array): Abundance function evaluated at the given radius
        """

        x = r / self.R500
        Z = 0.21*(x+0.021)**(-0.48) - 6.54*jnp.exp( - (x+0.0816)**2 / (0.0027) )

        #Clip the values to the [0.1-0.9] range (otherwise the T-Z interpolation does not work.)
        return Z #jnp.clip(Z, 0.1, 0.9)
