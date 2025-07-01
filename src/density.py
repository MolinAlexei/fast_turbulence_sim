from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import jax.numpy as jnp
from haiku.initializers import Constant


class VikhlininModel(hk.Module):
    """
    Density model which use a modified Vikhlinin functional form, with gamma fixed to 3.
    See Ghirardini et al. 2019

    """
    
    def __init__(self):
        super(VikhlininModel, self).__init__()

    def __call__(self, r):
        r"""Compute the density function for a given radius.
        
        $$n_e^2(x)= n_0^2 \frac{(\frac{x}{r_c})^{-\alpha}}{(1 + (\frac{x}{r_c})^2)^{3\beta -\alpha /2}} \frac{1}{(1 + (\frac{x}{r_s})^{\gamma})^{\frac{\epsilon}{\gamma}}}$$

        Parameters:
            r (jnp.array): Radius to compute the density function in R500 units

        Returns:
            (jnp.array): Density function evaluated at the given radius in cm$^{-6}$
        """

        n0 = hk.get_parameter("n0", [], init=Constant(jnp.exp(-4.9)))
        r_c = hk.get_parameter("r_c", [], init=Constant(jnp.exp(-2.7))) 
        

        R500 = 1309. #[kpc]
        x = r/R500
        gamma = 3
        r_s = jnp.exp(-0.51)
        alpha = 0.7
        beta = 0.39
        eps = 2.6
        
        emiss = n0**2 * (x/r_c)**-alpha / (1 + (x/r_c)**2 )**(3*beta - alpha/2) / (1 + (x/r_s)**gamma)**(eps/gamma)
        #Cut emission at 5 R500
        return jnp.clip(emiss * jnp.heaviside(5. - x, 0), 0, 0.05**2)