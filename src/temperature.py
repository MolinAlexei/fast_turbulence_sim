from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import jax.numpy as jnp
from astropy.cosmology import LambdaCDM


class GhirardiniModel(hk.Module):
    """
    Universal temperature profile as defined in Ghirardini 2018+ in the X-COP cluster sample
    """

    def __init__(self):
        super(GhirardiniModel, self).__init__()
        self.cosmo = LambdaCDM(H0 = 70, Om0 = 0.3, Ode0= 0.7)


    def __call__(self, r, z = 0.1):
        r"""
        Compute the temperature function for a given radius.
        $$\dfrac{T(x)}{T_{500}} = T_0 \dfrac{\frac{T_\mathrm{min}}{T_0} + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}}{1 + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}} \frac{1}{(1 + (\frac{x}{r_t})^2)^{\frac{c}{2}}}$$

        Parameters:
            r (jnp.array): Radius at which to compute the temperature, in kpc
            z (float): Redshift of cluster
        Returns:
            (jnp.array): Temperature function evaluated at the given radius in keV
        """


        M500 = 0.7 #(x10^15 Msun)
        Ez   = self.cosmo.efunc(z)
        h_70 = 0.7
    
        T500 = 8.85 * (M500*h_70)**(2./3.) * Ez**(2./3.)
        T0 = 1.09
        rcool = jnp.exp(-4.4)
        rt = 0.45
        TmT0 = 0.66
        acool = 1.33
        c2 = 0.3

        R500 = 1309. #[kpc]
        x = r/R500

        term1 = (TmT0 + (x / rcool) ** acool)
        term2 = (1 + (x / rcool) ** acool) * (1 + (x / rt) ** 2) ** c2

        T = T500 * T0 * term1 / term2

        #Clip the values to the [1-7] keV range (otherwise the T-Z interpolation does not work.)
        return T #jnp.clip(T, 1., 7.)
