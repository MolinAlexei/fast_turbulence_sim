from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.cosmology import LambdaCDM


class GhirardiniModel(object):
    """
    Universal temperature profile as defined in Ghirardini 2018+ in the X-COP cluster sample
    """

    def __init__(self,
                 M500 = 0.7,
                 h_70 = 0.7,
                 T0 = 1.09,
                 rcool = jnp.exp(-4.4),
                 rt = 0.45,
                 TmT0 = 0.66,
                 acool = 1.33,
                 c2 = 0.3,
                 R500 = 1309.,  # [kpc]
                 z = 0.1):
        """
        Parameters:
            M500 (jnp.array): M500 of the cluster
            h_70 (jnp.array): h_70 of the chosen cosmology
            T0 (float): Normalization factor
            rcool (float): Shape radius 1 in units of R/R500
            rt (float): Shape radius 2 in units of R/R500
            acool (float): Shape parameter 1
            c2 (float): Shape parameter 2
            z (float): Redshift of the cluster

        """

        self.M500 = M500
        self.h_70 = h_70
        self.T0 = T0
        self.rcool = rcool
        self.rt = rt
        self.TmT0 = TmT0
        self.acool = acool
        self.c2 = c2
        self.R500 = R500


        self.cosmo = LambdaCDM(H0 = 70, Om0 = 0.3, Ode0= 0.7)
        self.Ez = self.cosmo.efunc(z)
        self.T500 = (8.85
                     * (M500 * h_70) ** (2. / 3.)
                     * self.Ez ** (2. / 3.))

    def __call__(self, r):
        r"""Compute the temperature function for a given radius.

        $\dfrac{T(x)}{T_{500}} = T_0 \dfrac{\frac{T_\mathrm{min}}{T_0} + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}}{1 + (\frac{x}{r_\mathrm{cool}})^{a_\mathrm{cool}}} \frac{1}{(1 + (\frac{x}{r_t})^2)^{\frac{c}{2}}}$

        Parameters:
            r (jnp.array): Radius at which to compute the temperature, in kpc
        Returns:
            (jnp.array): Temperature function evaluated at the given radius in keV
        """

        x = r / self.R500

        term1 = (self.TmT0 + (x / self.rcool) ** self.acool)
        term2 = (1 + (x / self.rcool) ** self.acool) * (1 + (x / self.rt) ** 2) ** self.c2

        T = self.T500 * self.T0 * term1 / term2

        #Clip the values to the [1-7] keV range (otherwise the T-Z interpolation does not work.)
        return T #jnp.clip(T, 1., 7.)
