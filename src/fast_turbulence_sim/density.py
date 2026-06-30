from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


class VikhlininModel(object):
    """
    Density model which use a modified Vikhlinin functional form, with gamma fixed to 3.
    See Ghirardini et al. 2019

    """
    
    def __init__(self,
                 n0 = jnp.exp(-4.9),
                 r_c = jnp.exp(-2.7),
                 R500 = 1309.,
                 gamma = 3,
                 r_s = jnp.exp(-0.51) ,
                 alpha = 0.7,
                 beta = 0.39,
                 eps = 2.6):
        """
        Parameters:
            n0 (float): Density at core of cluster
            R500 (float): Characteristic size of cluster
            r_c (float): Shape radius 1 in units of R/R500
            gamma (float): Shape factor outside of r_s
            r_s (float): Shape radius 2 in units of R/R500
            beta (float): Shape factor outside of r_c
            alpha (float): Shape factor at center of radius
            eps (float): Shape factor
        """

        self.n0 = n0
        self.r_c = r_c
        self.R500 = R500
        self.gamma = gamma
        self.r_s = r_s
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self,
                 r : jnp.array):
        r"""Compute the density function for a given radius.
        
        $n_e^2(x)= n_0^2 \frac{(\frac{x}{r_c})^{-\alpha}}{(1 + (\frac{x}{r_c})^2)^{3\beta -\alpha /2}} \frac{1}{(1 + (\frac{x}{r_s})^{\gamma})^{\frac{\epsilon}{\gamma}}}$

        Parameters:
            r (jnp.array): Radius to compute the density function in R500 units

        Returns:
            (jnp.array): Density function evaluated at the given radius in cm$^{-6}$
        """

        x = r/self.R500

        
        emiss = (self.n0**2 * (x / self.r_c)**-self.alpha
                 / (1 + (x / self.r_c)**2 )**(3 * self.beta - self.alpha / 2)
                 / (1 + (x / self.r_s)**self.gamma)**(self.eps / self.gamma))
        #Cut emission at 5 R500
        return jnp.clip(emiss * jnp.heaviside(5. - x, 0), 0, 0.05**2)