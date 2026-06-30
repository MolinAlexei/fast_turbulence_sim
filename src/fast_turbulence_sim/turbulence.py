from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


class KolmogorovPowerSpectrum(object):
    """
    Kolmogorov power spectrum
    """
    
    def __init__(self):
        pass

    def __call__(self,
                 k : jnp.array,
                 sigma : float,
                 log_inj : float,
                 log_dis : float,
                 alpha : float,
                 k_max : float):
        r"""Kolmogorov power spectrum

        $\mathcal{P}_{3D}(k)= \sigma^2 \frac{e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} }{\int 4\pi k^2  \, e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} \text{d} k}$

        Parameters:
            k (jnp.array): Array of wavenumbers at which to evaluate the power spectrum
            sigma (float): Norm of the power spectrum
            log_inj (float): Injection scale of the power spectrum
            log_dis (float): Dissipation scale of the power spectrum
            alpha (float): Slope of the power spectrum
            k_max (float): Maximum wavenumber probed by spatial grid at which the spatial grid is evaluated
        """

        k_inj = 10 ** (-log_inj)
        k_dis = 10 ** (-log_dis)

        k_int = jnp.geomspace(k_inj/20, k_dis*20, 1000)

        norm = jnp.trapezoid(4*jnp.pi*k_int**3
                                    * jnp.exp(-(k_inj / k_int) ** 2)
                                    * jnp.exp(-(k_int/ k_dis) ** 2)
                                    * (k_int) ** (-alpha),
                                   x=jnp.log(k_int)
                                   )
        res = jnp.where(k > 0,
                        jnp.exp(-(k_inj / k) ** 2)
                        * jnp.exp(-(k/ k_dis) ** 2)
                        * (k) ** (-alpha),
                        0.)

        # Compute the correction factor
        k_int = jnp.geomspace(k_max, k_dis * 20, 1000)
        factor = jnp.trapezoid(
            4 * jnp.pi * k_int ** 3 * jnp.exp(-(k_inj / k_int) ** 2) * jnp.exp(-(k_int / k_dis) ** 2) * k_int ** (
                -alpha) / norm,
            x=jnp.log(k_int)
        ) * sigma ** 2

        return sigma**2 * res / norm, factor