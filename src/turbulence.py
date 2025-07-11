from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import jax.numpy as jnp
from haiku.initializers import Constant
from jax.scipy import integrate

class KolmogorovPowerSpectrum(hk.Module):
    """
    Kolmogorov power spectrum
    """
    
    def __init__(self):
        super(KolmogorovPowerSpectrum, self).__init__()

    def __call__(self, k):
        r"""Kolmogorov power spectrum

        $$\mathcal{P}_{3D}(k)= \sigma^2 \frac{e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} }{\int 4\pi k^2  \, e^{-\left(k/k_{\text{inj}}\right)^2} e^{-\left(k_{\text{dis}}/k\right)^2} k^{-\alpha} \text{d} k}$$
        
        """
        
        sigma = hk.get_parameter("sigma", [], init=Constant(250)) #km/s
        log_inj = hk.get_parameter("log_inj", [], init=Constant(jnp.log10(300))) #kpc
        log_dis = jnp.log10(10) #kpc
        alpha = hk.get_parameter("alpha", [], init=Constant(11/3))

        k_inj = 10 ** (-log_inj)
        k_dis = 10 ** (-log_dis)
        
        #sigma = 10**log_sigma
        
        k_int = jnp.geomspace(k_inj/20, k_dis*20, 1000)
        norm = integrate.trapezoid(4*jnp.pi*k_int**3*jnp.exp(-(k_inj / k_int) ** 2) * jnp.exp(-(k_int/ k_dis) ** 2) * (k_int) ** (-alpha), x=jnp.log(k_int))
        res = jnp.where(k > 0, jnp.exp(-(k_inj / k) ** 2) * jnp.exp(-(k/ k_dis) ** 2) * (k) ** (-alpha), 0.)

        return sigma**2 * res / norm