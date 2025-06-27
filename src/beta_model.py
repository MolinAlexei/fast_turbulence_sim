from jax import config
config.update("jax_enable_x64", True)
import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax.numpy.fft as fft
from haiku.initializers import Constant


class BetaModel(hk.Module):
    """
    Beta Model
    """
    
    def __init__(self):
        super(BetaModel, self).__init__()

    def __call__(self, r):
        
        n0 = hk.get_parameter("n0", [], init=Constant(1.)) * (u.kiloparsec.to(u.cm))**3
        r_c = hk.get_parameter("r_c", [], init=Constant(150.)) #kpc
        beta = 2/3

        return n0**2 * (1+(r/r_c)**2)**(-3*beta) 
    
class XBetaModel(hk.Module):
    """
    XBeta Model
    """
    
    def __init__(self):
        super(XBetaModel, self).__init__()

    def __call__(self, r):
        
        n0 = hk.get_parameter("n0", [], init=Constant(1.)) 
        r_c = hk.get_parameter("r_c", [], init=Constant(150.)) #kpc
        beta = 2/3

        return n0**2 * (1+(r/r_c)**2)**(-3*beta) 
    
class FlatModel(hk.Module):
    """
    Flat Model
    """
    
    def __init__(self):
        super(FlatModel, self).__init__()

    def __call__(self, r):
        n0 = hk.get_parameter("n0", [], init=Constant(1.))
        r_c = hk.get_parameter("r_c", [], init=Constant(150.)) #kpc
        return jnp.ones_like(r)
    
class VikhlininModel(hk.Module):
    """
    Flat Model
    """
    
    def __init__(self):
        super(VikhlininModel, self).__init__()

    def __call__(self, r):
        n0 = hk.get_parameter("n0", [], init=Constant(jnp.exp(-4.9)))
        r_c = hk.get_parameter("r_c", [], init=Constant(jnp.exp(-2.7))) 
        
        R500 = 1308. #[kpc]
        x = r/R500
        gamma = 3
        r_s = jnp.exp(-0.51)
        alpha = 0.7
        beta = 0.39
        eps = 2.6
        
        emiss = n0**2 * (x/r_c)**-alpha / (1 + (x/r_c)**2 )**(3*beta - alpha/2) / (1 + (x/r_s)**gamma)**(eps/gamma)
        #Cut emission at 5 R500
        return jnp.clip(emiss * jnp.heaviside(5. - x, 0), 0, 0.05**2)


