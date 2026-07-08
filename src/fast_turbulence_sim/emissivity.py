from jax import config
config.update("jax_enable_x64", True)
from .density import VikhlininModel
from .temperature import GhirardiniModel
from .abundance import XCOPAbundance
from .cooling import APECEmissivity
from astropy.cosmology import LambdaCDM
import astropy.units as u
import jax.numpy as jnp

class XrayEmissivity(object):
    """
    3D Xray emissivity build with temperature, cooling function, density model.
    It depends on the redshift of the cluster, since the cooling function is precomputed using XSPEC.
    The default models are the ones used in the papers i.e. Vikhlinin for density, Ghirardini for temperature
    and the interpolated cooling function.
    """
    def __init__(self,
                 TZ_grid_to_interp_from = '/xifu/home/mola/SBI_Turbulence/data/flux_table_APEC_oldXIFU.npy',
                 R500 : float = 1309.,
                 M500 : float = 0.7,
                 z : float = 0.1,):
        """
        Parameters:
            TZ_grid_to_interp_from (string): path to interpolated table
            R500 (float): R500 radius of the cluster
            M500 (float): M500 of the cluster
            z (float): redshift of the cluster
        """

        self.squared_density = VikhlininModel(R500=R500)
        self.temperature = GhirardiniModel(M500=M500, R500=R500, z = z)
        self.abundance = XCOPAbundance(R500=R500)
        self.cooling_function = APECEmissivity(TZ_grid_to_interp_from = TZ_grid_to_interp_from)
        self.cosmo = LambdaCDM(H0 = 70, Om0 = 0.3, Ode0= 0.7)
        self.z = z


    def __call__(self, r,
                exposure = 125e3, 
                pixsize_cm = 1.,
                filling_factor= (310.7/317)**2):
        """
        Compute the emissivity at a given radius, including $N_H$ absorption.

        Parameters:
            r (jnp.array): radius in units of $R_{500}$
            pixsize_cm (float) : Projected of the pixel in the cluster, in centimeters
            filling_factor (float) : Filling factor of the array (absorber size / pixel pitch)
        """

        angular_distance = self.cosmo.angular_diameter_distance(self.z).value * u.megaparsec.to(u.cm) #cm
        norm = self.squared_density(r) * 1e-14 / (4 *jnp.pi* (angular_distance*(1+self.z))**2 ) / 1.2
        
        factor = filling_factor * exposure * pixsize_cm**3

        return self.cooling_function(self.abundance(r), self.temperature(r)) * norm * factor
