import jax.numpy as jnp
import numpy as np
import pickle
from astropy.io import fits

class LoadBinning : 
    """
    Get the different arrays used for a simulation instance from a pickle 
    binning as created by xifu cluster sims. The count map is needed for returning the
    count_weighted barycentre of each bin.
    """
    
    def __init__(self,
                 shape = (360,360),
                 binning_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_region_200/region_files/19p_region_dict.p',
                 count_map_file = '/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_count_image.fits'):
        """
        Initialize binning

        Parameters:
            shape (tuple): Shape of the final map to provide (should match the shape from SpatialGrid3D)
            binning_file (str): Path to pickle file containing binning dict
            count_map_file (str): Path to count map
        """
        self.shape = shape
        self.binning_dict, self.region_image = pickle.load(open(binning_file, 'rb'), encoding="bytes")
        self.countmap = jnp.array(fits.getdata(count_map_file), dtype = 'float32')

    def __call__(self):

        """
        Create the different quantities used from the binning

        Returns:
            X_pixels (jnp.array): Array of x coordinate of each pixel on the xifusim images
            Y_pixels (jnp.array): Array of y coordinate of each pixel on the xifusim images 
            bin_num_pix (jnp.array): Array of the bin number of each pixel
            nb_bins (int): Number of bins
            xBar_bins (jnp.array): Arrays of the count-wieghted barycenters, x coordinate
            yBar_bins (jnp.array): Arrays of the count-wieghted barycenters, y coordinate 
            bin_nb_map (jnp.array):  Map of the bin numbers (mainly used as a sanity check)
        """
        
        # Initialize empty lists
        binnum = []
        X = []
        Y = []
        xBar = []
        yBar = []
        
        # Iterate on the number of bins
        for k in range(len(self.binning_dict)-1):
            
            binnum.extend([k] * len(self.binning_dict[k][0]) )
            x = self.binning_dict[k][1][0]
            y = self.binning_dict[k][1][1]
            X.extend(x)
            Y.extend(y)
            #Count weighted average for the barycenter of the bin
            xBar.append(jnp.average(jnp.array(x), weights = self.countmap[x,y]))
            yBar.append(jnp.average(jnp.array(y), weights = self.countmap[x,y]))
        
        # Arrays of x and y coordinate of each pixel on the xifusim images
        self.X_pixels = jnp.array(X)
        self.Y_pixels = jnp.array(Y)
        
        # Array of the bin number of each pixel
        self.bin_num_pix = jnp.array(binnum)
        
        # Number of bins
        self.nb_bins = len(jnp.unique(jnp.array(binnum)))
        
        # Arrays of the count-wieghted barycenters
        self.xBar_bins = jnp.array(xBar)
        self.yBar_bins = jnp.array(yBar)
        
        # Map of the bin numbers (mainly used as a sanity check)
        self.bin_nb_map = jnp.zeros(self.shape)
        self.bin_nb_map = self.bin_nb_map.at[self.X_pixels, self.Y_pixels].set(self.bin_num_pix)
        
        return self.X_pixels, self.Y_pixels, self.bin_num_pix, self.nb_bins, self.xBar_bins, self.yBar_bins, self.bin_nb_map
    
    