import jax.numpy as jnp
import numpy as np
import pickle
from astropy.io import fits
import pandas as pd

class LoadBinning:
    """
    Get the different arrays used for a simulation instance from a pickle 
    binning as created by xifusim. The count map is needed for returning the
    count_weighted barycentre of each bin.
    """

    def __init__(self,
                 binning_file='/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_region_200/region_files/19p_region_dict.p',
                 count_map_file='/xifu/home/mola/Turbu_300kpc_mosaics/repeat10_125ks/19p_count_image.fits'):
        """
        Initialize the loading

        Parameters:
            shape (tuple): Shape of the full image of the cluster that is used for the binning
            binning_file (str): Path to the pickle file
            count_map_file (str): Path to the count map used for binning
        """

        self.binning_dict, self.region_image = pickle.load(open(binning_file, 'rb'), encoding="bytes")
        self.countmap = np.array(fits.getdata(count_map_file))
        self.shape = self.countmap.shape

    def __call__(self):
        """
        Loads the binning

        Returns:
            X_pixels (jnp.array): Array of x coordinate of each pixel on the xifusim images
            Y_pixels (jnp.array): Array of y coordinate of each pixel on the xifusim images
            bin_num_pix (jnp.array): Array of the bin number of each pixel
            nb_bins (int): Number of bins
            xBar_bins (jnp.array): Arrays of the count-wieghted barycenters, x coordinate
            yBar_bins (jnp.array): Arrays of the count-wieghted barycenters, y coordinate
            bin_nb_map (jnp.array):  Map of the bin numbers (mainly used as a sanity check)
        """

        # There are a few strange manipulations here but they are worth it,
        # as it is much faster than the previous implementation

        # Convert to pandas dataframe
        df = pd.DataFrame.from_dict(self.binning_dict).drop(-1, axis=1)

        # Arrays of x and y coordinate of each pixel on the xifusim images
        self.X_pixels, self.Y_pixels = np.hstack(df.iloc[1])

        # Number of bins
        self.nb_bins = len(self.binning_dict) - 1

        # Array of the bin number of each pixel
        self.bin_num_pix = jnp.repeat(
            np.arange(self.nb_bins),
            np.array(df.map(len).loc[0])
        )

        # Arrays of the count-weighted barycenters
        self.xBar_bins = jnp.array(
            df.drop([0]).map(lambda x: np.average(x[0],
                                                  weights=self.countmap[x])
                             )
        )[0]
        self.yBar_bins = jnp.array(
            df.drop([0]).map(lambda x: np.average(x[1],
                                                  weights=self.countmap[x])
                             )
        )[0]

        # Map of the bin numbers (mainly used as a sanity check)
        self.bin_nb_map = jnp.ones(self.shape) * -1
        self.bin_nb_map = self.bin_nb_map.at[self.X_pixels, self.Y_pixels].set(self.bin_num_pix)

        return self.X_pixels, self.Y_pixels, self.bin_num_pix, self.nb_bins, self.xBar_bins, self.yBar_bins, self.bin_nb_map
    
    