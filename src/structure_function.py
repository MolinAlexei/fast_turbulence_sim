import numpy as np 
from itertools import combinations
import scipy.stats as stats
import jax.numpy as jnp

import time

class StructureFunction:

    def __init__(self, bins = np.geomspace(1,40,15)):

        self.bins = bins

    def __call__(self, v_bin_vec, xbary, ybary):
        """
        Computes the 2nd order structure function of an image with arbitrary binning.
        
        Parameters
        ----------
        v_bin_vec : array
            Array of the count weighted velocity in each bin
        xbary : array
            Array of the bin barycenters X coordinate
        ybary : array
            Array of the bin barycenters Y coordinate
        bins : array, optional
            Array of the binning chosen for the SF

        Returns
        ------- 
        bin_dists : array
            Separations of the SF
        bin_means : array 
            Values of the SF
        """
        
        #Indexes of all possible combinations
        idx = jnp.triu_indices(len(xbary), k = 1)
        
        #Vector of all possible separations
        sep_vec = jnp.hypot(xbary[idx[0]] - xbary[idx[1]],
                               ybary[idx[0]] - ybary[idx[1]])
        
        #Vector of (v_i - v_j)^2 
        v_vec = (v_bin_vec[idx[0]] - v_bin_vec[idx[1]])**2
        
        ### PREVIOUS VERSION
        #Now compute binned statistics
        #Average SF in each bin
        #bin_means, bin_edges, binnumber = stats.binned_statistic(sep_vec, v_vec, 'mean', bins = self.bins)
        #Average distance in each bin
        #bin_dists, bin_edges, binnumber = stats.binned_statistic(sep_vec, sep_vec, 'mean', bins = self.bins)

        ### JAX VERSION
        hist_weighted, bin_edges = jnp.histogram(sep_vec, bins = self.bins, weights = v_vec)
        hist_numbers, bin_edges = jnp.histogram(sep_vec, bins = self.bins)
        hist_dists, _ = jnp.histogram(sep_vec, bins = self.bins, weights = sep_vec)

        #Average SF in each bin
        bin_means = hist_weighted/hist_numbers

        #Average distance in each bin
        hist_dists = hist_dists/hist_numbers

        
        #Note : in principle, we shouldn't be taking the average distance in each bin, we should
        #take the bin center, but that's for comparison purposes with Edo's code
        
        return hist_dists, bin_means
