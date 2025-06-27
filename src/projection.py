import jax.numpy as jnp


class VCubeProjection :
    """
    Projection of velocity cube with emission weighting in binned map
    """
    def __init__(self, binning, em_cube):
        
        self.binning = binning
        self.em = em_cube
        
    def __call__(self,v):
        
        """
        Uses counts to weight and project the velocity cube
        into a centroid shift map and broadening map.
        
        Parameters
        ----------
        v : Array
            Velocity cube
        Returns
        -------
        v_map : Array
            Binned image of the emission weighted centroid shift
        std_map : Array
            Binned image of the emission weighted broadening
        binned_v_weighted : Array
            Vector of emission weighted centroid shift in each bin
        binned_std_weighted : Array
            Vector of emission weighted broadening in each bin
            
        """
        
        # This is a fully jaxified version
    
        # Value of the speed grouped by pixel
        v_vec = v[self.binning.X_pixels,
                  self.binning.Y_pixels, 
                  :]

        # Value of the emission grouped by pixel
        count_vec = self.em[self.binning.X_pixels,
                       self.binning.Y_pixels, 
                       :]

        # Indices of in which bin goes each pixel
        bins_unique, inverse_indices = jnp.unique(self.binning.bin_num_pix, 
                                                  return_inverse=True, 
                                                  size = self.binning.nb_bins)

        # Binned vectors
        bin_v = jnp.zeros(len(bins_unique))
        bin_counts = jnp.zeros(len(bins_unique))
        bin_std = jnp.zeros(len(bins_unique))

        # We add to each bin the weighted sum of all pixels belonging to it
        bin_v = bin_v.at[inverse_indices].add(jnp.sum(v_vec*count_vec, axis = -1))
        bin_counts = bin_counts.at[inverse_indices].add(jnp.sum(count_vec, axis = -1))
        bin_std = bin_std.at[inverse_indices].add(jnp.sum(v_vec**2 *count_vec, axis = -1))


        # Divide by weighing (i.e. summed emission)
        binned_v_weighted = bin_v / bin_counts
        binned_std_weighted = jnp.sqrt((bin_std/bin_counts - binned_v_weighted **2))

        # Create maps
        v_map = jnp.zeros(self.binning.shape)
        v_map = v_map.at[self.binning.X_pixels, self.binning.Y_pixels].set(binned_v_weighted[inverse_indices])

        std_map = jnp.zeros(self.binning.shape)
        std_map = std_map.at[self.binning.X_pixels, self.binning.Y_pixels].set(binned_std_weighted[inverse_indices])

        # Return weighted and binned values
        return v_map, std_map, binned_v_weighted, binned_std_weighted