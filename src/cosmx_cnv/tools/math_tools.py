import numpy as np
from scipy.stats import gaussian_kde
from scipy.linalg import cholesky

class FixedBandwidthKDE(gaussian_kde):
    """
    A 1D Gaussian KDE that uses an absolute, fixed bandwidth (in bp).
    It overrides `_compute_covariance()` to ignore the dataset variance
    and uses only the user-specified 'bw' for smoothing.
    """
    def __init__(self, dataset, bw):
        """
        Parameters
        ----------
        dataset : array_like of shape (n,)
            1D array of data points (e.g., genomic positions).
        bw : float
            Desired fixed bandwidth (std dev of the Gaussian kernel),
            in the same units as dataset (e.g. base pairs).
        """
        # Ensure the dataset has shape (1, n) as expected by gaussian_kde
        dataset = np.atleast_2d(dataset)
        
        # Store the desired bandwidth. We set it before calling super().__init__().
        self._fixed_bw = float(bw)
        
        # Call the parent constructor with a dummy bw_method (e.g. 1.0).
        # We'll override the covariance matrix below.
        super().__init__(dataset, bw_method=1.0)

    def _compute_covariance(self):
        """
        Override the parent's method to set a 1D covariance = bw^2,
        ignoring data variance.
        """
        # In 1D, covariance is simply a 1x1 matrix = [[(fixed_bw)^2]]
        cov = self._fixed_bw**2

        # Assign self.covariance and the inverse:
        self.covariance = np.array([[cov]])
        self._inv_cov = np.array([[1.0 / cov]])
        
        # SciPy's newer versions need self.cho_cov for evaluate()
        self.cho_cov = cholesky(self.covariance, lower=True)
        
        # Normalization factor in 1D: (2π * cov)^(1/2)
        # The factor in the SciPy code is
        #     self._norm_factor = 1 / (sqrt((2π)^d * det(self.covariance)) * n)
        # with d = 1 in 1D.
        n = self.n
        d = self.d
        norm = (2 * np.pi)**(0.5 * d) * np.sqrt(np.linalg.det(self.covariance))
        self._norm_factor = 1.0 / (norm * n)

    @property
    def inv_cov(self):
        """Read-only property returning the 1D inverse covariance matrix."""
        return self._inv_cov
