import numpy as np
from scipy.integrate import quad


class DifferentialEntropy:
    def __init__(self):
        """
        Initializes the DifferentialEntropy class.
        """
        print('Initialized DifferentialEntropy class')

    def entropy_function_gaussian(self, x: float, mu: float, sigma: float) -> float:
        """
        Computes the differential entropy for a Gaussian distribution at point x.

        Args:
            x (float): The value at which to compute the entropy.
            mu (float): The mean of the Gaussian distribution.
            sigma (float): The standard deviation of the Gaussian distribution.

        Returns:
            float: The differential entropy at point x.
        """
        tmp_entropy = (-1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))) * \
                      np.log2((1 / np.sqrt(2 * np.pi * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))))
        return tmp_entropy

    def entropy_function_uniform(self, x, a, b) -> float:
        """
        Computes the differential entropy for a uniform distribution at point x.

        Args:
            x: The value at which to compute the entropy.
            a: The lower bound of the uniform distribution.
            b: The upper bound of the uniform distribution.

        Returns:
            float: The differential entropy at point x.
        """
        tmp_entropy = (-(1 / (b - a)) * (x ** 0)) * np.log2(((1 / (b - a)) * (x ** 0)))
        return tmp_entropy

    def integration(self, distribution_type: str, params: dict) -> float:
        """
        Performs integration to compute the differential entropy.

        Args:
            distribution_type (str): The type of distribution ('uniform' or 'gaussian').
            params (dict): A dictionary containing parameters needed for the integration.

        Returns:
            float: The differential entropy.
        """
        a = params['lowBound']
        b = params['uppBound']

        if distribution_type == 'uniform':
            diff_entropy, _ = quad(self.entropy_function_uniform, a, b, args=(a, b))
        else:
            sigma = params['sigma']
            mu = params['mu']
            diff_entropy, _ = quad(self.entropy_function_gaussian, a, b, args=(mu, sigma))

        return diff_entropy