import numpy as np
import pandas as pd

class DiscreteEntropy:
    """
    Class for computing the discrete entropy of probability distributions.
    """

    def __init__(self):
        """
        Initializes the DiscreteEntropy class.
        """
        print('Initialized DiscreteEntropy class')

    def _estimate_marginal_distribution(self,
                                        sample: pd.Series,
                                        bins: str='auto'):
        """
        Estimates the marginal probability mass distribution of a given sample.

        Args:
            sample (array-like): The sample data.
            bins (int or sequence of scalars, optional): The number of bins or the bin edges. Default is 'auto'.

        Returns:
            pandas.Series: The marginal probability mass distribution.
            numpy.ndarray: The histogram of the sample data.
        """
        hist, bin_edges = np.histogram(sample, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        marginal_distribution = hist / len(sample)
        marginal_series = pd.Series(marginal_distribution, index=bin_centers)
        return marginal_series, hist

    def discrete_entropy(self,
                         margSttProb: pd.Series, 
                         no_states: int = 10, 
                         ignore_na: bool = False):
        """
        Computes the discrete entropy of a given marginal probability mass distribution.

        Args:
            margSttProb (pandas.Series): The marginal probability mass distribution.
            no_states (int): The number of states.
            ignore_na (bool, optional): Whether to ignore NaN values. Default is False.

        Returns:
            float: The discrete entropy.
        """
        marginal_series, _ = self._estimate_marginal_distribution(margSttProb, bins=no_states)
        if ignore_na:
            marginal_series = marginal_series.dropna()

        h = -1 * np.sum(marginal_series * np.log2(marginal_series))
        return h
