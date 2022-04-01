import numpy as np


class Lattice():
    """"
    Class to implement the Lattice method.

    Lattice Models or Binomial Options Pricing Model (BOPM) provide a way to generate a lattice of possible future prices for a given asset utilizing the binomial tree method with a given number of steps.
    Those steps "travels" in a discrete time line, creating bifurcations in the tree, and each one of those bifurcations create two possible paths, one for the asset price going up and one for the asset price going down.
    At the end of each step, the asset price is calculated by adding the probability of the asset price going up to the probability of the asset price going down. The model adress cases where the closed-form solution is not possible, and the model is a good way to estimate the asset price.

    The first binomial options pricing model was introduced by William Sharpe in the late 1970s. It was used to model the price of a European call option. Later on, the model was formalized by J. Cox, Ross and Rubinstein in the 1980s.
    """

    @staticmethod
    def Binomial(Price, Strike, RiskFreeRate, Time, Volatility, NumSteps):
        """
        Calculates the price of a American call option
        using the binomial lattice model

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.


        K : float
            Strike price
        r : float
            Constant risk-free short rate
        T : float
            Time to maturity
        Volatility : float
            Volatility
        n : int
            Number of periods


        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Time to maturity of the option.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate.

        Returns
        -------
        Price of the Call Option: :py:class:`float`
        """
        dt = Time / NumSteps
        u = np.exp(Volatility * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(RiskFreeRate * dt) - d) / (u - d)
        disc = np.exp(-RiskFreeRate * dt)
        # Stock price tree
        S = np.zeros((NumSteps + 1, NumSteps + 1))
        S[0, :] = Price
        for i in range(1, NumSteps + 1):
            S[i, :] = S[i - 1, :] * u
        # Option value tree
        C = np.zeros((NumSteps + 1, NumSteps + 1))
        for i in range(NumSteps + 1):
            for j in range(NumSteps - i + 1):
                C[i, j] = max(S[i, j] - Strike, 0)
        for i in range(NumSteps - 1, -1, -1):
            for j in range(i + 1):
                C[i, j] = disc * (q * C[i + 1, j] + (1 - q) * C[i + 1, j + 1])
        return C[0, 0]
