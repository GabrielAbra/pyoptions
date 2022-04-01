import numpy as np
from scipy import stats, optimize


class BlackScholes():

    @staticmethod
    def _d1(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02):
        return (np.log(Price / Strike) + (RiskFreeRate + Volatility ** 2 / 2) * Time / 365) / (Volatility * np.sqrt(Time / 365))

    @staticmethod
    def _d2(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02):
        return BlackScholes._d1(Price, Strike, Volatility, Time, RiskFreeRate) - Volatility * np.sqrt(Time / 365)

    @staticmethod
    def CallOptionPrice(Price: float, Strike: float, Volatility: float, Time: int, RiskFreeRate: float = 0.02) -> float:
        """
        Calculate the price of a american call option

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        Return
        -------
        CallOptionPrice : :py:class:`float`
        """
        d1 = BlackScholes._d1(Price, Strike, Volatility, Time, RiskFreeRate)
        d2 = BlackScholes._d2(Price, Strike, Volatility, Time, RiskFreeRate)
        return Price * stats.norm.cdf(d1) - Strike * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(d2)

    @staticmethod
    def PutOptionPrice(Price: float, Strike: float, Volatility: float, Time: int, RiskFreeRate: float = 0.02) -> float:
        """
        Calculate the price of a american put option

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        Return
        -------
        PutOptionPrice : :py:class:`float`
        """
        if Time == 0:  # if time is 0, the option is already expired
            return 0

        d1 = BlackScholes._d1(Price, Strike, Volatility, Time, RiskFreeRate)
        d2 = BlackScholes._d2(Price, Strike, Volatility, Time, RiskFreeRate)
        return Strike * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(-d2) - Price * stats.norm.cdf(-d1)

    @staticmethod
    def ImpliedVolatility(Price, Strike, Time, Premium, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the implied volatility of a american option

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Time : :py:class:`int` Days till option expiration.

        Premium : :py:class:`float` Premium of the option. 

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Volatility : :py:class:`float`
        """
        if OptionType == 'call':
            func = BlackScholes.CallOptionPrice
        elif OptionType == 'put':
            func = BlackScholes.PutOptionPrice
        else:
            raise ValueError('OptionType must be "call" or "put"')
        return optimize.newton(lambda x: func(Price, Strike, x, Time, RiskFreeRate) - Premium, 0.1)

    @staticmethod
    def Delta(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Delta Greek of an American option. Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price.

        Delta is also know as the first derivative of the value of the option with respect to the underlying instrument's price.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Delta : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return stats.norm.cdf(d1)

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return stats.norm.cdf(d1) - 1

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Gamma(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Gamma Greek of an American option. Measures the rate of change of the theoretical option value with respect to changes in the underlying asset's volatility.

        Gamma is also know as the second derivative of the value of the option with respect to the underlying instrument's volatility.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Gamma : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return stats.norm.pdf(d1) / (Price * Volatility * np.sqrt(Time))

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return stats.norm.pdf(d1) / (Price * Volatility * np.sqrt(Time))

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Theta(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Theta Greek of an American option. Measures the rate of change of the theoretical option value with respect to time.

        Theta is also know as the first derivative of the value of the option with respect to time.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Theta : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            d2 = BlackScholes._d2(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return -Price * stats.norm.pdf(d1) * Volatility / (2 * np.sqrt(Time)) - RiskFreeRate * Strike * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(d2)

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            d2 = BlackScholes._d2(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return -Price * stats.norm.pdf(d1) * Volatility / (2 * np.sqrt(Time)) + RiskFreeRate * Strike * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(-d2)

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Vega(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Vega Greek of an American option. Measurement of the sensitivity of the value of the option to changes in the volatility of the underlying asset.

        Vega is also know as the first derivative of the value of the option with respect to the underlying instrument's volatility.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Vega : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return Price * stats.norm.pdf(d1) * np.sqrt(Time)

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return Price * stats.norm.pdf(d1) * np.sqrt(Time)

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Rho(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Rhoe Greek of an American option. Measures the rate of change of the option value with respect to changes in the risk-free rate.

        Rho is also know as the first derivative of the value of the option with respect to the risk-free rate.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Rhox : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return Strike * Time * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(d1)

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return -Strike * Time * np.exp(-RiskFreeRate * Time) * stats.norm.cdf(-d1)

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Lambda(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call') -> float:
        """
        Calculate the Lambda Greek (Elasticity) of an American option. Measures the rate of change of the option value with respect to changes in the premium.

        Lambda is also know as the first derivative of the value of the option with respect to the premium.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Lambda : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            value = BlackScholes.CallOptionPrice(
                Price, Strike, Volatility, Time, RiskFreeRate)
            delta = BlackScholes.Delta(
                Price, Strike, Volatility, Time, RiskFreeRate, OptionType)
            return delta * Price / value

        elif OptionType.lower() == 'put':
            value = BlackScholes.PutOptionPrice(
                Price, Strike, Volatility, Time, RiskFreeRate)
            delta = BlackScholes.Delta(
                Price, Strike, Volatility, Time, RiskFreeRate, OptionType)
            return delta * Price / value

        else:
            raise ValueError('OptionType must be "call" or "put"')

    @staticmethod
    def Vanna(Price, Strike, Volatility, Time, RiskFreeRate: float = 0.02, OptionType: str = 'call'):
        """
        Calculate the Vanna Greek of an American option. Measures the sensitivity of the premium to changes in the volatility of the underlying asset.

        Vanna is also know as the first derivative of the premium with respect to the underlying instrument's volatility.

        Parameters
        ----------
        Price : :py:class:`float` Actual asset price.

        Strike : :py:class:`float` Option strike price.

        Volatility : :py:class:`float` Annualized volatility of the asset.

        Time : :py:class:`int` Days till option expiration.

        RiskFreeRate : :py:class:`float` Annualized risk-free rate. (default: 0.02)

        OptionType : :py:class:`str` Type of the option.
            - 'call' for call options (default)
            - 'put' for put options

        Return
        -------
        Vanna : :py:class:`float`
        """
        if OptionType.lower() == 'call':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            d2 = BlackScholes._d2(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return np.exp(-RiskFreeRate * Time / 365) * np.sqrt(Time / 365) * (d2 / Volatility) * np.exp(-(d1 ** 2) / 2) / (2 * np.pi)

        elif OptionType.lower() == 'put':
            d1 = BlackScholes._d1(
                Price, Strike, Volatility, Time, RiskFreeRate)
            d2 = BlackScholes._d2(
                Price, Strike, Volatility, Time, RiskFreeRate)
            return (np.exp(-RiskFreeRate * Time / 365) * np.sqrt(Time / 365) * (d2 / Volatility) * np.exp(-(d1 ** 2) / 2) / (2 * np.pi)) * - 1

        else:
            raise ValueError('OptionType must be "call" or "put"')
