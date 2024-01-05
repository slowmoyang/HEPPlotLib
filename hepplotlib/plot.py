from dataclasses import dataclass
import operator
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from hist import Hist
from hist.intervals import clopper_pearson_interval


@dataclass
class StatError:
    x: NDArray
    xerr: NDArray # bin half width
    y: NDArray
    yerr: NDArray

    @property
    def ylow(self) -> NDArray:
        return self.y - self.yerr

    @property
    def yup(self) -> NDArray:
        return self.y + self.yerr

    @property
    def edges(self) -> NDArray:
        zipped = zip(self.x, self.xerr)
        op_list = [operator.sub, operator.add]
        return np.array([op(x, xerr) for x, xerr in zipped for op in op_list])

    @classmethod
    def from_hist(cls, hist: Hist) -> 'StatError':
        x_axis = hist.axes[0]
        x = x_axis.centers
        xerr = x_axis.widths / 2
        y = hist.values()
        yerr = np.sqrt(hist.variances())
        return cls(x=x, xerr=xerr, y=y, yerr=yerr)

    def plot(self,
             ax: plt.Axes | None = None,
             hatch='////',
             facecolor='none',
             lw=0,
             binwnorm: bool = False,
             **kwargs
    ):
        ylow = np.repeat(self.ylow, 2)
        yup = np.repeat(self.yup, 2)

        if binwnorm:
            bin_widths = np.repeat(2 * self.xerr, 2)
            ylow /= bin_widths
            yup /= bin_widths

        ax = ax or plt.gca()
        return ax.fill_between(x=self.edges, y1=ylow, y2=yup,
                               hatch=hatch, facecolor=facecolor, lw=lw, **kwargs)


@dataclass
class Ratio1D:
    x: NDArray
    y: NDArray
    xerr: NDArray
    yerr: NDArray

    @classmethod
    def from_hist(cls, data_hist: Hist, mc_hist: Hist) -> 'Ratio1D':
        # TODO check if data_hist and mc_hist share x-axis
        x_axis = mc_hist.axes[0]
        x = x_axis.centers
        y = data_hist.values() / mc_hist.values()
        yerr = np.sqrt(mc_hist.variances()) / mc_hist.values()
        xerr = x_axis.widths / 2
        return cls(x=x, y=y, xerr=xerr, yerr=yerr)

    def plot(self,
            ax: plt.Axes | None = None,
            ls='',
            marker='s',
            color='black',
            **kwargs
    ):
        ax = ax or plt.gca()
        return ax.errorbar(x=self.x, y=self.y, yerr=self.yerr, xerr=self.xerr,
                           ls=ls, marker=marker, color=color, **kwargs)


@dataclass
class RatioStatError1D:
    edges: NDArray
    yerr: NDArray

    @property
    def ylow(self) -> NDArray:
        return 1 - self.yerr

    @property
    def yup(self) -> NDArray:
        return 1 + self.yerr

    @classmethod
    def from_hist(cls, hist: Hist) -> 'RatioStatError1D':
        edges = hist.axes[0].edges
        yerr = np.sqrt(hist.variances()) / hist.values()
        return cls(edges=edges, yerr=yerr)

    def fill_between(self,
                     ax: plt.Axes | None = None,
                     alpha: float = 0.2,
                     color: str = 'gray',
                     **kwargs
    ):
        ax = ax or plt.gca()

        x = np.repeat(self.edges, 2)[1:-1]
        ylow = np.repeat(self.ylow, 2)
        yup = np.repeat(self.yup, 2)

        return ax.fill_between(x=x,
                               y1=ylow,
                               y2=yup,
                               alpha=alpha,
                               color=color,
                               **kwargs)


@dataclass
class Efficiency1D:
    x: NDArray
    y: NDArray
    ylow: NDArray
    yup: NDArray
    xerr: NDArray

    @classmethod
    def from_hist(cls, h_num, h_den) -> 'Efficiency1D':
        num = h_num.values()
        den = h_den.values()

        x = h_den.axes[0].centers
        y = np.divide(num, den, where=den > 0)
        ylow, yup = clopper_pearson_interval(num, den)

        xerr = h_den.axes[0].widths
        return cls(x, y, ylow, yup, xerr=xerr)

    @property
    def yerr_low(self) -> NDArray:
        return self.y - self.ylow

    @property
    def yerr_up(self) -> NDArray:
        return self.yup - self.y

    @property
    def yerr(self) -> tuple[NDArray, NDArray]:
        return (self.yerr_low, self.yerr_up)

    def plot(self,
            ax: plt.Axes | None = None,
            xerr: bool = False,
            **kwargs):
        ax = ax or plt.gca()

        xerr_arr = self.xerr if xerr else None

        return ax.errorbar(self.x, self.y, self.yerr, xerr=xerr_arr, **kwargs)
