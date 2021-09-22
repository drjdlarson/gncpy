"""Defines utility functions for plotting routines."""
import numpy as np
from numpy.linalg import eigh
import numpy.random as rnd
import matplotlib.pyplot as plt


def calc_error_ellipse(cov, n_sig):
    """Calculates parameters for an error ellipse.

    This calucates the error ellipse for a given sigma
    number according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`.

    Parameters
    ----------
    cov : 2 x 2 numpy array
        covariance matrix.
    n_sig : float
        Sigma number, must be positive.

    Returns
    -------
    width : float
        The width of the ellipse
    height :float
        The height of the ellipse
    angle : float
        The rotation angle in degrees of the semi-major axis. Measured up from
        the positive x-axis.
    """
    # get and sort eigne values
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # find rotation angle from positive x-axis, and width/height
    angle = 180 / np.pi * np.arctan2(*vecs[:, 0][::-1])
    width, height = 2 * n_sig * np.sqrt(vals)

    return 2 * width, 2 * height, angle


def init_plotting_opts(f_hndl=None, lgnd_loc=None, sig_bnd=1, time_vec=None,
                       true_states=None, rng=rnd.default_rng(1), meas_inds=None,
                       marker='o', ttl_fontsize=12, ttl_fontstyle='normal',
                       ttl_fontfamily='sans-serif', ax_fontsize=10,
                       ax_fontstyle='normal', ax_fontfamily='sans-serif'):
    """Processes common plotting options in a common interface.

    Parameters
    ----------
    f_hndl : matplotlib figure, optional
        Current to figure to plot on. Pass None to create a new figure. The
        default is None.
    lgnd_loc : string, optional
        Location of the legend. Set to none to skip creating a legend. The
        default is None.
    sig_bnd : int, optional
        If set and the covariances are saved, the sigma bounds are scaled by
        this number and plotted for each track. The default is 1.
    time_vec : list, optional
        List of time values. The default is None.
    true_states : list, optional
        Each element is a N x 1 numpy array  representing the true state.
        If not given true states are not plotted. The default is None.
    rng : numpy random generator, optional
        For generating random numbers. The default is rnd.default_rng(1).
    meas_inds : list, optional
        List of indices in the measurement vector to plot if this is specified
        all available measurements will be plotted. Note, x-axis is first, then
        y-axis. Also note, if gating is on then gated measurements will not be
        plotted. The default is None.
    marker : string, optional
        Shape to use as a marker, can be any valid value used by matplotlib.
        The default is 'o'.
    ttl_fontsize : int, optional
        Title font size. The default is 12.
    ttl_fontstyle : string, optional
        Matplotlib font style for the title. The default is 'normal'.
    ttl_fontfamily : string, optional
        Matplotlib font family for the title. The default is 'sans-serif'.
    ax_fontsize : int, optional
        Axis label font size. The default is 10.
    ax_fontstyle : string, optional
        Matplotlib font style for the axis label. The default is 'normal'.
    ax_fontfamily : string, optional
        Matplotlib font family for the axis label. The default is 'sans-serif'.

    Returns
    -------
    opts : dict
        Plotting options with default values where custom ones were not specified.
    """
    opts = {}

    opts['f_hndl'] = f_hndl
    opts['lgnd_loc'] = lgnd_loc

    opts['sig_bnd'] = sig_bnd
    opts['time_vec'] = time_vec
    opts['true_states'] = true_states
    opts['rng'] = rng
    opts['meas_inds'] = meas_inds

    opts['marker'] = marker

    opts['ttl_fontsize'] = ttl_fontsize
    opts['ttl_fontstyle'] = ttl_fontstyle
    opts['ttl_fontfamily'] = ttl_fontfamily

    opts['ax_fontsize'] = ax_fontsize
    opts['ax_fontstyle'] = ax_fontstyle
    opts['ax_fontfamily'] = ax_fontfamily

    return opts


def set_title_label(fig, ax_num, opts, ttl=None, x_lbl=None, y_lbl=None):
    """Sets the figure/window title, and axis labels with the given options.

    Parameters
    ----------
    fig : matplot figure object
        Current figure to edit.
    ax_num : int
        Index into the axes object to modify.
    opts : dict
        Standard dictionary from :func:`.plotting.init_plotting_opts`.
    ttl : string, optional
        Title string to use. This is set to the proper size, family, and
        style. It also becomes the window title. The default is None.
    x_lbl : string, optional
        Label for the x-axis. This is set to the proper size, family, and
        style. The default is None.
    y_lbl : string, optional
        Label for the y-axis. This is set to the proper size, family, and
        style. The default is None.

    Returns
    -------
    None.
    """
    if ttl is not None:
        fig.suptitle(ttl, fontsize=opts['ttl_fontsize'],
                     fontstyle=opts['ttl_fontstyle'],
                     fontfamily=opts['ttl_fontfamily'])
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(ttl)

    if x_lbl is not None:
        fig.axes[ax_num].set_xlabel(x_lbl, fontsize=opts['ax_fontsize'],
                                    fontstyle=opts['ax_fontstyle'],
                                    fontfamily=opts['ax_fontfamily'])
    if y_lbl is not None:
        fig.axes[ax_num].set_ylabel(y_lbl, fontsize=opts['ax_fontsize'],
                                    fontstyle=opts['ax_fontstyle'],
                                    fontfamily=opts['ax_fontfamily'])


def get_cmap(n, name='Dark2'):
    """Returns a function that generates a color map.

    Returns a function thata maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    Parameters
    ----------
    n : int
        Number of colors in the map.
    name : string
        name of the colormap, valid for `pyplot.cm.get_cmap` function
    """
    return plt.cm.get_cmap(name, n)
