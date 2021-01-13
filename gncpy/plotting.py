""" Defines utility functions for plotting routines.
"""
import numpy as np
from numpy.linalg import eigh
import numpy.random as rnd


def calc_error_ellipse(cov, n_sig):
    """ Calculates parameters for an error ellipse.

    This calucates the error ellipse for a given sigma
    number according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`.

    Args:
        cov (2 x 2 numpy array): covariance matrix.
        n_sig (float): Sigma number, must be positive.

    Returns:
        tuple containing

                - width (float): The width of the ellipse
                - height (float): The height of the ellipse
                - angle (float): The rotation angle in degrees
                of the semi-major axis. Measured up from the
                positive x-axis.
    """
    # get and sort eigne values
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # find rotation angle from positive x-axis, and width/height
    angle = 180 / np.pi * np.arctan2(*vecs[:, 0][::-1])
    width, height = 2 * n_sig * np.sqrt(vals)

    return 2*width, 2*height, angle


def init_plotting_opts(**kwargs):
    """ Processes common plotting options in a common interface.

    Keyword Args:
            f_hndl (Matplotlib figure): Current to figure to plot on. Always
                plots on axes[0], pass None to create a new figure
            lgnd_loc (string): Location of the legend. Set to none to skip
                creating a legend.
            sig_bnd (int): If set and the covariances are saved, the sigma
                bounds are scaled by this number and plotted for each track
            time_vec (list): list of time values
            true_states (list): list where each element is a list of numpy
                N x 1 arrays of each true state. If not given true states
                are not plotted.
            rng (Generator): A numpy random generator, leave as None for
                default.
            meas_inds (list): List of indices in the measurement vector to plot
                if this is specified all available measurements will be
                plotted. Note, x-axis is first, then y-axis. Also note, if
                gating is on then gated measurements will not be plotted.
            marker (string): Shape to use as a marker, can be any valid value
                used by matplotlib
            ttl_fontsize (int): Title font size
            ttl_fontstyle (string): Matplotlib font style for the title
            ttl_fontfamily (string): Matplotlib font family for the title
            ax_fontsize (int): Axis label font size
            ax_fontstyle (string): Matplotlib font style for the axis label
            ax_fontfamily (string): Matplotlib font family for the axis label


        Returns:
            (dict): Dictionary with standard keys for common values
    """
    opts = {}

    opts['f_hndl'] = kwargs.get('f_hndl', None)
    opts['lgnd_loc'] = kwargs.get('lgnd_loc', None)

    opts['sig_bnd'] = kwargs.get('sig_bnd', 1)
    opts['time_vec'] = kwargs.get('time_vec', None)
    opts['true_states'] = kwargs.get('true_states', None)
    opts['rng'] = kwargs.get('rng', rnd.default_rng(1))
    opts['meas_inds'] = kwargs.get('meas_inds', None)

    opts['marker'] = kwargs.get('marker', 'o')

    opts['ttl_fontsize'] = kwargs.get('ttl_fontsize', 12)
    opts['ttl_fontstyle'] = kwargs.get('ttl_fontstyle', 'normal')
    opts['ttl_fontfamily'] = kwargs.get('ttl_fontfamily', 'sans-serif')

    opts['ax_fontsize'] = kwargs.get('ax_fontsize', 10)
    opts['ax_fontstyle'] = kwargs.get('ax_fontstyle', 'normal')
    opts['ax_fontfamily'] = kwargs.get('ax_fontfamily', 'sans-serif')

    return opts


def set_title_label(fig, ax_num, opts, ttl="", x_lbl="", y_lbl=""):
    fig.axes[ax_num].set_title(ttl, fontsize=opts['ttl_fontsize'],
                               fontstyle=opts['ttl_fontstyle'],
                               fontfamily=opts['ttl_fontfamily'])
    fig.axes[ax_num].set_xlabel(x_lbl, fontsize=opts['ax_fontsize'],
                                fontstyle=opts['ax_fontstyle'],
                                fontfamily=opts['ax_fontfamily'])
    fig.axes[ax_num].set_ylabel(y_lbl, fontsize=opts['ax_fontsize'],
                                fontstyle=opts['ax_fontstyle'],
                                fontfamily=opts['ax_fontfamily'])
