#!/usr/bin/env python
import glob
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cycler
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--xmax", help="upper limit of Matsubara frequency axis "
                    "range",
                    type=float, default=3.0)
parser.add_argument("--w2doutfile",
                    help="HDF5 file with w2dynamics data to be used (default: "
                    "lexicographically last in current directory)",
                    type=h5py.File, default=None)
args = parser.parse_args()

if args.w2doutfile is not None:
    w2dout = args.w2doutfile
else:
    w2dout = h5py.File(sorted(glob.iglob("*.hdf5"), reverse=True)[0], "r")

try:
    niw = w2dout["/stat-last/ineq-001/siw-full/value"][()].shape[-1]
    iterkey = "/stat-last/ineq-001"
except KeyError:
    niw = w2dout["/dmft-last/ineq-001/siw-full/value"][()].shape[-1]
    iterkey = "/dmft-last/ineq-001"


def w2d_im_iw_spins_avgerr(group):
    def get_array(subkey):
        # read only positive frequencies, take diagonal, transpose to
        # (o, s, iw)
        return np.transpose(
            np.diagonal(
                np.diagonal(
                    group[subkey][:, :, :, :, niw//2:],
                    axis1=0, axis2=2),
                axis1=0, axis2=1),
            (1, 2, 0)
        )

    errs = get_array("error")
    vals = np.imag(get_array("value"))

    # error-weighted spin average
    vals = np.sum(vals / errs**2, axis=1) / np.sum(1.0 / errs**2, axis=1)
    errs = np.sqrt(1.0 / np.sum(1.0 / errs**2, axis=1))
    return vals, errs


w2dsiw, w2dsiwerr = w2d_im_iw_spins_avgerr(w2dout[f"{iterkey}/siw-full"])
w2dgiw, w2dgiwerr = w2d_im_iw_spins_avgerr(w2dout[f"{iterkey}/giw-full"])
w2diw = (2 * np.pi * (np.arange(0, w2dsiw.shape[-1]) + 0.5)
         / w2dout[".config"].attrs['general.beta'])

edsiw = None
edgiw = None
ediw = None

for orb in range(w2dsiw.shape[0]):
    partsiw = np.loadtxt(f"impSigma_l{orb+1}{orb+1}_s1_iw.ed")
    if edsiw is None:
        edsiw = np.zeros((w2dsiw.shape[0], partsiw.shape[0]))
        edgiw = np.zeros((w2dsiw.shape[0], partsiw.shape[0]))
        ediw = partsiw[:, 0]
    edsiw[orb, :] = partsiw[:, 1]

    partgiw = np.loadtxt(f"impG_l{orb+1}{orb+1}_s1_iw.ed")
    edgiw[orb, :] = partgiw[:, 1]

minlen = min(len(ediw), len(w2diw))
assert np.allclose(ediw[:minlen], w2diw[:minlen]), \
    "Error: Frequency mismatch between outputs"

xmin = 0
xmax = args.xmax
yminG = min(np.amin(w2dgiw[:, :np.searchsorted(w2diw, xmax)]),
            np.amin(edgiw[:, :np.searchsorted(ediw, xmax)]))
yminS = min(np.amin(w2dsiw[:, :np.searchsorted(w2diw, xmax)]),
            np.amin(edsiw[:, :np.searchsorted(ediw, xmax)]))
ymaxG = max(np.amax(w2dgiw[:, :np.searchsorted(w2diw, xmax)]),
            np.amax(edgiw[:, :np.searchsorted(ediw, xmax)]))
ymaxS = max(np.amax(w2dsiw[:, :np.searchsorted(w2diw, xmax)]),
            np.amax(edsiw[:, :np.searchsorted(ediw, xmax)]))
yminG, ymaxG = yminG - 0.05 * (ymaxG - yminG), ymaxG + 0.05 * (ymaxG - yminG)
yminS, ymaxS = yminS - 0.05 * (ymaxS - yminS), ymaxS + 0.05 * (ymaxS - yminS)
ymaxG, ymaxS = max(0.0, ymaxG), max(0.0, ymaxS)

figG, axG = plt.subplots(figsize=(9, 6))
figS, axS = plt.subplots(figsize=(9, 6))


def colorcycler(size):
    return cycler.cycler(
        color=plt.cm.inferno(
            np.linspace(0, 1, size + 2)[1:-1]
        )
    )


def markercycler():
    return cycler.cycler(
        # choice of markers reasonably well marking the points (prefer
        # crossing lines to surrounding, prefer symmetric to
        # unsymmetric) with reasonably little overlap with each other
        # or the (vertical) errorbars
        marker=['x', '3', '4', '.', 'o', 'D', 'p', 's']
    )


colors = defaultdict(lambda g=iter(colorcycler(w2dsiw.shape[0])): next(g))
markers = defaultdict(lambda g=iter(markercycler()): next(g))

for orb in range(w2dsiw.shape[0]):
    axS.plot(ediw, edsiw[orb, :], ':', label=f"EDIpack ED orb {orb+1}",
             fillstyle='none', **markers[f"ed{orb}"], **colors[orb])
    axS.errorbar(w2diw, w2dsiw[orb, :], yerr=w2dsiwerr[orb, :],
                 label=f"w2dynamics CTHYB orb {orb+1}", zorder=2.5,
                 fillstyle='none', **markers[f"w2{orb}"], **colors[orb])

    axG.plot(ediw, edgiw[orb, :], ':', label=f"EDIpack ED orb {orb+1}",
             fillstyle='none', **markers[f"ed{orb}"], **colors[orb])
    axG.errorbar(w2diw, w2dgiw[orb, :], yerr=w2dgiwerr[orb, :],
                 label=f"w2dynamics CTHYB orb {orb+1}", zorder=2.5,
                 fillstyle='none', **markers[f"w2{orb}"], **colors[orb])
axG.set_xlim(xmin, xmax)
axG.set_ylim(yminG, ymaxG)
axS.set_xlim(xmin, xmax)
axS.set_ylim(yminS, ymaxS)
axG.set_ylabel(r"$G(i\omega_n)$")
axS.set_ylabel(r"$\Sigma(i\omega_n)$")
for ax in (axG, axS):
    ax.grid()
    ax.minorticks_on()
    ax.legend()
    ax.set_xlabel(r"$\omega_n$")
figG.savefig("ED_CTHYB_comparison_Giw.pdf",
             bbox_inches='tight')
figS.savefig("ED_CTHYB_comparison_Sigmaiw.pdf",
             bbox_inches='tight')


w2dgiw -= edgiw
w2dsiw -= edsiw

yminG = np.amin(w2dgiw[:, :np.searchsorted(w2diw, xmax)])
yminS = np.amin(w2dsiw[:, :np.searchsorted(w2diw, xmax)])
ymaxG = np.amax(w2dgiw[:, :np.searchsorted(w2diw, xmax)])
ymaxS = np.amax(w2dsiw[:, :np.searchsorted(w2diw, xmax)])
yminG, ymaxG = yminG - 0.05 * (ymaxG - yminG), ymaxG + 0.05 * (ymaxG - yminG)
yminS, ymaxS = yminS - 0.05 * (ymaxS - yminS), ymaxS + 0.05 * (ymaxS - yminS)

figG, axG = plt.subplots(figsize=(9, 6))
figS, axS = plt.subplots(figsize=(9, 6))
axS.axhline(0.0, color='r')
axG.axhline(0.0, color='r')
markers = defaultdict(lambda g=iter(markercycler()): next(g))
for orb in range(w2dsiw.shape[0]):
    axS.errorbar(w2diw, w2dsiw[orb, :], yerr=w2dsiwerr[orb, :],
                 label=r"difference $\Sigma_{\mathrm{CTHYB}} "
                 r"- \Sigma_{\mathrm{ED}}$ orb " f"{orb+1}",
                 **colors[orb], **markers[orb],
                 capsize=4, solid_capstyle='butt')

    axG.errorbar(w2diw, w2dgiw[orb, :], yerr=w2dgiwerr[orb, :],
                 label=r"difference $G_{\mathrm{CTHYB}} "
                 r"- G_{\mathrm{ED}}$ orb " f"{orb+1}",
                 **colors[orb], **markers[orb],
                 capsize=4, solid_capstyle='butt')
axG.set_xlim(xmin, xmax)
axG.set_ylim(yminG, ymaxG)
axS.set_xlim(xmin, xmax)
axS.set_ylim(yminS, ymaxS)
axG.set_ylabel(r"$G(i\omega_n)$")
axS.set_ylabel(r"$\Sigma(i\omega_n)$")
for ax in (axG, axS):
    ax.grid()
    ax.minorticks_on()
    ax.legend()
    ax.set_xlabel(r"$\omega_n$")
figG.savefig("ED_CTHYB_difference_Giw.pdf",
             bbox_inches='tight')
figS.savefig("ED_CTHYB_difference_Sigmaiw.pdf",
             bbox_inches='tight')
