import argparse
import tempfile
import sys
import os

import numpy as np
import scipy.integrate as igr
import edipy as ed
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    def print(*args, **kwargs):
        pass


def exception_mpi_abort(type, value, traceback):
    """Call original exception hook, then terminate all processes."""
    sys.__excepthook__(type, value, traceback)
    sys.stderr.write(f"Error: Exception at top-level on rank {rank} "
                     "(see previous output for error message)\n")
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)


sys.excepthook = exception_mpi_abort


def bethe_dos(omega, hbw):
    """Density of states for a Bethe lattice of half-bandwidth hbw at frequency
    omega.
    """
    return 2 * np.sqrt(1.0 - (omega/hbw)**2 + 0.0j).real / np.pi / hbw


template = """
 NORB={norb}
 NBATH={nbath}
 NSPIN=1
 NPH=0
 BATH_TYPE=normal
 ULOC={uloc}
 UST={ust}
 JH={jh}
 JX={jx}
 JP={jp}
 BETA={beta}
 XMU={mu}
 PH_TYPE=1
 G_PH=0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0
 W0_PH=0.d0
 NLOOP={niter}
 DMFT_ERROR={dmft_threshold}
 NSUCCESS=2
 SB_FIELD=1.000000000E-01
 ED_FINITE_TEMP=T
 ED_TWIN=F
 ED_SECTORS=F
 ED_SECTORS_SHIFT=1
 ED_SPARSE_H=T
 ED_TOTAL_UD=T
 ED_SOLVE_OFFDIAG_GF=F
 ED_PRINT_SIGMA=T
 ED_PRINT_G=T
 ED_PRINT_G0=T
 ED_VERBOSE=2
 ED_HW_BATH={inithbw}
 ED_OFFSET_BATH=1.000000000E-01
 LMATS=2000
 LREAL=2000
 LTAU=2000
 LPOS=100
 NREAD=0.d0
 NERR=1.000000000E-04
 NDELTA=1.000000000E-01
 NCOEFF=1.000000000
 WINI=-5.000000000
 WFIN=5.000000000
 XMIN=-3.000000000
 XMAX=3.000000000
 CHISPIN_FLAG=F
 CHIDENS_FLAG=F
 CHIPAIR_FLAG=F
 CHIEXCT_FLAG=F
 HFMODE=F
 EPS=1.000000000E-02
 CUTOFF=1.000000000E-09
 GS_THRESHOLD=1.000000000E-09
 LANC_METHOD=arpack
 LANC_NSTATES_SECTOR=20
 LANC_NSTATES_TOTAL=200
 LANC_NSTATES_STEP=2
 LANC_NCV_FACTOR=2
 LANC_NCV_ADD=0
 LANC_NITER=768
 LANC_NGFITER=768
 LANC_TOLERANCE=1.000000000E-18
 LANC_DIM_THRESHOLD=4096
 CG_METHOD=0
 CG_GRAD=0
 CG_LFIT=1000
 CG_FTOL=1.000000000E-05
 CG_STOP=0
 CG_NITER=500
 CG_WEIGHT=1
 CG_SCHEME=delta
 CG_POW=2
 SECTORFILE=sectors
 HFILE=hamiltonian
 LOGFILE=6
"""

parser = argparse.ArgumentParser(description="Run a DMFT calculation for a "
                                 "Bethe lattice Hubbard model using EDIpack "
                                 "as impurity solver.")
parser.add_argument("--mode", metavar='MODE',
                    help="Used by w2dynamics config generator only, ignored")
parser.add_argument("--half-bandwidths", metavar='HALF_BW',
                    type=lambda x: abs(float(x)), default=[],
                    help="half-bandwidths of all bands",
                    action='extend', nargs='*')
parser.add_argument("--impurity-levels", metavar='LEVEL',
                    type=float, default=[],
                    help="impurity levels (crystal field)",
                    action='extend', nargs='*')
parser.add_argument("--Uloc", metavar='U',
                    type=float, default=[],
                    help="intra-orbital interaction",
                    action='extend', nargs='*')
parser.add_argument("--Ust", metavar='V',
                    type=float, default=0.0,
                    help="inter-orbital different-spin interaction")
parser.add_argument("--Jh", metavar='JH',
                    type=float, default=0.0,
                    help="Hund's coupling (density)")
parser.add_argument("--Jx", metavar='JX',
                    type=float, default=0.0,
                    help="Hund's coupling (spin-exchange)")
parser.add_argument("--Jp", metavar='JP',
                    type=float, default=0.0,
                    help="Hund's coupling (pair-hopping)")
parser.add_argument("--beta", metavar='BETA',
                    type=lambda x: abs(float(x)), default=100,
                    help="Inverse temperature")
parser.add_argument("--mu", metavar='MU',
                    type=float, default=1.0,
                    help="Chemical potential")
parser.add_argument("--mixing", metavar='OLD_SHARE',
                    type=float, default=0.2,
                    help="Share of old hybridization in new mixed input")
parser.add_argument("--mixhist", metavar='OLD_HISTSIZE',
                    type=int, default=4,
                    help="Number of old hybridizations used for modified "
                    "Broyden's method mixing (<= 1: linear mixing) ")
parser.add_argument("--nbath", metavar='NBATH',
                    type=int, default=None,
                    help="Bath sites per orbital (default: total size < 10)")
parser.add_argument("--nwarmups", metavar='NWARMUPS',
                    help="Used by w2dynamics config generator only, ignored")
parser.add_argument("--nmeas", metavar='NMEAS',
                    help="Used by w2dynamics config generator only, ignored")
parser.add_argument("--ncorr", metavar='NCORR',
                    help="Used by w2dynamics config generator only, ignored")
parser.add_argument("--maxiter", metavar='MAXITER',
                    type=int, default=50,
                    help="Maximum number of DMFT iterations")
parser.add_argument("--convergence-threshold", metavar='CONV_THRESH',
                    type=lambda x: abs(float(x)), default=1e-02,
                    help="Convergence threshold (relative error of summed "
                    "absolute values of hybridization) for DMFT termination")
parser.add_argument("--override-configfile", metavar='FILE',
                    type=argparse.FileType('r'), default=None,
                    help="Use custom config file without modifications "
                    "instead of template (WARNING: you must avoid "
                    "inconsistencies between the file and other flags "
                    "yourself!)")
args = parser.parse_args()


norb = len(args.half_bandwidths)
assert norb > 0, ("The half-bandwidth for EVERY band must be passed "
                  "EXPLICITLY (at least one)")

if len(args.impurity_levels) == 0:
    args.impurity_levels = [0.0] * norb
    print(f"WARNING: all {norb} impurity levels implicitly set to 0",
          file=sys.stderr)
elif len(args.impurity_levels) == 1 and norb != 1:
    args.impurity_levels = args.impurity_levels * norb
    print(f"WARNING: all {norb} impurity levels implicitly set to "
          f"{args.impurity_levels[0]}",
          file=sys.stderr)
elif len(args.impurity_levels) != norb:
    raise ValueError(f"Passed {len(args.impurity_levels)} impurity levels for "
                     f"{norb} orbitals (inferred from half-bandwidths)")

if len(args.Uloc) == 0:
    args.Uloc = [0.0] * norb
    print(f"WARNING: all {norb} intra-orbital interactions U implicitly "
          "set to 0",
          file=sys.stderr)
elif len(args.Uloc) == 1 and norb != 1:
    args.Uloc = args.Uloc * norb
    print(f"WARNING: all {norb} intra-orbital interactions U implicitly set "
          f"to {args.Uloc[0]}",
          file=sys.stderr)
elif len(args.Uloc) != norb:
    raise ValueError(f"Passed {len(args.Uloc)} intra-orbital interactions U "
                     f"for {norb} orbitals (inferred from half-bandwidths)")


Uloc = [0.0] * 5
Uloc[:len(args.Uloc)] = args.Uloc[:]

# fill arguments into EDIpack configfile template, write it to a
# temporary file and read it
if rank == 0:
    cfgfd, cfgfname = tempfile.mkstemp(".conf", "inputED.", os.getcwd(), True)

    with os.fdopen(cfgfd, "x") as f:
        if args.override_configfile is not None:
            f.write(args.override_configfile.read())
        else:
            f.write(
                template.format(norb=norb,
                                nbath=(args.nbath
                                       if args.nbath is not None
                                       else (10 - norb) // norb),
                                uloc=(",".join([str(U).replace("e", "d")
                                                for U in Uloc])),
                                ust=str(args.Ust).replace("e", "d"),
                                jh=str(args.Jh).replace("e", "d"),
                                jx=str(args.Jx).replace("e", "d"),
                                jp=str(args.Jp).replace("e", "d"),
                                beta=str(args.beta).replace("e", "d"),
                                mu=str(args.mu).replace("e", "d"),
                                niter=args.maxiter,
                                dmft_threshold=(
                                    str(args.convergence_threshold)
                                    .replace("e", "d")
                                ),
                                inithbw=np.mean(args.half_bandwidths),
                                # Intended for ed_total_ud (optional):
                                # totalqn=("T"
                                #          if args.Jx != 0.0 or args.Jp != 0.0
                                #          else "F")
                                )
            )

    cfgfname = MPI.COMM_WORLD.bcast(cfgfname.split('/')[-1], root=0)
else:
    cfgfname = MPI.COMM_WORLD.bcast(None, root=0)

MPI.COMM_WORLD.Barrier()
ed.read_input(cfgfname)
MPI.COMM_WORLD.Barrier()


bethe_hbws = args.half_bandwidths
mixing = args.mixing
mu = args.mu

Erange = np.linspace(-max(bethe_hbws), max(bethe_hbws), 10000)
dos = np.stack([bethe_dos(Erange, hbw) for hbw in bethe_hbws], axis=0)

omega_matsubara = (2 * np.arange(ed.Lmats) + 1) * np.pi / ed.beta

Sigma_iw = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lmats),
                    dtype=np.complex128, order='F')
G_iw = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lmats),
                dtype=np.complex128, order='F')
Delta_iw = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lmats),
                    dtype=np.complex128, order='F')
Hloc = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb),
                dtype=np.double, order='F')
for i in range(ed.Nspin):
    Hloc[i, i, :, :] = np.diagflat(args.impurity_levels)


Nbath = ed.get_bath_dimension()
bath = np.zeros(Nbath, dtype=np.double, order='F')
bath_history = np.zeros((0, Nbath), dtype=np.double, order='F')
residual_history = np.zeros((0, Nbath), dtype=np.double, order='F')
last_bath = None
ed.init_solver(bath)

if args.mixhist > 1:
    def mix(new_trial):
        """Mix the bath parameters using modified Broyden's method
        (D.D. Johnson, PRB 38, 12807)"""
        global bath_history
        global residual_history

        if bath_history.shape[0] <= 0:
            proposed_trial = new_trial
        else:
            last_trial = bath_history[-1, :]
            residual = new_trial - last_trial
            if residual_history.shape[0] >= args.mixhist:
                residual_history = residual_history[min(-args.mixhist + 1, -1):,
                                                    :].copy()
                bath_history = bath_history[-args.mixhist:, :].copy()
            residual_history = np.append(residual_history,
                                         residual[np.newaxis, :],
                                         axis=0)

            if residual_history.shape[0] <= 1:
                proposed_trial = last_trial + (1.0 - mixing) * residual
            else:
                deltaF = np.diff(residual_history, axis=0)
                normDeltaF = np.linalg.norm(deltaF, axis=1, keepdims=True)
                deltaF /= normDeltaF
                deltaV = np.diff(bath_history, axis=0)/normDeltaF

                # first calculate A, then overwrite it with beta
                beta = np.matmul(deltaF, np.conj(np.transpose(deltaF)))
                beta = np.linalg.pinv((0.001 * np.eye(beta.shape[0]) + beta),
                                      hermitian=True)

                proposed_trial = last_trial + (1.0 - mixing) * residual \
                    - np.linalg.multi_dot((residual,
                                           np.conj(np.transpose(deltaF)),
                                           beta,
                                           (1.0 - mixing) * deltaF + deltaV))
        bath_history = np.append(bath_history,
                                 proposed_trial[np.newaxis, :],
                                 axis=0)
        return proposed_trial
else:
    def mix(new_trial):
        """Linear mixing"""
        global last_bath
        if last_bath is not None:
            proposed_trial = (1.0 - mixing) * new_trial + mixing * last_bath
        else:
            proposed_trial = new_trial
        last_bath = proposed_trial
        return proposed_trial


for iloop in range(1, ed.Nloop + 1):
    print("Starting DMFT iteration ", iloop, flush=True)

    ed.solve(bath, Hloc)

    ed.get_sigma_matsubara(Sigma_iw)

    for isp in range(ed.Nspin):
        for iorb in range(ed.Norb):
            G_iw[isp, isp, iorb, iorb, :] = (
                igr.simpson(dos[iorb, :] /
                            (1.0j * omega_matsubara[:, np.newaxis]
                             + mu
                             - Hloc[isp, isp, iorb, iorb]
                             - Erange[:]
                             - Sigma_iw[isp, isp, iorb, iorb, :, np.newaxis]),
                            x=Erange, axis=-1)
            )

    Delta_iw[:, :, :, :, :] = (np.eye(ed.Nspin)[:,
                                                :,
                                                np.newaxis,
                                                np.newaxis,
                                                np.newaxis]
                               * (np.diagflat(bethe_hbws)**2/4.0)[np.newaxis,
                                                                  np.newaxis,
                                                                  :,
                                                                  :,
                                                                  np.newaxis]
                               * G_iw[:, :, :, :, :])
    for isp in range(ed.Nspin):
        for iorb in range(ed.Norb):
            ed.chi2_fitgf(Delta_iw, bath, ispin=(isp+1), iorb=(iorb+1))

    bath = mix(bath)

    _, converged = ed.check_convergence(Delta_iw[:, :, :, :, :],
                                        ed.dmft_error,
                                        ed.Nsuccess,
                                        ed.Nloop)
    if converged:
        break
