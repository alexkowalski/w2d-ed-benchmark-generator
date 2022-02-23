import argparse
import tempfile
import sys
import os

import numpy as np


template = """
[General]
DOS = {dosmode}
half-bandwidth = {bethe_hbws}
muimpFile = {muimp_fname}
beta = {beta}
EPSN = 0.0
NAt = 1
DMFTsteps = {dmftiter}
StatisticSteps = {statiter}
FileNamePrefix = edbenchmark
mixing = {mixing}
magnetism = para
FTType = none
[Atoms]
[[1]]
Nd = {norb}
Hamiltonian = {hmode}
Udd = {uloc}
Jdd = {jh}
Vdd = {ust}
QuantumNumbers = Nt Szt {qn}
[QMC]
Nwarmups = {nwarmups}
Nmeas = {nmeas}
NCorr = {ncorr}
Ntau = 2000
Niw = 2000
Nftau = 2000
NLegMax = 1
NLegOrder = 1
MeasGiw = 1
NLookup_nfft = 200000
"""

parser = argparse.ArgumentParser(description="Run a DMFT calculation for a "
                                 "Bethe lattice Hubbard model using EDIpack "
                                 "as impurity solver.")
parser.add_argument("--mode", metavar='MODE',
                    choices=("CTHYB", "DMFT"),
                    default="CTHYB",
                    help="Benchmark mode (CTHYB: impurity solution comparison "
                    "based on last ED DMFT iteration, DMFT: DMFT solution "
                    "comparison, cannot be expected to coincide with ED due "
                    "to ED bath discretization error)")
parser.add_argument("--half-bandwidths", metavar='HALF_BW',
                    type=lambda x: abs(float(x)), default=[],
                    help="half-bandwiths of all bands",
                    action='extend', nargs='*')
parser.add_argument("--impurity-levels", metavar='LEVEL',
                    type=float, default=[],
                    help="impurity levels (crystal field)",
                    action='extend', nargs='*')
parser.add_argument("--Uloc", metavar='U',
                    type=float, default=[],
                    help="same-orbital interaction",
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
parser.add_argument("--nwarmups", metavar='NWARMUPS',
                    type=int, default=20000000,
                    help="Number of warmup steps")
parser.add_argument("--nmeas", metavar='NMEAS',
                    type=int, default=100000,
                    help="Number of measurements")
parser.add_argument("--ncorr", metavar='NCORR',
                    type=int, default=1000,
                    help="Number of steps between measurements")
parser.add_argument("--dmftiter", metavar='MAXITER',
                    type=int, default=50,
                    help="Maximum number of DMFT iterations")
args = parser.parse_args()

assert len(args.half_bandwidths) == len(args.impurity_levels), \
    "# half-bws must equal # imp. levels"

norb = len(args.half_bandwidths)

if args.Jx == 0.0 and args.Jp == 0.0:
    hmode = "Density"
    qn = "Azt"
elif args.Jx == args.Jh and args.Jp == args.Jh:
    hmode = "Kanamori"
    qn = "Qzt"
else:
    raise NotImplementedError("Automatic generation for given Jp or Jx "
                              "parameter unsupported.")

if any(entry != args.Uloc[0] for entry in args.Uloc):
    raise NotImplementedError("Automatic generation for orbital-dependent U "
                              "unsupported.")

cfgfd, cfgfilename = tempfile.mkstemp(".in",
                                      "w2dparams.",
                                      os.getcwd(),
                                      True)
levelfd, levelfilename = tempfile.mkstemp(".dat",
                                          "w2dlevels.",
                                          os.getcwd(),
                                          True)

if args.mode == "CTHYB":
    # Read bath levels and hybridization from last ED DMFT iteration
    # and write w2d EDcheck input
    try:
        bath_Es_Vs = np.loadtxt("hamiltonian.used")
    except Exception as e:
        raise RuntimeError("The working directory must contain a Hamiltonian "
                           "file from a finished EDIpack calculation to "
                           "generate input files for benchmark mode CTHYB.") \
                           from e

    bath_Es = bath_Es_Vs[:, 0::2]
    bath_Vs = bath_Es_Vs[:, 1::2]

    with open("epsk", "x") as epskfile:
        np.savetxt(epskfile, bath_Es)
    with open("Vk", "x") as Vkfile:
        np.savetxt(Vkfile, bath_Vs)
    dosmode = "EDcheck"
else:
    sys.stderr.write("WARNING: Using benchmark mode DMFT which will not "
                     "reproduce ED results exactly due to ED discretization "
                     "error")
    dosmode = "Bethe"

with os.fdopen(levelfd, "x") as f:
    np.savetxt(f, args.mu - np.array((args.impurity_levels,
                                      args.impurity_levels)).T.flatten())

with os.fdopen(cfgfd, "x") as f:
    f.write(
        template.format(dosmode=dosmode,
                        bethe_hbws=", ".join(map(str, args.half_bandwidths)),
                        norb=norb,
                        muimp_fname=levelfilename,
                        beta=args.beta,
                        dmftiter=(args.dmftiter if args.mode == "DMFT" else 0),
                        statiter=(1 if args.mode == "CTHYB" else 0),
                        mixing=args.mixing,
                        hmode=hmode,
                        uloc=(args.Uloc[0]),
                        ust=str(args.Ust),
                        jh=str(args.Jh),
                        qn=qn,
                        nwarmups=args.nwarmups,
                        nmeas=args.nmeas,
                        ncorr=args.ncorr
                        )
        )