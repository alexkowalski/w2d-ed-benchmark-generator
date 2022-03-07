# Scripts for simple and fast generation of ED benchmark input data for the w2dynamics CT-HYB solver derived from Hubbard model DMFT calculations

This repository contains scripts that can be used to perform ED DMFT calculations for the Hubbard model on a Bethe lattice using the [EDIpack](https://github.com/QcmPlab/EDIpack) exact diagonalization solver, generate [w2dynamics](https://github.com/w2dynamics/w2dynamics) input files for one impurity solution step, and create plots of resulting Green's functions and self-energies from the ED and CT-HYB impurity solutions.

## Instructions

When following these instructions, you should keep in mind that for both ED and QMC there are calculation parameters whose optimal values may vary strongly depending on the model parameters. Not all of them are adjustable using just the user interface provided by the scripts in this repository and manual modifications to the solver input files and the scripts may be necessary for correct results.

* Use the `edipack_hubbard_bethe.py` script to run a DMFT calculation (parallelizable using MPI) and pass the model parameters as options. The number of orbitals is inferred from the number of half-bandwidths passed as arguments. You can view a full list of options by passing the `-h` flag. Consider that the w2dynamics input generator, which takes the same command line options, does not (at the moment) implement support for all possible combinations of input values; in particular, you must pass the same intra-orbital interaction parameter U for all orbitals and the spin-exchange and pair-hopping parameters must be either both zero or both equal to the Hund's coupling parameter, or you will need to generate the corresponding input for w2dynamics yourself.

* After the DMFT calculation is done, use the `w2d_hubbard_bethe_configgen.py` script in the directory containing the ED results to generate w2dynamics input files that allow you to calculate the solution for the impurity problem that was solved in the last DMFT iteration of the ED DMFT calculation. Pass the same model parameters to this script as to the previous one. The QMC control parameters need to be adjusted in the usual way, and while the script can take some of them as command-line parameters, it cannot automatically set suitable values and there may be parameters that need to be changed manually in the generated configuration file to produce satisfactory results for your specific choice of parameters.

* Use the `cthyb` script included in w2dynamics to run a CT-HYB calculation (parallelizable using MPI) for the impurity problem by passing the path to the generated `w2dparams.*.in` file (with random infix) as argument.

* Use the `generate_plots.py` script in the directory containing the EDIpack and w2dynamics result files to generate comparison plots of the Green's functions and self-energies on Matsubara frequencies.

Examples for results can be found in [this repository](https://github.com/alexkowalski/w2d-ed-benchmark-examples).
