# SMSA_2024

This repository contains data, figures and scripts used in the manuscript "Proportional asymptotics of piecewise exponential proportional hazards models".

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```required_functions.py``` | Collections of functions and classes that is used to generate the data, fit the PieceWise Exponential (PWE) proportional hazards model with ridge regularization, solve the RS equations      |                               |                |
| ```run_rs.py``` | Solve of the RS equations along a user specified regularization path at a fixed $\zeta$, using a (user defined) population of size $m$ |
| ```run_sim.py```| Script that : 1) generates $m$ (user defined) data-sets, with the specifics indicated in the reference manuscript, 2) fits the PWE model along a regularization path and 3) computes the relative Brier Score as defined in the manuscript and the Harrel's C-Index on a test data-set of user defined size. 
| ```plots.py ```   | Script  for plotting.

