# FAQ

## Error in LatticeReduction.cpp when using a trajectory from AMBER

```
(tools/LatticeReduction.cpp:42) static void PLMD::LatticeReduction::sort(PLMD::Vector*)
+++ assertion failed: m[1]<=m[2]*onePlusEpsilon
```

This is related to how PLUMED reads the lattice vector information from the input files. It might be
a problem specific to AMBER, see this
[discussion](https://groups.google.com/g/plumed-users/c/k6QoUu5LGoE/m/uzt4VGooCAAJ). In some cases
it can be solved by **converting the trajectory to pdb format** and then **erasing the CRYST
record**. Otherwise try a different PLUMED version or convert the trajectory to a different format.
