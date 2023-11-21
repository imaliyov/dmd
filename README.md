# Dynamic Mode Decomposition

This repository provides a Python implementation of the standard and high-order dynamic mode decomposition (DMD[1-3] and HODMD[4]). The most time consuming part, the singular value decomposition (SVD) is performed using numpy. Loops over the space are vectorized.

# References

1. S. L Brunton and J. N. Kutz, *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*, Cambridge University Press, 2022. [DOI: 10.1017/9781108380690](https://doi.org/10.1017/9781108380690)

2. J. Yin, Y. Chan, F. H. da Jornada, D. Y. Qiu, C. Yang, and S. G. Louie, *Analyzing and predicting non-equilibrium many-body dynamics via dynamic mode decomposition*,  J. Comput. Phys., **477**, 2023, pp. 111909. [DOI: 10.1016/j.jcp.2023.111909](https://doi.org/10.1016/j.jcp.2023.111909)

3. I. Maliyov, J. Yin, J. Yao, C. Yang, and M. Bernardi, *Dynamic mode decomposition of nonequilibrium electron-phonon dynamics: accelerating the first-principles real-time Boltzmann equation*, [arXiv: 2311.07520](https://arxiv.org/abs/2311.07520)


