# RMWSPy
Version 1.0

# Introduction
RMWSPy is a Python package for conditional spatial random field simulation and inverse modelling, that provides great flexibility in its range of conditioning constraints and its applications. To get an idea of the underlying RMWS algorithm, the interested reader is referred to https://doi.org/10.1016/j.advwatres.2018.11.012 (S. Hörning and J. Sreekanth and A. Bárdossy, 2019, Computational efficient inverse groundwater modeling using Random Mixing and Whittaker–Shannon interpolation, Advances in Water Resources 123, pages 109-119). A paper describing the actual Python package will be available soon.

# Contributing
Bug reports, code contributions, and improvements are welcome from the community.

# Examples
In Version 1.0, three examples for RMWSPy are provided. The basic use of the package is described in the basic_example.ipynb Jupyter notebook. 

The second example (cml_example) combines rain gauge observations and Commercial Microwave Link data in an inversion framework to improve spatial precipitation estimation (for more details see  Haese, B., Hörning, S., Chwala, C., Bárdossy, A., Schalge, B., & Kunstmann, H., 2017, Stochastic reconstruction and interpolation of precipitation fields using combined information of commercial microwave links and rain gauges. Water Resources Research, 53, 10,740– 10,756. https://doi.org/10.1002/2017WR021015)

The third example (modflow_example) shows the application of RMWSPy to inverse groundwater modelling.
