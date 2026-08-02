# Overview

Welcome to the DARe Hub and Climate Co-Centre weather generator repository !

The weather generator repository contains python scripts to generate stochastic single-site and multi-site simulations of rainfall and single-site simulations for temperature. The python scripts presented here are drawn from - https://rwgen1.github.io/rwgen/html/index.html and are modified as per the need. This weather generator (WG) has the potential to simulate variables such as wind speed, vapour pressure and sunshine duration at daily and hourly temporal scales conditional on data availability of those variables. The input data to run these scripts are provided for single-site simulations, however for multi-site simulations data shall be provided upon request. 

These scripts are straightforward to run, the **BaseScripts** folder contains the essential python scripts for single-site and multi-site simulations which generate stochastic simulations of rainfall and temperature followed by diagnosis of the simulations. Using these scripts as the base the Neyman-Scott Rectangular Pulse (NSRP) rainfall model parameters and regression parameters for temperature are derived for the future periods 2041 - 2060 and 2061 - 2080 using change factors from UK Climate Projections 2018 (UKCP18) ensemble members. Thereby a parameter dataset is formulated to derive projections of future climate and the corresponding scripts for that purpose are provided in **ParameterComputationScripts** folder. 

# Components of the WG

This WG has two components, first being the rainfall generator based on NSRP model and the second being the non-rainfall variable generator. The linkage between these two components is established via transition states of rainfall. The schematic below gives a brief overview of the WG.

<img width="521" height="266" alt="image" src="https://github.com/user-attachments/assets/5c06bd27-6d38-481a-a5df-22109904c3bc" />

The output from the weather generator (WG) looks as seen below for a site named Shawbury. The grey lines in the plots below are stochastic simulations from the WG, averaging these simulations brings the value close to the observed ones. It is important that a user verifies how the WG works in terms of simulating means and extremes of the variable of interest before employing the WG outputs across any downstream applications.  
<img width="11882" height="5885" alt="Shawbury" src="https://github.com/user-attachments/assets/eab341da-9267-4cee-8ee6-6213b46f57f6" />
<img width="11851" height="5851" alt="Shawbury_WG" src="https://github.com/user-attachments/assets/e0f74d53-2c78-4799-beca-c2342f8b5f0d" />

# Status
While these scripts are a straight forward implementation of the framework mentioned above, efforts will be directed in the future to consolidate these scripts to a python package.

# Acknowledgements
This work is supported by funding grants from - [National Hub for Decarbonised, Adaptable, and Resilient Transport Infrastructures](url) and [Climate Co-Centre](url)

# References
These scripts are built upon two foundational works given below. The current work here uses the established framework in deriving parameter datasets to generate future climate projections

1. Kilsby, C.G., Jones, P.D., Burton, A., Ford, A.C., Fowler, H.J., Harpham, C., James, P., Smith, A. and Wilby, R.L., 2007. A daily weather generator for use in climate change studies. Environmental Modelling & Software, 22(12), pp.1705-1719. https://doi.org/10.1016/j.envsoft.2007.02.005
2. Burton, A., Kilsby, C.G., Fowler, H.J., Cowpertwait, P.S.P. and O'connell, P.E., 2008. RainSim: A spatial–temporal stochastic rainfall modelling system. Environmental Modelling & Software, 23(12), pp.1356-1369. https://doi.org/10.1016/j.envsoft.2008.04.003
