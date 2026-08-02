#Overview

Welcome to the DARe Hub and Climate Co-Centre weather generator repository !

The weather generator repository contains python scripts to generate stochastic single-site and multi-site simulations of rainfall and single-site simulations for temperature. The python scripts presented here are drawn from - https://rwgen1.github.io/rwgen/html/index.html and are modified as per the need. It is to be noted that this weather generator also simulates variables such as wind speed, vapour pressure and sunshine duration at daily and hourly temporal scales conditional on data availability of those variables. The input data to run these scripts are provided for single-site simulations, however for multi-site simulations data shall be provided upon request. 

These scripts are straightforward to use, the BaseScripts folder contains the essential python scripts for single-site and multi-site simulations which generate stochastic simulations of rainfall and temperature followed by diagnosis of the simulations. Using these scripts as the base the Neyman-Scott Rectangular Pulse (NSRP) rainfall model parameters and regression parameters for temperature are derived for the future periods 2041 - 2060 and 2061 - 2080 using change factors from UK Climate Projections 2018 (UKCP18) ensemble members. Thereby a parameter dataset is formulated to derive projections of future climate and the corresponding scripts for that purpose are provided in ParameterComputationScripts folder. 

The output from the weather generator (WG) looks as seen below for a site named Shawbury. This WG has two components, first being the rainfall generator and the second being the non-rainfall variable generator. The non linkage between these two components is established via transition states of rainfall. 
<img width="11882" height="5885" alt="Shawbury" src="https://github.com/user-attachments/assets/eab341da-9267-4cee-8ee6-6213b46f57f6" />
<img width="11851" height="5851" alt="Shawbury_WG" src="https://github.com/user-attachments/assets/e0f74d53-2c78-4799-beca-c2342f8b5f0d" />


Status

The scripts will be consolidated to a python package soon and users are welcome to reach out to me with any questions.
