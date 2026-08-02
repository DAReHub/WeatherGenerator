Welcome to the DARe Hub and Climate Co-Centre weather generator repository !

The weather generator repository contains python scripts to generate stochastic single-site and multi-site simulations of rainfall and single-site simulations for temperature. The python scripts presented here are drawn from - https://rwgen1.github.io/rwgen/html/index.html. It is to be noted that this weather generator also simulates variables such as wind speed, vapour pressure and sunshine duration at daily and hourly temporal scales conditional of data availability of those variables. The input data to run these scripts are provided for single-site simulations, however for multi-site simulations data shall be provided upon request. 

A documentation is provided on how to use these python scripts along with some example data to run these scripts.

The output from the weather generator (WG) looks as seen below for a site named Shawbury. This WG has two components, first being the rainfall generator and the second being the non-rainfall variable generator. The non linkage between these two components is established via transition states of rainfall. 
<img width="11882" height="5885" alt="Shawbury" src="https://github.com/user-attachments/assets/eab341da-9267-4cee-8ee6-6213b46f57f6" />
<img width="11851" height="5851" alt="Shawbury_WG" src="https://github.com/user-attachments/assets/e0f74d53-2c78-4799-beca-c2342f8b5f0d" />


Status

The scripts will be consolidated to a python package soon and users are welcome to reach out to me with any questions.
