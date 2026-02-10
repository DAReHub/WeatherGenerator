Welcome to the DARe Hub weather generator repository !


The python scripts provided here are single-site and multi-site formulations of the weather generator. This weather generator simulates synthetic time series of rainfall, mean, minimum and maximum temperatures, wind speed, vapour pressure and sunshine duration at daily and hourly temporal scales. The input datasets used in this study are derived from a few sources which include - [Gauge based measurements]([url](https://catalogue.ceh.ac.uk/datastore/eidchub/44c577d3-665f-40de-adce-74ecad7b304a/historical_hourly-rainfall/)), Gridded data sets such as [HadUK]([url](https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/overview)) and [CEH-GEAR]([url](https://catalogue.ceh.ac.uk/documents/fc9423d6-3d54-467f-bb2b-fc7357a3941f)). 

A documentation is provided on how to use these python scripts along with some example data to run these scripts.

The output from the weather generator (WG) looks as seen below for a site named Shawbury. This WG has two components, first being the rainfall generator and the second being the non-rainfall variable generator. The non linkage between these two components is established via transition states of rainfall. 
<img width="11882" height="5885" alt="Shawbury" src="https://github.com/user-attachments/assets/eab341da-9267-4cee-8ee6-6213b46f57f6" />
<img width="11851" height="5851" alt="Shawbury_WG" src="https://github.com/user-attachments/assets/e0f74d53-2c78-4799-beca-c2342f8b5f0d" />
