# 22-23: PROGRAMMING FOR DATA ANALYSIS
<b>Lais Coletta Pereira</b>
<br>

![Alt text](relative/path/to/img.jpg?raw=true "Title")

## Description

For this module we were asked to model and synthesise a dataset using Python. The dataset should be a real-world phenomenon of data set of our choice and should contain at least one-hundred data points across at least four different variables. 

For my project I chose to analyse the Earth's temperature dataset that I found at https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data


Requirements of the project:

• Investigate the types of variables involved, their likely distributions, and their relationships with each other.
• Synthesise/simulate a data set as closely matching their properties as possible.
• Detail your research and implement the simulation in a Jupyter notebook – the data set itself can simply be displayed in an output cell within the notebook.

About Dataset
Some say climate change is the biggest threat of our age while others say it’s a myth based on dodgy science. We are turning some of the data over to you so you can form your own view.

us-climate-change

Even more than with other data sets that Kaggle has featured, there’s a huge amount of data cleaning and preparation that goes into putting together a long-time study of climate trends. Early data was collected by technicians using mercury thermometers, where any variation in the visit time impacted measurements. In the 1940s, the construction of airports caused many weather stations to be moved. In the 1980s, there was a move to electronic thermometers that are said to have a cooling bias.

Given this complexity, there are a range of organizations that collate climate trends data. The three most cited land and ocean temperature data sets are NOAA’s MLOST, NASA’s GISTEMP and the UK’s HadCrut.

We have repackaged the data from a newer compilation put together by the Berkeley Earth, which is affiliated with Lawrence Berkeley National Laboratory. The Berkeley Earth Surface Temperature Study combines 1.6 billion temperature reports from 16 pre-existing archives. It is nicely packaged and allows for slicing into interesting subsets (for example by country). They publish the source data and the code for the transformations they applied. They also use methods that allow weather observations from shorter time series to be included, meaning fewer observations need to be thrown away.

In this dataset, we have include several files:

Global Land and Ocean-and-Land Temperatures (GlobalTemperatures.csv):

Date: starts in 1750 for average land temperature and 1850 for max and min land temperatures and global ocean and land temperatures
LandAverageTemperature: global average land temperature in celsius
LandAverageTemperatureUncertainty: the 95% confidence interval around the average
LandMaxTemperature: global average maximum land temperature in celsius
LandMaxTemperatureUncertainty: the 95% confidence interval around the maximum land temperature
LandMinTemperature: global average minimum land temperature in celsius
LandMinTemperatureUncertainty: the 95% confidence interval around the minimum land temperature
LandAndOceanAverageTemperature: global average land and ocean temperature in celsius
LandAndOceanAverageTemperatureUncertainty: the 95% confidence interval around the global average land and ocean temperature
Other files include:

Global Average Land Temperature by Country (GlobalLandTemperaturesByCountry.csv)
Global Average Land Temperature by State (GlobalLandTemperaturesByState.csv)
Global Land Temperatures By Major City (GlobalLandTemperaturesByMajorCity.csv)
Global Land Temperatures By City (GlobalLandTemperaturesByCity.csv)
The raw data comes from the Berkeley Earth data page.