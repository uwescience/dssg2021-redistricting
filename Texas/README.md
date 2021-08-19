## Voting Rights Act Compliance in Texas

The results from this analysis are summarized on our project website: https://uwescience.github.io/DSSG2021-redistricting-website/texas/

This analysis expands upon work done by the MGGG Redistricting Lab, and relies heavily on the modeling code and input data sources outlined [here](https://github.com/mggg/VRA_ensembles).

03_TX_model.py is the main model file (adapted from the Texas modeling [file](https://github.com/mggg/VRA_ensembles/blob/master/TX/TX_elections_model.py) provided by MGGG) and the run_functions file has supporting functions. 

To get started, download all Data & Input files outlined on the MGGG VRA ensembles 
github into the same local TX/ directory with the following exception:

**Instead of using the MGGG provided shapefile, unzip the 'TX_VTDs_POP2019.zip' directory and use the related shapefiles.** This shapefile has been created by augmenting the orginal MGGG Texas shapefile with additional ACS 2019 population data for the ensemble analysis comparing plans from 2010 to plans from 2020. The data wrangling steps to combine the two data sources are outlined in 01_TX_data_wrangling.py.

As outlined on the MGGG github, you'll also need to download the two csv files with ecological inference precinct data that are too large and hosted by MGGG on [dropbox](https://www.dropbox.com/sh/k78n2hyixmv9xdg/AABmZG5ntMbXtX1VKThR7_t8a?dl=0).

Running the 03_TX_model.py script will make an 'outputs' folder in the same directory.
