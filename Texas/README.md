## Voting Rights Act Compliance in Texas

The results from this analysis are summarized on our project website: https://uwescience.github.io/DSSG2021-redistricting-website/texas/

This analysis expands upon work done by the MGGG Redistricting Lab, and relies heavily on the modeling code and input data sources outlined [here](https://github.com/mggg/VRA_ensembles). To get started, download all Data & Input files outlined on the MGGG VRA ensembles 
[github](https://github.com/mggg/VRA_ensembles) into the same local Texas/ directory with the following exception:

**Instead of using the MGGG provided shapefile, unzip the 'TX_VTDs_POP2019.zip' directory and use the related shapefiles.** This shapefile has been created by augmenting the orginal MGGG Texas shapefile with additional ACS 2019 population data for the ensemble analysis comparing plans from the previous redistricting cycle (enacted in 2013) to the upcoming redistricting cycle (planned for 2021). The data wrangling steps to combine the two data sources are outlined in `01_TX_data_wrangling.py`.

As outlined on the MGGG github, you'll also need to download the two csv files with ecological inference precinct data that are too large and hosted by MGGG on [dropbox](https://www.dropbox.com/sh/k78n2hyixmv9xdg/AABmZG5ntMbXtX1VKThR7_t8a?dl=0).

`03_TX_model.py` is the main model file (adapted from the Texas modeling [file](https://github.com/mggg/VRA_ensembles/blob/master/TX/TX_elections_model.py) provided by MGGG) and the run_functions file has supporting functions. Running the `03_TX_model.py` script will make an 'outputs' folder in the same directory. The `02_TX_EDA_Modeling_Decisions.ipynb` notebook outlines some of the modeling considerations we explored for this analysis, and accompanies our [GerryChain User Guide](https://uwescience.github.io/DSSG2021-redistricting-website/guide/). Additionally, the `04_TX_Ensemble_Analysis.ipynb` notebook includes the code used to produce our summarized findings. 
