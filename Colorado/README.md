## Political Competitiveness in Colorado

The case study analysis provided here focuses on the map drawing phase and demonstrates some approaches to using GerryChain during this time period. Our analysis translate Colorado's redistricting rules into a series of quantitative specifications to ensure that the proposed district maps produced by GerryChain conform to state guidelines and respect the priorities of the legislature. We specifically focused on proposed Congressional district maps that maximize politically competitive districts while minimizing county splits. Please review our [GerryChain Stakeholder User Guide](https://uwescience.github.io/DSSG2021-redistricting-website/guide/) for more details of this project.

- `01_CO_Data_Wrangling.py`: This script cleaned and merged archival election data from the Colorado Secretary of State, and built a precinct-level panel data of all federal election results from 2004 - 2020. The raw files and the generated csv files can be found in the `Data` subfolder.
- `02_CO_EDA_Modeling_Decisions.ipynb`: This notebook walks through seven decision points users can consider when conducting specific state-level analyses and considering GerryChain-specific modeling decisions.
- `03_CO_Gerrychain.py`: This script applies the modeling decisions explored in `02_CO_EDA_Modeling_Decisions.ipynb`.

The results from this analysis are summarized on our project website: https://uwescience.github.io/DSSG2021-redistricting-website/colorado/
