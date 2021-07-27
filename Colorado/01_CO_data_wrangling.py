
"""
01. Data Wrangling

Description: This file pulls election data from Colorado's Secretary of State Office and 
merges it into a single shapefile to use for analysis. 

"""

import os
import openpyxl
import pandas as pd
import geopandas as gpd

try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))

df = gpd.read_file("Data/co_precincts.shp")

#--- 2020

precincts_2020 = pd.read_excel("Data/Elections/2020_general_precincts.xlsx",
                               engine="openpyxl")
   
#precincts_2020.info()
#print(precincts_2020["Office/Issue/Judgeship"].unique())

precincts_2020_pres_d = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2020["Party"] == "Democratic Party")]  
precincts_2020_senate_d = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2020["Party"] == "Democratic Party")]  
precincts_2020_house_d = precincts_2020.loc[(precincts_2020["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2020["Party"] == "Democratic Party")]  

precincts_2020_pres_r = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2020["Party"] == "Republican Party")]  
precincts_2020_senate_r = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2020["Party"] == "Republican Party")]  
precincts_2020_house_r = precincts_2020.loc[(precincts_2020["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2020["Party"] == "Republican Party")]  

election_build = precincts_2020_pres_r.loc[:, ["Precinct", "Candidate Votes"]]
election_build.columns = ["PRECID", "PRES20R"]
election_build.applymap(int)

df_elections = df.join(election_build.set_index('PRECID'), on='PRECID')
#Break: Issue with joining on precinct ID from MGGG shapefile and CO SOS data
