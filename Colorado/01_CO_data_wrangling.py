
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

## TO DO: Group results by precinct ID to sum up all votes regardless of candidate

precincts_2020 = pd.read_excel("Data/Elections/2020_general_precincts.xlsx",
                               engine="openpyxl")
   
#precincts_2020.info()
#print(precincts_2020["Office/Issue/Judgeship"].unique())

precincts_2020_pres_d = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2020["Party"] == "Democratic Party")]  
precincts_2020_pres_r = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2020["Party"] == "Republican Party")]  

precincts_2020_senate_d = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2020["Party"] == "Democratic Party")]  
precincts_2020_senate_r = precincts_2020[(precincts_2020["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2020["Party"] == "Republican Party")]  

precincts_2020_house_d = precincts_2020.loc[(precincts_2020["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2020["Party"] == "Democratic Party")]  
precincts_2020_house_r = precincts_2020.loc[(precincts_2020["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2020["Party"] == "Republican Party")]  

election_build_df = [precincts_2020_pres_d,
                     precincts_2020_pres_r,
                     precincts_2020_senate_d,
                     precincts_2020_senate_r,
                     precincts_2020_house_d,
                     precincts_2020_house_r
                     ]

election_build_var = ["PRE20D",
                      "PRE20R",
                      "SEN20D",
                      "SEN20R",
                      "HOU20D",
                      "HOU20R"
                      ]

for n in range(6):    
    election_build = election_build_df[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var[n])]
    election_build.applymap(int)
    if (n == 0):
        df_elections = election_build            
    else:
        df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')


#--- 2018

## TO DO: Group results by precinct ID to sum up all votes regardless of candidate

precincts_2018 = pd.read_excel("Data/Elections/2018_general_precincts.xlsx",
                               engine="openpyxl")

#elections_2018 = sorted(precincts_2018["Office/Issue/Judgeship"].unique())

precincts_2018_house_d = precincts_2018.loc[(precincts_2018["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2018["Party"] == "Democratic Party")]  

precincts_2018_house_r = precincts_2018.loc[(precincts_2018["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2018["Party"] == "Republican Party")]  


election_build_df = [precincts_2018_house_d,
                     precincts_2018_house_r
                     ]

election_build_var = ["HOU18D",
                      "HOU18R"
                      ]

for n in range(2):    
    election_build = election_build_df[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var[n])]
    election_build.applymap(int)
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2016

## TO DO: Group results by precinct ID to sum up all votes regardless of candidate
## PRECID coming in as str

precincts_2016 = pd.read_excel("Data/Elections/2016_general_precincts.xlsx",
                               engine="openpyxl")
   
#precincts_2020.info()
#print(precincts_2020["Office/Issue/Judgeship"].unique())

precincts_2016_pres_d = precincts_2016[(precincts_2016["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2016["Party"] == "Democratic Party")]  

precincts_2016_pres_r = precincts_2016[(precincts_2016["Office/Issue/Judgeship"] == "President/Vice President") 
                                       & (precincts_2016["Party"] == "Republican Party")]  


precincts_2016_senate_d = precincts_2016[(precincts_2016["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2016["Party"] == "Democratic Party")]  
precincts_2016_senate_r = precincts_2016[(precincts_2016["Office/Issue/Judgeship"] == "United States Senator")
                                         & (precincts_2016["Party"] == "Republican Party")]  


precincts_2016_house_d = precincts_2016.loc[(precincts_2016["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2016["Party"] == "Democratic Party")]  
precincts_2016_house_r = precincts_2016.loc[(precincts_2016["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                           & (precincts_2016["Party"] == "Republican Party")]  




