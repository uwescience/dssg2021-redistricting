
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

#df = gpd.read_file("Data/co_precincts.shp") #MGGG Shapefile

#-- CLEANING FUNCTIONS

def election_df(precinct_df, election_name):
    """
    Parameters
    ----------
    precinct_df: DataFrame
        Dataframe imported from Secretary of State general_precinct level file
        
    election_name_list : String 
        Input desired election name under "Office/Issue/Judgeship" in the raw data from Secretary of State 

    Returns
    -------
    A tuple of Democratic and Republican votes for election_name

    """
    #precinct_df.info()
    
    precinct_df_trim = precinct_df.groupby(['Precinct', 'Office/Issue/Judgeship', 'Party'], as_index=False).sum()
    precinct_df_trim = precinct_df_trim[(precinct_df_trim["Party"] == "Republican Party") 
                                        | (precinct_df_trim["Party"] == "Democratic Party")]
    
    if (election_name == "United States Representative"):
        election_name_d = precinct_df_trim.loc[(precinct_df_trim["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                                 & (precinct_df_trim["Party"] == "Democratic Party")]  
        election_name_r = precinct_df_trim.loc[(precinct_df_trim["Office/Issue/Judgeship"].str.startswith("United States Representative"))
                                                 & (precinct_df_trim["Party"] == "Republican Party")]  
    else:
        election_name_d = precinct_df_trim[(precinct_df_trim["Office/Issue/Judgeship"] == election_name) 
                                           & (precinct_df_trim["Party"] == "Democratic Party")]
        election_name_r = precinct_df_trim[(precinct_df_trim["Office/Issue/Judgeship"] == election_name) 
                                           & (precinct_df_trim["Party"] == "Republican Party")]  
    
    return election_name_d, election_name_r

def election_build(election_build_df, election_build_var):
   
    eb_count = len(election_build_df)
    elections_all = pd.DataFrame(index=range(3205))
    
    for n in range(eb_count):    
        election_build = election_build_df[n].loc[:, ["Precinct", "Candidate Votes"]]
        election_build.columns = ["PRECID", str(election_build_var[n])]
        election_build.applymap(int)
        elections_all = elections_all.join(election_build.set_index('PRECID'), on='PRECID')
            
    return elections_all
            
#--- 2020

precincts_2020 = pd.read_excel("Data/Elections/2020_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2020_pres_d = election_df(precincts_2020, "President/Vice President")[0]
precincts_2020_pres_r = election_df(precincts_2020, "President/Vice President")[1]

precincts_2020_senate_d = election_df(precincts_2020, "United States Senator")[0]
precincts_2020_senate_r = election_df(precincts_2020, "United States Senator")[1]

precincts_2020_house_d = election_df(precincts_2020, "United States Representative")[0]
precincts_2020_house_r = election_df(precincts_2020, "United States Representative")[1]

election_build_df_2020 = [precincts_2020_pres_d,
                          precincts_2020_pres_r,
                          precincts_2020_senate_d,
                          precincts_2020_senate_r,
                          precincts_2020_house_d,
                          precincts_2020_house_r
                          ]

election_build_var_2020 = ["PRE20D",
                           "PRE20R",
                           "SEN20D",
                           "SEN20R",
                           "HOU20D",
                           "HOU20R"
                           ]

df_elections = precincts_2020_pres_d["Precinct"].to_frame()
df_elections = elections_all.rename({'Precinct':'PRECID'}, axis=1)

for n in range(6):    
    election_build = election_build_df_2020[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2020[n])]
    election_build.applymap(int)
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2018

precincts_2018 = pd.read_excel("Data/Elections/2018_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2018_house_d = election_df(precincts_2018, "United States Representative")[0]
precincts_2018_house_r = election_df(precincts_2018, "United States Representative")[1]

election_build_df_2018 = [precincts_2018_house_d,
                     precincts_2018_house_r
                     ]

election_build_var_2018 = ["HOU18D",
                      "HOU18R"
                      ]
for n in range(2):    
    election_build = election_build_df_2018[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2018[n])]
    election_build.applymap(int)
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2016

precincts_2016 = pd.read_excel("Data/Elections/2016_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2016_pres_d = election_df(precincts_2016, "President/Vice President")[0]
precincts_2016_pres_r = election_df(precincts_2016, "President/Vice President")[1]

precincts_2016_senate_d = election_df(precincts_2016, "United States Senator")[0]
precincts_2016_senate_r = election_df(precincts_2016, "United States Senator")[1]

precincts_2016_house_d = election_df(precincts_2016, "United States Representative")[0]
precincts_2016_house_r = election_df(precincts_2016, "United States Representative")[1]


election_build_df_2016 = [precincts_2016_pres_d,
                          precincts_2016_pres_r,
                          precincts_2016_senate_d,
                          precincts_2016_senate_r,
                          precincts_2016_house_d,
                          precincts_2016_house_r
                          ]

election_build_var_2016 = ["PRE16D",
                           "PRE16R",
                           "SEN16D",
                           "SEN16R",
                           "HOU16D",
                           "HOU16R"
                           ]
for n in range(6):    
    election_build = election_build_df_2016[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2016[n])]
    election_build.applymap(int)
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2014

precincts_2014 = pd.read_excel("Data/Elections/2014_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2014_senate_d = election_df(precincts_2014, "United States Senator")[0]
precincts_2014_senate_r = election_df(precincts_2014, "United States Senator")[1]

precincts_2014_house_d = election_df(precincts_2014, "United States Representative")[0]
precincts_2014_house_r = election_df(precincts_2014, "United States Representative")[1]


election_build_df_2014 = [precincts_2014_senate_d,
                          precincts_2014_senate_r,
                          precincts_2014_house_d,
                          precincts_2014_house_r
                          ]

election_build_var_2014 = ["SEN14D",
                           "SEN14R",
                           "HOU14D",
                           "HOU14R"
                           ]

for n in range(4):    
    election_build = election_build_df_2014[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2014[n])]
    election_build.applymap(int)
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2012

precincts_2012 = pd.read_excel("Data/Elections/2012_general_precincts.xlsx",
                               engine="openpyxl")

#--- 2010

precincts_2010 = pd.read_excel("Data/Elections/2010_general_precincts.xlsx",
                               engine="openpyxl")