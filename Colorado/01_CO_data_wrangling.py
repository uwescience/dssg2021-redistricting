
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
"""
def election_build(election_build_df, election_build_var):
   
    eb_count = len(election_build_df)
    elections_all = pd.DataFrame(index=range(3205))
    
    for n in range(eb_count):    
        election_build = election_build_df[n].loc[:, ["Precinct", "Candidate Votes"]]
        election_build.columns = ["PRECID", str(election_build_var[n])]
        election_build.applymap(int)
        elections_all = elections_all.join(election_build.set_index('PRECID'), on='PRECID')
            
    return elections_all
"""     

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
df_elections = df_elections.rename({'Precinct':'PRECID'}, axis=1)
df_elections = df_elections.astype("string")

for n in range(6):    
    election_build = election_build_df_2020[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2020[n])]
    election_build = election_build.astype("string")
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
    election_build = election_build.astype("string")
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
    election_build = election_build.astype("string")
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
    election_build = election_build.astype("string")
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

#--- 2012

precincts_2012 = pd.read_excel("Data/Elections/2012_general_precincts.xlsx",
                               engine="openpyxl")


precincts_2012_pres_d = election_df(precincts_2012, "President/Vice President")[0]
precincts_2012_pres_r = election_df(precincts_2012, "President/Vice President")[1]

precincts_2012_house_d = election_df(precincts_2012, "United States Representative")[0]
precincts_2012_house_r = election_df(precincts_2012, "United States Representative")[1]


election_build_df_2012 = [precincts_2012_pres_d,
                          precincts_2012_pres_r,
                          precincts_2012_house_d,
                          precincts_2012_house_r
                          ]

election_build_var_2012 = ["PRE12D",
                           "PRE12R",
                           "HOU12D",
                           "HOU12R"
                           ]
for n in range(4):    
    election_build = election_build_df_2012[n].loc[:, ["Precinct", "Candidate Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2012[n])]
    election_build = election_build.astype("string")
    df_elections = df_elections.join(election_build.set_index('PRECID'), on='PRECID')

elections_2010_2020 = df_elections

### Handcleaning SOS files from 2004-2010 due to differences in variable names and format for each year 

#--- 2010

precincts_2010 = pd.read_excel("Data/Elections/2010_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2010_trim = precincts_2010[~precincts_2010["Precinct"].astype(str).str.startswith("P")] #Removing provisional

precincts_2010_trim = precincts_2010_trim.groupby(['Precinct', 'Office/Question', 'Party'], as_index=False).sum()
precincts_2010_trim = precincts_2010_trim[(precincts_2010_trim["Party"] == "DEM") 
                                          | (precincts_2010_trim["Party"] == "REP")]

precincts_2010_senate_d = precincts_2010_trim[(precincts_2010_trim["Office/Question"] == "UNITED STATES SENATOR")
                                              & (precincts_2010_trim["Party"] == "DEM")]  
precincts_2010_senate_r = precincts_2010_trim[(precincts_2010_trim["Office/Question"] == "UNITED STATES SENATOR")
                                              & (precincts_2010_trim["Party"] == "REP")]  

precincts_2010_house_d = precincts_2010_trim.loc[(precincts_2010_trim["Office/Question"].str.startswith("REPRESENTATIVE TO THE 112th UNITED STATES CONGRESS"))
                                                 & (precincts_2010_trim["Party"] == "DEM")]  
precincts_2010_house_r = precincts_2010_trim.loc[(precincts_2010_trim["Office/Question"].str.startswith("REPRESENTATIVE TO THE 112th UNITED STATES CONGRESS"))
                                           & (precincts_2010_trim["Party"] == "REP")]  

election_build_df_2010 = [precincts_2010_senate_d,
                          precincts_2010_senate_r,
                          precincts_2010_house_d,
                          precincts_2010_house_r
                          ]

election_build_var_2010 = ["SEN10D",
                           "SEN10R",
                           "HOU10D",
                           "HOU10R"
                           ]

elections_2004_2010 = precincts_2010_senate_d["Precinct"].to_frame()
elections_2004_2010 = elections_2004_2010.rename({'Precinct':'PRECID'}, axis=1)
elections_2004_2010 = elections_2004_2010.astype("string")
elections_2004_2010 = elections_2004_2010.drop_duplicates(subset=['PRECID'])

for n in range(4):    
    election_build = election_build_df_2010[n].loc[:, ["Precinct", "Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2010[n])]
    election_build = election_build.astype("string")
    elections_2004_2010 = elections_2004_2010.join(election_build.set_index('PRECID'), on='PRECID')

"""
precincts_2010_house_d[['Office/Question', 'CONDIS_2010']] = precincts_2010_house_d['Office/Question'].str.split(' - ', expand=True)

condist_2010 = precincts_2010_house_d.loc[:, ["Precinct", "CONDIS_2010"]]
condist_2010.columns = ["PRECID", "CONDIS_2010"]
condist_2010 = condist_2010.astype("string")
elections_2004_2010 = elections_2004_2010.join(condist_2010.set_index('PRECID'), on='PRECID')
"""
#--- 2008 

precincts_2008 = pd.read_excel("Data/Elections/2008_general_precincts.xlsx",
                               engine="openpyxl")

precincts_2008_trim = precincts_2008[precincts_2008["Precinct"].astype(str).str.startswith("P")] #Removing provisional
precincts_2008_trim = precincts_2008.groupby(['Precinct', 'Office/Ballot Issue', 'Party'], as_index=False).sum()
precincts_2008_trim = precincts_2008_trim.astype("string")

precincts_2008_trim["Party"] = precincts_2008_trim["Party"].str.strip()
precincts_2008_trim["Office/Ballot Issue"] = precincts_2008_trim["Office/Ballot Issue"].str.strip()
precincts_2008_trim["Precinct"] = precincts_2008_trim["Precinct"].str.strip()

precincts_2008_trim = precincts_2008_trim[(precincts_2008_trim["Party"] == "Democrat") 
                                          | (precincts_2008_trim["Party"] == "Republican")]

precincts_2008_pres_d = precincts_2008_trim[(precincts_2008_trim["Office/Ballot Issue"] == "PRESIDENTIAL ELECTORS/ PRESIDENTIAL ELECTORS (VICE)")
                                            & (precincts_2008_trim["Party"] == "Democrat")]  
precincts_2008_pres_r = precincts_2008_trim[(precincts_2008_trim["Office/Ballot Issue"] == "PRESIDENTIAL ELECTORS/ PRESIDENTIAL ELECTORS (VICE)")
                                            & (precincts_2008_trim["Party"] == "Republican")]  

precincts_2008_senate_d = precincts_2008_trim[(precincts_2008_trim["Office/Ballot Issue"] == "UNITED STATES SENATOR")
                                              & (precincts_2008_trim["Party"] == "Democrat")]  
precincts_2008_senate_r = precincts_2008_trim[(precincts_2008_trim["Office/Ballot Issue"] == "UNITED STATES SENATOR")
                                              & (precincts_2008_trim["Party"] == "Republican")]  


precincts_2008_house_d = precincts_2008_trim.loc[(precincts_2008_trim["Office/Ballot Issue"].str.startswith("REPRESENTATIVE TO THE 111th UNITED STATES CONGRESS"))
                                                 & (precincts_2008_trim["Party"] == "Democrat")]  

precincts_2008_house_r = precincts_2008_trim.loc[(precincts_2008_trim["Office/Ballot Issue"].str.startswith("REPRESENTATIVE TO THE 111th UNITED STATES CONGRESS"))
                                           & (precincts_2008_trim["Party"] == "Republican")]  


election_build_df_2008 = [precincts_2008_pres_d,
                          precincts_2008_pres_r,
                          precincts_2008_senate_d,
                          precincts_2008_senate_r,
                          precincts_2008_house_d,
                          precincts_2008_house_r
                          ]

election_build_var_2008 = ["PRES08D",
                           "PRES08R",
                           "SEN08D",
                           "SEN08R",
                           "HOU08D",
                           "HOU08R"
                           ]

for n in range(6):    
    election_build = election_build_df_2008[n].loc[:, ["Precinct", "Votes"]]
    election_build.columns = ["PRECID", str(election_build_var_2008[n])]
    election_build = election_build.astype("string")
    elections_2004_2010 = elections_2004_2010.join(election_build.set_index('PRECID'), on='PRECID')

"""
#Delim congressional district Number 
#precincts_2008_house_d[['Office/Ballot Issue', 'CONDIS_2008']] = precincts_2008_house_d['Office/Ballot Issue'].str.split(' - ', expand=True)

condist_2008 = precincts_2008_house_d.loc[:, ["Precinct", "CONDIS_2008"]]
condist_2008.columns = ["PRECID", "CONDIS_2008"]
condist_2008 = condist_2008.astype("string")
elections_2004_2010 = elections_2004_2010.join(condist_2008.set_index('PRECID'), on='PRECID')
"""

#--- 2006

precincts_2006 = pd.read_excel("Data/Elections/2006_general_precincts.xlsx",
                               engine="openpyxl")

#--- 2004

precincts_2004 = pd.read_excel("Data/Elections/2004_general_precincts.xlsx",
                               engine="openpyxl")




