
"""
02. Exploratory Data Analysis: Metrics Development and Modeling Tests

Description: This file is dedicated to state-specific metrics development and appropriate
tests to refine modeling decisions.

https://redistricting.colorado.gov/content/redistricting-laws Congressional Language (Section 44.3):
“Thereafter, the commission shall, to the extent possible, maximize the number of politically competitive districts…
‘competitive’ means having a reasonable potential for the party affiliation of the district’s representative to change
at least once between federal decennial censuses. Competitiveness may be measured by factors such as a proposed district’s
past election results,
a proposed district’s political party registration data,
and evidence-based analyses of proposed districts.”

"""

import os
import sys
import pandas as pd
import geopandas as gpd
from gerrychain import Graph

import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
import utility_functions as uf

try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
       
#--- IMPORT DIFFERENT DATA TO EXPLORE

#Metric Geometry and Gerrymandering Group
graph_mggg = Graph.from_json("Data/co_precincts.json")
df_mggg = gpd.read_file("Data/co_precincts.shp")

uf.plot_district_map(df_mggg, df_mggg['CD116FP'].to_dict(), "Current Congressional District Map")

nx.draw(graph_mggg,pos = {node:(graph_mggg.nodes[node]["C_X"],graph_mggg.nodes[node]["C_Y"]) 
                     for node in graph_mggg.nodes()},node_color=[graph_mggg.nodes[node]["CD116FP"] 
                                                            for node in graph_mggg.nodes()],node_size=10,cmap='tab20')
#Redistricting Data Hub
df_rdh_2016 = gpd.read_file("Data/RDH/co_vest_16/co_vest_16.shp")                      
df_rdh_2018 = gpd.read_file("Data/RDH/co_vest_18/co_vest_18.shp")
df_rdh_2020 = gpd.read_file("Data/RDH/co_vest_20/co_vest_20.shp")

#Secretary of State Election Data (2004-2020)

df_elections_2004_2010 = pd.read_csv("Data/co_elections_2004_2010.csv")
df_elections_2012_2020 = pd.read_csv("Data/co_elections_2012_2020.csv")

#--- Political Competitiveness

"""
- Different vote bands, 5%, 10%, and 15%
- Win margin between the two major political parties
- Probability the party affiliation of the district’s representative to change
at least once between federal decennial censuses

"""


                                                                 
