
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

"“Competitive elections for members of the United States House of Representatives 
provide voters with a meaningful choice among candidates, 
promote a healthy democracy, 
help ensure that constituents receive fair and effective representation, and 
contribute to the political well-being of key communities of interest and political subdivisions”"


"""

import os
import sys
import pandas as pd
import geopandas as gpd
from gerrychain import Graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept, 
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part, bipartition_tree_random

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

#--- DATA WORK

#POC Voting age population for Colorado in 2018

df_mggg["POC_VAP"] = (df_mggg["HVAP"] + df_mggg["BVAP"] + df_mggg["AMINVAP"] + df_mggg["ASIANVAP"] 
 + df_mggg["NHPIVAP"] + df_mggg["OTHERVAP"] + df_mggg["OTHERVAP"])

#df_mggg["POC_VAP_PCT"] = df_mggg["POC_VAP"]/df_mggg["VAP"]

print(df_mggg["WVAP"].sum()/df_mggg["VAP"].sum())
#0.7388 White VAP across the state

print(df_mggg["POC_VAP"].sum() / df_mggg["VAP"].sum())
#0.246 POC VAP across the state

#uf.plot_district_map(df_mggg, df_mggg['POC_VAP_PCT'].to_dict(), "Distribution of POC VAP")

#plt.hist(df_mggg['POC_VAP_PCT'])
#plt.title("Precinct-level Distribution of CO POC Voting Age Population")

for node in graph_mggg.nodes():
    graph_mggg.nodes[node]["POC_VAP"] = (graph_mggg.nodes[node]["HVAP"] + graph_mggg.nodes[node]["BVAP"] 
                                    + graph_mggg.nodes[node]["AMINVAP"] + graph_mggg.nodes[node]["ASIANVAP"] 
                                    + graph_mggg.nodes[node]["NHPIVAP"] + graph_mggg.nodes[node]["OTHERVAP"] 
                                    + graph_mggg.nodes[node]["OTHERVAP"])
    graph_mggg.nodes[node]["nPOC_VAP"] = graph_mggg.nodes[node]["VAP"] - graph_mggg.nodes[node]["POC_VAP"]

#--- SEED PLAN/STARTING PLAN

"""
CONTEXT: Need to decide the appropriate starting plan to begin the ensemble
One option is the 2012 enacted plan -- however it has been claimed to be a Democrat
gerrymandered map by Republicans. 

Another option is to generate a collection of neutral seed plans made up of 8 districts
organized by Democratic share of votes to indicate competitiveness

PROCESS: I generated 10 seed plans and built a dataframe of their Democrat vote totals
and Democrat seat wins using the 2018 US House election data 
    
DECISION: The Colorado analysis will use the ReCom proposal, which means the starting plan
is not as important compared to a Flip proposal. This step wasn't very necessary, but 
explores 
"""

state_abbr="CO"
housen="CON"
num_districts=8
pop_col="TOTPOP"
num_elections= 2

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), 
    "cut_edges": cut_edges
            }

election_names=[
    "POC_VAP", 
    "USH18", 

    ]

election_columns=[
    ["POC_VAP", "nPOC_VAP"], 
    ["USH18D", "USH18R"], 
    ]

elections = [
    Election(
        election_names[i], 
        {"First": election_columns[i][0], "Second": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}
updater.update(election_updaters)

totpop = df_mggg.TOTPOP.sum()

plan_2012 = Partition(graph_mggg,
                      df_mggg["CD116FP"],
                      updater)


seeds_stats = pd.DataFrame(columns=[],
                           index=range(8))

#Running multiple seeds to note the distribution of metrics of starting plans
for n in range(10):
    plan_seed = recursive_tree_part(graph_mggg, #graph object
                                    range(num_districts), #How many districts
                                    totpop/num_districts, #population target
                                    "TOTPOP", #population column, variable name
                                    .01, #epsilon value
                                    1)
    uf.plot_district_map(df_mggg, 
                         plan_seed, 
                         "Seed" + str(n)) 
    
    partition_seed = Partition(graph_mggg,
                           plan_seed, 
                           updater)
    
    stats_df = uf.export_election_metrics_per_partition(partition_seed)
    
    seed_loop = pd.DataFrame(columns=["USH18D",
                                      "dem_seats"],
                           index=range(8))
    
    for y in range(8):
        seed_loop["USH18D"][y] = stats_df.percent["USH18"][y]
        seed_loop["dem_seats"][y] = stats_df.wins["USH18"]

    seed_loop = seed_loop.sort_values("USH18D")
    seed_loop = seed_loop.reset_index()
    seed_loop=seed_loop.rename(columns={"USH18D": "USH18D_seed" + str(n), 
                                        "dem_seats": "dem_seats_seed" + str(n)})    

    seeds_stats = pd.concat([seeds_stats, seed_loop], axis=1)
    
print(seeds_stats)

#--- PROPOSAL

"""

DECISION: Given that our analysis explores different options across the entire state space of Colorado,
the ReCOM proposal is better suited for map drawing phase.


proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.01,
                   node_repeats=1,
                   method=bipartition_tree_random
)

"""


#--- ACCEPTANCE FUNCTIONS

"""
CONTEXT:

Political Competitiveness

- Vote band metrics (5, 50) 45%-55%
- Win margin between the two major political parties
- Probability the party affiliation of the district’s representative to change
at least once between federal decennial censuses
- Cook Partisan Voting Index (CPVI): a district is described as competitive 
if its CPVI value is between D+5 and R+5, 
meaning that the district’s vote is within 5% of the nationwide average.
- CVPI for Colorado U.S. Congressional Races

PROCESS: 

DECISION:
    
No comparison between answers between the two
State vs. Republican

"""

       
def competitiveness_accept(partition): 
  
    new = sum(x<.55 & x > .45 for x in partition["USH18"].percents("First"))
    
    old = sum(x<.55 & x > .45 for x in partition.parent["USH18"].percents("First")) 

    if new > old:  
        return True
    
    else:
        return False
    
def squeeze_accept(partition): 

    """
    Write a function that 
    - Sort districts by most Democratic heavy and most Republican heavy 
    - Assign a base value of competitiveness for each district
    - Run chain, accept only if districts satisfy values under or order
    """
    


                                                          
#--- CONSTRAINTS

"""
CONTEXT:
PROCESS:
DECISION:
"""


compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(initial_partition["cut_edges"])
)


#--- 
