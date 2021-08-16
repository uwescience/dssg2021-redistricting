
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
import random
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
from gerrychain.updaters import cut_edges, county_splits
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

uf.plot_district_map(df_mggg,
                     df_mggg['CD116FP'].to_dict(),
                     title="Current Congressional District Map",
                     map_colors="Set2")

nx.draw(graph_mggg,pos = {node:(graph_mggg.nodes[node]["C_X"],graph_mggg.nodes[node]["C_Y"])
                     for node in graph_mggg.nodes()},node_color=[graph_mggg.nodes[node]["CD116FP"]
                                                            for node in graph_mggg.nodes()],node_size=10,cmap='tab20')
#Redistricting Data Hub
"""
df_rdh_2016 = gpd.read_file("Data/RDH/co_vest_16/co_vest_16.shp")
df_rdh_2018 = gpd.read_file("Data/RDH/co_vest_18/co_vest_18.shp")
df_rdh_2020 = gpd.read_file("Data/RDH/co_vest_20/co_vest_20.shp")

#Secretary of State Election Data (2004-2020)

df_elections_2004_2010 = pd.read_csv("Data/co_elections_2004_2010.csv")
df_elections_2012_2020 = pd.read_csv("Data/co_elections_2012_2020.csv")
"""
#--- DATA WORK

#POC Voting age population for Colorado in 2018

df_mggg["POC_VAP"] = (df_mggg["HVAP"] + df_mggg["BVAP"] + df_mggg["AMINVAP"] + df_mggg["ASIANVAP"]
 + df_mggg["NHPIVAP"] + df_mggg["OTHERVAP"] + df_mggg["OTHERVAP"])

df_mggg["POC_VAP_PCT"] = df_mggg["POC_VAP"]/df_mggg["VAP"]

print(df_mggg["WVAP"].sum()/df_mggg["VAP"].sum())
#0.7388 White VAP across the state

print(df_mggg["POC_VAP"].sum() / df_mggg["VAP"].sum())
#0.246 POC VAP across the state

uf.plot_district_map(df_mggg, df_mggg['POC_VAP_PCT'].to_dict(), "Distribution of POC VAP")

plt.hist(df_mggg['POC_VAP_PCT'])
plt.title("Precinct-level Distribution of CO POC Voting Age Population")

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

Another option is to generate a collection of neutral seed plans made up of 7 districts
organized by Democratic share of votes to indicate competitiveness

PROCESS: I generated 10 seed plans and built a dataframe of their Democrat vote totals
and Democrat seat wins using the 2018 US House election data

DECISION: The Colorado analysis will use the ReCom proposal, which means the starting plan
is not as important compared to a Flip proposal. This step wasn't very necessary, but
explores
"""

state_abbr="CO"
housen="CON"
num_districts=7
pop_col="TOTPOP"
num_elections= 7

def num_splits(partition, df=df_mggg):
    df["current"] = df.index.map(partition.assignment)
    return sum(df.groupby('COUNTYFP')['current'].nunique() > 1)

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
    "PP":polsby_popper,
    "count_splits": num_splits
            }

election_names=[
    "POC_VAP",
    "USH18",
    "GOV18",
    "AG18",
    "SOS18",
    "TRE18",
    "REG18",
    ]

election_columns=[
    ["POC_VAP", "nPOC_VAP"],
    ["USH18D", "USH18R"],
    ["GOV18D", "GOV18R"],
    ["AG18D", "AG18R"],
    ["SOS18D", "SOS18R"],
    ["TRE18D", "TRE18R"],
    ["REG18D", "REG18R"]
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

plan_2012_stats = uf.export_election_metrics_per_partition(plan_2012)

#Tidy df
plan_2012_stats.iloc[0, 1] = float('NaN')
plan_2012_stats.iloc[0, 2] = float('NaN')
plan_2012_stats = plan_2012_stats.rename({'wins':'dem_wins'}, axis=1)

plan_2012_names = plan_2012_stats.index.values
plan_2012_comp=[]
for n in range(7):
    plan_2012_comp.append(sum([.45<x<.55 for x in plan_2012[plan_2012_names[n]].percents('First')]))

plan_2012_stats['comp_dist'] = np.array(plan_2012_comp)
plan_2012_stats.iloc[0, 4] = float('NaN')

seeds_stats = pd.DataFrame(columns=[],
                           index=range(7))

seeds_county=[]
seeds_comp=[]
seeds_wasted=[]
#Running multiple seeds to note comp districts and county splits of starting plans
for n in range(50):
    plan_seed = recursive_tree_part(graph_mggg, #graph object
                                    range(num_districts), #How many districts
                                    totpop/num_districts, #population target
                                    "TOTPOP", #population column, variable name
                                    .01, #epsilon value
                                    1)
    #uf.plot_district_map(df_mggg, plan_seed, "Seed" + str(n))

    partition_seed = Partition(graph_mggg,
                           plan_seed,
                           updater)

    stats_df = uf.export_election_metrics_per_partition(partition_seed)

    stats_df.iloc[0, 1] = float('NaN') #efficiency_gap
    stats_df.iloc[0, 2] = float('NaN') #mean_median

    seeds_county.append(partition_seed["count_splits"])
    seeds_comp.append(sum([.45<x<.55 for x in partition_seed['USH18'].percents('First')]))


plt.hist(seeds_county)
plt.hist(seeds_comp)

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

def competitive_county_accept(partition):
    new_score = 0
    old_score = 0
    for i in range(7):
        if .45 < partition.parent['USH18'].percents("First")[i] <.55:
            old_score += 1

        if .45 < partition['USH18'].percents("First")[i] <.55:
            new_score += 1

    if (new_score >= old_score) & (partition["count_splits"] < partition.parent["count_splits"]):
        return True
    elif (new_score >= old_score)  & (random.random() < .05):
        return True
    elif (partition["count_splits"] < partition.parent["count_splits"]) & (random.random() < .05):
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

cut_edges_constraint = constraints.UpperBound(number_of_cut_edges, 250)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 1.5 * len(partition_seed["cut_edges"])
)


#---
