#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorado Case Study
"""

import os
import sys
import re

import csv
import json
import random
import math
import numpy as np

from functools import partial
import pandas as pd
import geopandas as gpd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept, #always_accept, acceptance functions
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes, partisan_bias
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part, bipartition_tree_random

sys.path.insert(0, os.getenv("REDISTRICTING_HOME"))
import utility_functions as uf

#plt.style.use('seaborn-whitegrid')

#--- IMPORT DATA

try:
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
except OSError:
    os.mkdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))
    os.chdir(os.path.join(os.getenv("REDISTRICTING_HOME", default=""),
                          "Colorado"))

graph = Graph.from_json("Data/co_precincts.json")
df = gpd.read_file("Data/co_precincts.shp")

#--- CREATE SHORTCUTS

state_abbr="CO"
housen="CON"
num_districts=8
pop_col="TOTPOP"
num_elections=2

newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)
os.makedirs(os.path.dirname(newdir), exist_ok=True)

# Visualize districts for existing plans
uf.plot_district_map(df, df['CD116FP'].to_dict(), "2012 Congressional District Map")

#--- DATA CLEANING


df["POC_VAP"] = (df["HVAP"] + df["BVAP"] + df["AMINVAP"] + df["ASIANVAP"] 
                 + df["NHPIVAP"] + df["OTHERVAP"] + df["OTHERVAP"])

print(df["WVAP"].sum()/df["VAP"].sum())
#0.7388 White VAP across the state

print(df["POC_VAP"].sum() / df["VAP"].sum())
#0.246 POC VAP across the state

#uf.plot_district_map(df_mggg, df_mggg['POC_VAP_PCT'].to_dict(), "Distribution of POC VAP")

#plt.hist(df_mggg['POC_VAP_PCT'])
#plt.title("Precinct-level Distribution of CO POC Voting Age Population")

for node in graph.nodes():
    graph.nodes[node]["POC_VAP"] = (graph.nodes[node]["HVAP"] + graph.nodes[node]["BVAP"] 
                                    + graph.nodes[node]["AMINVAP"] + graph.nodes[node]["ASIANVAP"] 
                                    + graph.nodes[node]["NHPIVAP"] + graph.nodes[node]["OTHERVAP"] 
                                    + graph.nodes[node]["OTHERVAP"])
    graph.nodes[node]["nPOC_VAP"] = graph.nodes[node]["VAP"] - graph.nodes[node]["POC_VAP"]

#--- GENERIC UPDATERS

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), 
    "cut_edges": cut_edges
            }

#--- ELECTION UPDATERS

election_names=[
    "POC_VAP", 
    "USH18"
    ]

election_columns=[
    ["POC_VAP", "nPOC_VAP"], 
    ["USH18D", "USH18R"] 
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

totpop = df.TOTPOP.sum()


#--- ENACTED PLAN BASELINE STATS

partition_2012 = Partition(graph,
                           df["CD116FP"],
                           updater)

baseline_partisan_stats_names=["mean_median_value",
                               "efficiency_gap_value",
                               "partisan_bias_value",
                               "wasted_votes_value",
                               "dem_seat_wins"
                               ]

df_baseline_partisan=pd.DataFrame(index=election_names,
                         columns=baseline_partisan_stats_names)

for n in range(len(election_names)):
    df_baseline_partisan.iloc[n,0] = mean_median(partition_2012[election_names[n]])
    df_baseline_partisan.iloc[n,1] = efficiency_gap(partition_2012[election_names[n]])
    df_baseline_partisan.iloc[n,2] = partisan_bias(partition_2012[election_names[n]])

    df_baseline_partisan.iloc[n,3] = wasted_votes(df[election_columns[n][0]].sum(), 
                                         df[election_columns[n][1]].sum())
    df_baseline_partisan.iloc[n,4] = partition_2012[election_names[n]].wins("First")


df_enacted_map = pd.DataFrame(index=["2010 Cycle"],
                              columns=["cut_edges_value",
                                       "ideal_population"])

df_enacted_map.loc[:, "cut_edges_value"] = len(partition_2012["cut_edges"])
df_enacted_map.loc[:, "ideal_population"] = sum(partition_2012["population"].values()) / len(partition_2012)


#--- STARTING PLAN (SEED PLAN)

plan_seed = recursive_tree_part(graph, #graph object
                                range(num_districts), #How many districts
                                totpop/num_districts, #population target
                                "TOTPOP", #population column, variable name
                                .01, #epsilon value
                                1)

uf.plot_district_map(df, 
                     plan_seed, 
                     "Random Seed Plan") 
    
# --- PARTITION (SEED PLAN)

partition_seed = Partition(graph,
                           plan_seed, 
                           updater)
    
  

#--- STATS (SEED PLAN)

stats_seed = uf.export_election_metrics_per_partition(partition_seed)
  

# --- PROPOSAL

#--- CREATE CONSTRAINTS

#--- ACCEPTANCE FUNCTIONS

#--- MCMC CHAINS

#--- RUN CHAINS

#--- BUILD VISUALIZATIONS