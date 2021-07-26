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

plt.style.use('seaborn-whitegrid')

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
num_districts=7
pop_col="TOTPOP"
num_elections=2

newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)
os.makedirs(os.path.dirname(newdir), exist_ok=True)

# Visualize districts for existing plans
uf.plot_district_map(df, df['CD116FP'].to_dict(), "Current Congressional District Map")
#uf.plot_district_map(df, df['SLDUST'].to_dict(), "Current State Senate District Map")
#uf.plot_district_map(df, df['SLDLST'].to_dict(), "Current State House District Map")


#--- DATA CLEANING

#--- GENERIC UPDATERS

updater = {
    "population": updaters.Tally("TOTPOP", alias="population"), 
    "cut_edges": cut_edges
            }

#--- ELECTION UPDATERS

election_names=[
    "2018_REG_VOTERS", 
    "2018_HOUSE" 
    ]

election_columns=[
    ["REG18D", "REG18R"], 
    ["USH18D", "USH18R"] 
    ]

elections = [
    Election(
        election_names[i], #Name of election
        {"First": election_columns[i][0], "Second": election_columns[i][1]},
    )
    for i in range(num_elections)
]

election_updaters = {election.name: election for election in elections}
updater.update(election_updaters)


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

# --- PARTITION

#--- STARTING PLAN STATS

# --- PROPOSAL

#--- CREATE CONSTRAINTS

#--- ACCEPTANCE FUNCTIONS

#--- MCMC CHAINS

#--- RUN CHAINS

#--- BUILD VISUALIZATIONS