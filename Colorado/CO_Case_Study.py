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
from gerrychain.metrics import efficiency_gap, mean_median, polsby_popper, wasted_votes
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
#num_elections=

#Make an output directory to dump files in
newdir = "./Outputs/"+state_abbr+housen+"_Precincts/"
print(newdir)
os.makedirs(os.path.dirname(newdir), exist_ok=True)

# Visualize districts for existing plans
uf.plot_district_map(df, df['CD116FP'].to_dict(), "Current Congressional District Map")
uf.plot_district_map(df, df['SLDUST'].to_dict(), "Current State Senate District Map")
uf.plot_district_map(df, df['SLDLST'].to_dict(), "Current State House District Map")

