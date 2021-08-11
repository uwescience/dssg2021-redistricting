"""
Texas model analyzing the addition of 2 new districts in the 2021 redistricting cycle
"""

import random
a = random.randint(0,10000000000)
import networkx as nx
from gerrychain.random import random
random.seed(a)
import csv
import os
import shutil
from functools import partial
import json
import math
import numpy as np
import geopandas as gpd
import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges
from gerrychain.updaters import *
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally
from gerrychain import GeographicPartition
from scipy.spatial import ConvexHull
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from gerrychain.constraints import single_flip_contiguous, Validator
import collections
from enum import Enum
import re
import operator
import time
import heapq
import scipy
from scipy import stats
import sys
from functools import partial
from run_functions import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
cand_pref_all_draws_outcomes, precompute_state_weights, compute_district_weights
from ast import literal_eval


#user input parameters######################################
total_steps = 1000
pop_tol = .01 #U.S. Congress (deviation from ideal district population)
run_name = 'Texas_neutral_run'
start_map = 'new_seed' #CD, 'Seed_Demo', or "new_seed"
effectiveness_cutoff = .6
ensemble_inclusion = False
ensemble_inclusion_demo = False
record_statewide_modes = True
record_district_mode = True
model_mode = 'statewide' #'district', 'equal', 'statewide'

store_interval = 100  #how many Markov chain steps between data storage


#fixed parameters#################################################
num_districts = 38 #38 Congressional districts
enacted_black = 4 #number of districts in enacted map with Black effectiveness> 60%
enacted_hisp = 8 #number of districts in enacted map with Latino effectiveness > 60%
enacted_distinct = 11 #number of districts in enacted map with B > 60% or L > 60% or both
plot_path = 'TX_VTDs/TX_VTDs.shp'  #for shapefile

DIR = ''
if not os.path.exists(DIR + 'outputs'):
    os.mkdir(DIR + 'outputs')