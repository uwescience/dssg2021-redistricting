"""
This script adds the ACS 2019 population totals extracted from Redisticting Data Hub
(https://redistrictingdatahub.org/dataset/texas-block-group-acs5-data-2019/)
to the shape file provided by MGGG (https://www.dropbox.com/sh/k78n2hyixmv9xdg/AABmZG5ntMbXtX1VKThR7_t8a?dl=0)
for updated demographic info since this analysis is conducted prior to the release of the 2020 US Census Data.

Additionally, a seed plan is created using the 2019 ACS population info and 38 districts that
conforms to the 'ensemble_inclusion' constraint (minimum of 11 minority effective districts)
in 03_TX_model.py.

"""
import os
import sys

import geopandas as gpd
import pandas as pd
import maup

from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    GeographicPartition,
    accept,
    constraints,
    updaters,
    tree
)

from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph, \
                            find_balanced_edge_cuts_memoization, random_spanning_tree

from run_functions import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
cand_pref_all_draws_outcomes, precompute_state_weights, compute_district_weights

import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)

# Read in Texas shapefile created by MGGG
df = gpd.read_file("Data/TX_VTDs/TX_VTDs.shp")

# ACS block group data from redistricting hub (TX has 15,811 block groups)
block_groups = gpd.read_file("Data/tx_acs5_2019_bg/tx_acs5_2019_bg.shp")
# maup.doctor(block_groups, df)

# update projection coordinate reference system to resolve area inaccuracy issues
df = df.to_crs('epsg:3083')
block_groups = block_groups.to_crs('epsg:3083')

with maup.progress():
    block_groups['geometry'] = maup.autorepair(block_groups)
    df['geometry'] = maup.autorepair(df)


# map block groups to VTDs based on geometric intersections
# Include area_cutoff=0 to ignore any intersections with no area,
# like boundary intersections, which we do not want to include in
# our proration.
pieces = maup.intersections(block_groups, df, area_cutoff=0)

# Option 1: Weight by prorated population from blocks
# block_proj = gpd.read_file("Data/tx_b_proj_P1_2020tiger/tx_b_proj_P1_2020tiger.shp")
# block_proj = block_proj.to_crs('epsg:3083')
# block_proj['geometry'] = maup.autorepair(block_proj)
#
# with maup.progress():
#     bg2pieces = maup.assign(block_proj, pieces.reset_index())
# weights = block_proj['p20_total'].groupby(bg2pieces).sum()
# weights = maup.normalize(weights, level=0)

# Option 2: Alternative: Weight by relative area
weights = pieces.geometry.area
weights = maup.normalize(weights, level=0)

with maup.progress():
    df['TOTPOP19'] = maup.prorate(pieces, block_groups['TOTPOP19'], weights=weights)


# sanity check for Harris County
print(len(df[df.TOTPOP19.isna()]))
print(df[df.CNTY_x == 201].sum()[['TOTPOP_x', 'TOTPOP19']])


# Create a random seed plan for 38 districts that has a minimum number of minority
# effective districts (enacted_distinct = 11) to satisfy 'ensemble_inclusion' constraint
# relevant code excerpts copied from '03_TX_model.py'

# user input parameters######################################
tot_pop = 'TOTPOP19'
num_districts = 38 #38 Congressional districts in 2020
pop_tol = .01 #U.S. Congress (deviation from ideal district population)

effectiveness_cutoff = .6
record_statewide_modes = True
record_district_mode = False
model_mode = 'statewide'  # 'district', 'equal', 'statewide'

# fixed parameters#################################################
enacted_black = 4  # number of districts in enacted map with Black effectiveness> 60%
enacted_hisp = 8  # number of districts in enacted map with Latino effectiveness > 60%
enacted_distinct = 11  # number of districts in enacted map with B > 60% or L > 60% or both

##################################################################
# key column names from Texas VTD shapefile
white_pop = 'NH_WHITE'
CVAP = "1_2018"
WCVAP = "7_2018"
HCVAP = "13_2018"
BCVAP = "5_2018"  # with new CVAP codes!
geo_id = 'CNTYVTD'
C_X = "C_X"
C_Y = "C_Y"

# read files###################################################################
elec_data = pd.read_csv("TX_elections.csv")
TX_columns = list(pd.read_csv("TX_columns.csv")["Columns"])
dropped_elecs = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("ingroup_weight.csv")
cand_race_table = pd.read_csv("Candidate_Race_Party.csv")
EI_statewide = pd.read_csv("statewide_rxc_EI_preferences.csv")
prec_ei_df = pd.read_csv("prec_count_quants.csv", dtype={'CNTYVTD': 'str'})
mean_prec_counts = pd.read_csv("mean_prec_vote_counts.csv",
                               dtype={'CNTYVTD': 'str'})
logit_params = pd.read_csv('TX_logit_params.csv')

# initialize state_gdf########################################################
# reformat/re-index enacted map plans
state_gdf = df
state_gdf.columns = state_gdf.columns.str.replace("-", "_")

# replace cut-off candidate names from shapefile with full names
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index + 1] = TX_columns
state_gdf.columns = state_gdf_cols
state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis=1)

# build graph from geo_dataframe #####################################################
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids = state_gdf.centroid
c_x = centroids.x
c_y = centroids.y
for node in graph.nodes():
    graph.nodes[node]["C_X"] = c_x[node]
    graph.nodes[node]["C_Y"] = c_y[node]

# set up elections data structures ################################################
elections = list(elec_data["Election"])
elec_type = elec_data["Type"]
elec_cand_list = TX_columns

elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop=True)
elec_sets = list(set(elec_data_trunc["Election Set"]))
elections = list(elec_data_trunc["Election"])
general_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs = list(
    elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)

# this dictionary matches a specific election with the election set it belongs to
elec_set_dict = {}
for elec_set in elec_sets:
    elec_set_df = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(elec_set_df.Type, elec_set_df.Election))

elec_match_dict = dict(
    zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

# dictionary that maps an election to its candidates
# only include 2 major party candidates in generals (assumes here major party candidates are first in candidate list)
candidates = {}
for elec in elections:
    # get rid of republican candidates in primaries or runoffs (primary runoffs)
    cands = [y for y in elec_cand_list if
             elec in y and "R_" not in y.split('1')[0]] if \
        "R_" in elec[:4] or "P_" in elec[:4] else [y for y in elec_cand_list if
                                                   elec in y]

    elec_year = \
    elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]
    if elec in general_elecs:
        # assumes D and R are always first two candidates
        cands = cands[:2]
    candidates[elec] = dict(zip(list(range(len(cands))), cands))

cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
min_cand_weights_dict = {key: min_cand_weights.to_dict()[key][0] for key in
                         min_cand_weights.to_dict().keys()}

######################## pre-compute as much as possible for elections model ##############
# pre-compute election recency weights "W1" df for all model modes
elec_years = [elec_data_trunc.loc[
                  elec_data_trunc["Election Set"] == elec_set, 'Year'].values[
                  0].astype(str) \
              for elec_set in elec_sets]
recency_scores = [recency_weights[elec_year][0] for elec_year in elec_years]
recency_W1 = np.tile(recency_scores, (num_districts, 1)).transpose()

# precompute statewide EI and recency (W1), in-group candidate(W2),
# and candidate confidence (W3) for statewide/equal modes
if record_statewide_modes:
    black_weight_state, hisp_weight_state, neither_weight_state, black_weight_equal, \
    hisp_weight_equal, neither_weight_equal, black_pref_cands_prim_state, hisp_pref_cands_prim_state, \
    black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state \
        = precompute_state_weights(num_districts, elec_sets, elec_set_dict,
                                   recency_W1, EI_statewide, primary_elecs, \
                                   runoff_elecs, elec_match_dict,
                                   min_cand_weights_dict, cand_race_dict)

# precompute set-up for district mode (need precinct-level EI data)
if record_district_mode:
    demogs = ['BCVAP', 'HCVAP']
    bases = {col.split('.')[0] + '.' + col.split('.')[1] for col in
             prec_ei_df.columns if
             col[:5] in demogs and 'abstain' not in col and \
             not any(x in col for x in general_elecs)}
    base_dict = {b: (
    b.split('.')[0].split('_')[0], '_'.join(b.split('.')[1].split('_')[1:-1]))
                 for b in bases}
    outcomes = {val: [] for val in base_dict.values()}
    for b in bases:
        outcomes[base_dict[b]].append(b)

    precs = list(state_gdf[geo_id])
    prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases,
                                                       outcomes)


############################################################################################################
# UPDATERS FOR CHAIN

# The elections model function (used as an updater). Takes in partition and returns effectiveness distribution per district
def final_elec_model(partition):
    """
    The output of the elections model is a probability distribution for each district:
    (Latino, Black, Neither or Overlap)-effective
    To compute these, each election set is first weighted (different for Black and Latino)
    according to three factors:
    a recency weight (W1), "in-group"-minority-preference weight (W2) and
    a preferred-candidate-confidence weight (W3).
    If the Black (Latino) preferred candidate wins the election (set) a number of points equal to
    the set's weight is accrued. The ratio of the accrued points to the total possible points
    is the raw Black (Latino)-effectiviness score for the district.

    Raw scores are adjusted by multiplying them by a "Group Control" factor,
    which measures the share of votes cast
    for a minority-preferred candidate by the minority group itself.

    Finally, the Black, Latino, Overlap, and Neither distribution (the values sum to 1)
    is computed, by feeding the adjusted effectiveness scores through a fitted logit function,
    and interpolating for the final four values. The output scores can be interpreted as the
    probability a district is effective for each group.

    We need to track several entities in the model, which will be dataframes or arrays,
    whose columns are districts and rows are election sets (or sometimes individual elections).
    These dataframes each store one of the following: Black (Latino) preferred candidates (in the
    election set's primary), Black (Latino) preferred candidates in runoffs, winners of primary,
    runoff and general elections, weights W1, W2 and W3
    and final election set weights for Black and Latino voters.
    """
    ###########################################################
    # We only need to run model on two ReCom districts that have changed in each step
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = set(
            [dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]]).union(
            set([dict2[k] for k in dict2.keys() if dict1[k] != dict2[k]]))

    dist_changes = range(num_districts) if partition.parent is None else sorted(
        differences)

    # dictionary to store district-level candidate vote shares
    dist_elec_results = {}
    order = [x for x in partition.parts]
    for elec in elections:
        cands = candidates[elec]
        dist_elec_results[elec] = {}
        outcome_list = [dict(zip(order, partition[elec].percents(cand))) for
                        cand in cands.keys()]
        dist_elec_results[elec] = {
            d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in
            range(num_districts)}
    ##########################################################################################
    # compute winners of each election in each district and store:
    map_winners = pd.DataFrame(columns=dist_changes)
    map_winners["Election"] = elections
    map_winners["Election Set"] = elec_data_trunc["Election Set"]
    map_winners["Election Type"] = elec_data_trunc["Type"]
    for i in dist_changes:
        map_winners[i] = [
            max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[
                0] for elec in elections]

    ######################################################################################
    # If we compute statewide scores: compute district effectivness probabilities  #################
    if record_statewide_modes:
        # district probability distribution: statewide
        final_state_prob_dict = compute_final_dist(map_winners,
                                                   black_pref_cands_prim_state,
                                                   black_pref_cands_runoffs_state, \
                                                   hisp_pref_cands_prim_state,
                                                   hisp_pref_cands_runoffs_state,
                                                   neither_weight_state, \
                                                   black_weight_state,
                                                   hisp_weight_state,
                                                   dist_elec_results,
                                                   dist_changes,
                                                   cand_race_table,
                                                   num_districts, candidates,
                                                   elec_sets, elec_set_dict, \
                                                   "statewide", partition,
                                                   logit_params, logit=True)

        # district probability distribution: equal
        final_equal_prob_dict = compute_final_dist(map_winners,
                                                   black_pref_cands_prim_state,
                                                   black_pref_cands_runoffs_state, \
                                                   hisp_pref_cands_prim_state,
                                                   hisp_pref_cands_runoffs_state,
                                                   neither_weight_equal, \
                                                   black_weight_equal,
                                                   hisp_weight_equal,
                                                   dist_elec_results,
                                                   dist_changes,
                                                   cand_race_table,
                                                   num_districts, candidates,
                                                   elec_sets, elec_set_dict, \
                                                   "equal", partition,
                                                   logit_params, logit=True)

    # If we are computing district score: ######################################################
    # compute district weights, preferred candidates and district probability distribution: district
    if record_district_mode:
        black_weight_dist, hisp_weight_dist, neither_weight_dist, black_pref_cands_prim_dist, \
        black_pref_cands_runoffs_dist, hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist \
            = compute_district_weights(dist_changes, elec_sets, elec_set_dict,
                                       state_gdf, partition,
                                       prec_draws_outcomes, \
                                       geo_id, primary_elecs, runoff_elecs,
                                       elec_match_dict, bases, outcomes, \
                                       recency_W1, cand_race_dict,
                                       min_cand_weights_dict)

        final_dist_prob_dict = compute_final_dist(map_winners,
                                                  black_pref_cands_prim_dist,
                                                  black_pref_cands_runoffs_dist, \
                                                  hisp_pref_cands_prim_dist,
                                                  hisp_pref_cands_runoffs_dist,
                                                  neither_weight_dist, \
                                                  black_weight_dist,
                                                  hisp_weight_dist,
                                                  dist_elec_results,
                                                  dist_changes,
                                                  cand_race_table,
                                                  num_districts, candidates,
                                                  elec_sets, elec_set_dict, \
                                                  'district', partition,
                                                  logit_params, logit=True)

    # New vector of probability distributions-by-district is the same as last ReCom step,
    # except in 2 changed districts
    if partition.parent == None:
        final_state_prob = {key: final_state_prob_dict[key] for key in
                            sorted(final_state_prob_dict)} \
            if record_statewide_modes else {key: "N/A" for key in
                                            sorted(dist_changes)}

        final_equal_prob = {key: final_equal_prob_dict[key] for key in
                            sorted(final_equal_prob_dict)} \
            if record_statewide_modes else {key: "N/A" for key in
                                            sorted(dist_changes)}

        final_dist_prob = {key: final_dist_prob_dict[key] for key in
                           sorted(final_dist_prob_dict)} \
            if record_district_mode else {key: "N/A" for key in
                                          sorted(dist_changes)}

    else:
        final_state_prob = partition.parent["final_elec_model"][0].copy()
        final_equal_prob = partition.parent["final_elec_model"][1].copy()
        final_dist_prob = partition.parent["final_elec_model"][2].copy()

        for i in dist_changes:
            if record_statewide_modes:
                final_state_prob[i] = final_state_prob_dict[i]
                final_equal_prob[i] = final_equal_prob_dict[i]

            if record_district_mode:
                final_dist_prob[i] = final_dist_prob_dict[i]

    return final_state_prob, final_equal_prob, final_dist_prob


def effective_districts(dictionary):
    """
    Given district effectiveness distributions, this function returns the total districts
    that are above the effectivness threshold for Black and Latino voters, and the total
    distinct effective districts.
    """
    black_threshold = effectiveness_cutoff
    hisp_threshold = effectiveness_cutoff

    if "N/A" not in dictionary.values():
        hisp_effective = [i + l for i, j, k, l in dictionary.values()]
        black_effective = [j + l for i, j, k, l in dictionary.values()]

        hisp_effect_index = [i for i, n in enumerate(hisp_effective) if
                             n >= hisp_threshold]
        black_effect_index = [i for i, n in enumerate(black_effective) if
                              n >= black_threshold]

        total_hisp_final = len(hisp_effect_index)
        total_black_final = len(black_effect_index)
        total_distinct = len(set(hisp_effect_index + black_effect_index))

        return total_hisp_final, total_black_final, total_distinct

    else:
        return "N/A", "N/A", "N/A"


#####construct updaters for Chain###############################################
my_updaters = {
    "population": updaters.Tally(tot_pop, alias="population"),
    "CVAP": updaters.Tally(CVAP, alias="CVAP"),
    "WCVAP": updaters.Tally(WCVAP, alias="WCVAP"),
    "HCVAP": updaters.Tally(HCVAP, alias="HCVAP"),
    "BCVAP": updaters.Tally(BCVAP, alias="BCVAP"),
    "Sum_CX": updaters.Tally(C_X, alias="Sum_CX"),
    "Sum_CY": updaters.Tally(C_Y, alias="Sum_CY"),
    "cut_edges": cut_edges,
    "final_elec_model": final_elec_model,
}

# add elections updaters
elections_track = [
    Election("PRES16", {"Democratic": 'ClintonD_16G_President',
                        "Republican": 'TrumpR_16G_President'}, alias="PRES16"),
    Election("PRES12", {"Democratic": 'ObamaD_12G_President',
                        "Republican": 'RomneyR_12G_President'}, alias="PRES12"),
    Election("SEN18", {"Democratic": "ORourkeD_18G_US_Sen",
                       "Republican": 'CruzR_18G_US_Sen'}, alias="SEN18"),
    Election("GOV18", {"Democratic": "ValdezD_18G_Governor",
                       "Republican": 'AbbottR_18G_Governor'}, alias="GOV18"),

]

election_updaters = {election.name: election for election in elections_track}
my_updaters.update(election_updaters)

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)


# constraints ############################################
def inclusion(partition):
    """
    Returns 'True' if proposed plan has greater than or equal to the number of Black, Latino and
    distinct effective districts as the enacted map.
    """
    final_state_prob, final_equal_prob, final_dist_prob = partition[
        "final_elec_model"]
    inclusion_dict = final_state_prob if model_mode == 'statewide' else final_equal_prob if model_mode == 'equal' else final_dist_prob
    hisp_vra_dists, black_vra_dists, total_distinct = effective_districts(
        inclusion_dict)

    return total_distinct >= enacted_distinct and \
           black_vra_dists >= enacted_black and hisp_vra_dists >= enacted_hisp

# initial partition#######################################################
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population / num_districts

start_map_2019 = recursive_tree_part(graph, range(num_districts),
                                    ideal_population, tot_pop, pop_tol, 3)

initial_partition_2019 = GeographicPartition(graph=graph, assignment=start_map,
                                        updaters=my_updaters)

while not inclusion(initial_partition_2019):
    start_map_2019 = recursive_tree_part(graph, range(num_districts),
                                         ideal_population, tot_pop,
                                         pop_tol)

    initial_partition_2019 = GeographicPartition(graph = graph,
                                                 assignment = start_map_2019,
                                                 updaters = my_updaters)


df['Seed_Demo_'] = df.index.map(initial_partition_2019.assignment.to_dict())

# save appended file
DIR = os.path.join('./Data', 'TX_VTDs_POP2019')
if not os.path.exists(DIR):
    os.mkdir(DIR)

df.to_file(os.path.join(DIR, "texas_population2019.shp"))
