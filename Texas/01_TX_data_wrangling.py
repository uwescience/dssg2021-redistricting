"""
This script adds the ACS 2019 population totals extracted from Redisticting Data Hub
(https://redistrictingdatahub.org/dataset/texas-block-group-acs5-data-2019/)
to the shape file provided by MGGG (https://www.dropbox.com/sh/k78n2hyixmv9xdg/AABmZG5ntMbXtX1VKThR7_t8a?dl=0)
"""

import geopandas as gpd
import pandas as pd
import maup

import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)


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

# save appended file
df.to_file("~/Desktop/texas_population2020.shp")