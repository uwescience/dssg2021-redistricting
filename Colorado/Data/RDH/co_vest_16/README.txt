2016 Colorado precinct and election results shapefile.

## RDH Date retrieval
11/26/2020

## Sources
Election results from the Colorado Secretary of State (https://www.sos.state.co.us/pubs/elections/Results/Archives.html)
Precinct shapefiles primarily from the U.S. Census Bureau's 2020 Redistricting Data Program Phase 2 release. The following counties used shapefiles sourced from the respective county governments instead: Adams, Arapahoe, Boulder, Chaffee, Delta, Denver, Douglas, El Paso, Fremont, Garfield, Gilpin, Jefferson, La Plata, Larimer, Mesa, Pitkin, Pueblo, Summit, Weld.

## Fields metadata

Vote Column Label Format
------------------------
Columns reporting votes follow a standard label pattern. One example is:
G16PREDCli
The first character is G for a general election, P for a primary, C for a caucus, R for a runoff, S for a special.
Characters 2 and 3 are the year of the election.
Characters 4-6 represent the office type (see list below).
Character 7 represents the party of the candidate.
Characters 8-10 are the first three letters of the candidate's last name.

Office Codes
AGR - Commissioner of Agriculture
ATG - Attorney General
AUD - Auditor
COM - Comptroller
COU - City Council Member
DEL - Delegate to the U.S. House
GOV - Governor
H## - U.S. House, where ## is the district number. AL: at large.
HOD - House of Delegates, accompanied by a HOD_DIST column indicating district number
HOR - U.S. House, accompanied by a HOR_DIST column indicating district number
INS - Commissioner of Insurance
LAB - Commissioner of Labor
LTG - Lieutenant Governor
LND - Commissioner of Public Lands
PRE - President
PSC - Public Service Commissioner
PUC - Public Utilities Commissioner
RGT - State University Regent
RRC - Railroad Commissioner
SAC - State Court of Appeals
SOS - Secretary of State
SOV - Senate of Virginia, accompanied by a SOV_DIST column indicating district number
SPI - Superintendent of Public Instruction
SSC - State Supreme Court
TRE - Treasurer
USS - U.S. Senate

Party Codes
D and R will always represent Democrat and Republican, respectively.
See the state-specific notes for the remaining codes used in a particular file; note that third-party candidates may appear on the ballot under different party labels in different states.

## Fields
G16PREDCLI - Hillary Clinton (Democratic Party)
G16PRERTRU - Donald J. Trump (Republican Party)
G16PRELJOH - Gary Johnson (Libertarian Party)
G16PREGSTE - Jill Stein (Green Party)
G16PREUMCM - Evan McMullin (Unaffiliated)
G16PREOOTH - Other Candidates

G16USSDBEN - Michael Bennet (Democratic Party)
G16USSRGLE - Darryl Glenn (Republican Party)
G16USSLTAN - Lily Tang Williams (Libertarian Party)
G16USSGMEN - Arn Menconi (Green Party)
G16USSOOTH - Other Candidates

G16RGTDMAD - Alice Madden (Democratic Party)
G16RGTRGAN - Heidi Ganahl (Republican Party)


## Processing Steps
Las Animas County precinct assignments in the voter file differ markedly from both the Census VTD boundaries and from maps received from the county. All precincts were revised to match the geocoded voter file and the list of districts assigned to precinct splits. As appropriate, precinct boundaries were revised using Census blocks, the Trinidad municipal boundary shapefile, school district or fire district boundaries, and the parcel shapefile from the Las Animas County Assessor.

The following additional revisions were made to match the 2016 precinct boundaries:

Lake: Revise Precincts 1/4, 2/3 to reverse 2017 redraw
Logan: Align Sterling City precincts with city limits
Montezuma: Precincts renumbered to match county maps
Otero: Align La Junta City precincts with county maps
Prowers: All precincts adjusted to match county maps
Rio Grande: Adjust Precincts 2/3 to match county maps

Larimer County reported provisional votes countywide. These were distributed to precincts by candidate based on their share of the vote reported by precinct.
