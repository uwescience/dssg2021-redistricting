2020 Colorado precinct and election results shapefile.

## RDH Date retrieval
06/30/2021

## Sources
Election results from the Colorado Secretary of State (https://www.sos.state.co.us/pubs/elections/Results/Archives.html)
Precinct shapefiles primarily from the U.S. Census Bureau's 2020 Redistricting Data Program. The following counties used shapefiles sourced from the respective county governments instead: Adams, Arapahoe, Boulder, Delta, Denver, Douglas, El Paso, Fremont, Garfield, Gilpin, Jefferson, La Plata, Larimer, Mesa, Pitkin, Pueblo, Summit, Weld.

## Fields metadata

Vote Column Label Format
------------------------
Columns reporting votes follow a standard label pattern. One example is:
G20PRERTRU
The first character is G for a general election, C for recount results, P for a primary, S for a special, and R for a runoff.
Characters 2 and 3 are the year of the election.
Characters 4-6 represent the office type (see list below).
Character 7 represents the party of the candidate.
Characters 8-10 are the first three letters of the candidate's last name.

Office Codes
AGR - Agriculture Commissioner
ATG - Attorney General
AUD - Auditor
COC - Corporation Commissioner
COU - City Council Member
DEL - Delegate to the U.S. House
GOV - Governor
H## - U.S. House, where ## is the district number. AL: at large.
INS - Insurance Commissioner
LAB - Labor Commissioner
LTG - Lieutenant Governor
PRE - President
PSC - Public Service Commissioner
SAC - State Appeals Court (in AL: Civil Appeals)
SCC - State Court of Criminal Appeals
SOS - Secretary of State
SSC - State Supreme Court
SPI - Superintendent of Public Instruction
TRE - Treasurer
USS - U.S. Senate

Party Codes
D and R will always represent Democrat and Republican, respectively.
See the state-specific notes for the remaining codes used in a particular file; note that third-party candidates may appear on the ballot under different party labels in different states.

## Fields
G20PREDBID - Joseph R. Biden (Democratic Party)
G20PRERTRU - Donald J. Trump (Republican Party)
G20PRELJOR - Jo Jorgensen (Libertarian Party)
G20PREGHAW - Howie Hawkins (Green Party)
G20PRECBLA - Don Blankenship (American Constitution Party)
G20PREUWES - Kanye West (Unaffiliated)
G20PREOOTH - Other Candidates

G20USSDHIC - John W. Hickenlooper (Democratic Party)
G20USSRGAR - Cory Gardner (Republican Party)
G20USSLDOA - Raymon Anthony Doane (Libertarian Party)
G20USSODOY - Daniel Doyle (Approval Voting Party)
G20USSOEVA - Stehpan "Seku" Evans (Unity Party)
G20USSOWRI - Write-in Votes

## Processing Steps
Las Animas County precinct assignments in the voter file differ markedly from both the Census VTD boundaries and from maps received from the county. All precincts were revised to match the geocoded voter file and the list of districts assigned to precinct splits. As appropriate, precinct boundaries were revised using Census blocks, the Trinidad municipal boundary shapefile, school district or fire district boundaries, and the parcel shapefile from the Las Animas County Assessor.

The following additional revisions were made to match the 2020 precinct boundaries:

Logan: Align Sterling City precincts with city limits
Montezuma: Precincts renumbered to match county maps
Morgan: Split Precincts 1/18 to match county maps
Otero: Align La Junta City precincts with county maps
Prowers: All precincts adjusted to match county maps
Rio Grande: Adjust Precincts 2/3 to match county maps
