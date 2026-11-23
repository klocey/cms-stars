"""Loads the compiled star-ratings data and the what-if input data, and derives
the lists used by the app's controls. Paths are relative to the repository root."""

import numpy as np
import pandas as pd

################################# LOAD DATA ##################################
month = 'Nov'
year = '2026'
main_df = pd.read_pickle('dataframe_data/hosp_stars_dat_from_'+month+'_'+year+'.pkl')
beds_max = np.nanmax(main_df['Beds'])

what_if_df_2026 = pd.read_pickle('dataframe_data/2026_data_for_whatifs.pkl')
what_if_df_2027 = pd.read_pickle('dataframe_data/2027_data_for_whatifs.pkl')

HOSPITALS = main_df['Name and Num'].tolist()

beds = main_df['Beds'].tolist()
states = main_df['State'].tolist()
htypes = main_df['Hospital Type'].tolist()
ctypes = main_df['Hospital Ownership'].tolist()

main_df['Release year'] = main_df['Release year'].astype(int)

latest_yr = np.max(main_df['Release year'])
latest_release_yr = latest_yr - 1

# The latest year in the data holds the app's predictions for the next CMS release
PREDICTED_YEAR = latest_yr


def year_label(yr):
    """Dropdown label for a release year. The predicted year is marked as such."""
    return str(yr) + (" (predicted)" if yr == PREDICTED_YEAR else "")

states = ['NaN' if x is np.nan else x for x in states]
htypes = ['NaN' if x is np.nan else x for x in htypes]
ctypes = ['NaN' if x is np.nan else x for x in ctypes]

domains = ['Patient Experience', 'Readmission', 'Mortality', 
           'Safety of Care', 'Timely and Effective Care']
