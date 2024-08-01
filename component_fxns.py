from dash import dash_table

from scipy import stats
from scipy.stats import percentileofscore

import pandas as pd
import numpy as np
import re


def whatif_table(hospital, whatif_df):    
    
    whatif_df = pd.read_json(whatif_df)
    name = hospital[:-9]
    
    measures = []
    domains = []
          
    m1 = ['H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 
                   'H_COMP_3_STAR_RATING', 'H_COMP_5_STAR_RATING', 
                   'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING', 
                   'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING']
    domains.extend(['Patient Experience']*len(m1))
    measures.extend(m1)
    
    m2 = ['EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
                  'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE', 
                  'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36']
    domains.extend(['Readmission']*len(m2))
    measures.extend(m2)
    
    m3 = ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF', 
                 'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP']
    domains.extend(['Mortality']*len(m3))
    measures.extend(m3)
    
    m4 = ['COMP_HIP_KNEE',  'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 
                   'HAI_5', 'HAI_6', 'PSI_90_SAFETY']
    domains.extend(['Safety of Care']*len(m4))
    measures.extend(m4)
    
    m5 = ['HCP_COVID_19', 'IMM_3', 'OP_10', 'OP_13', 'OP_18B',
                    #'OP_2',
                    'OP_22', 'OP_23', 'OP_29', 'OP_3B', 
                    'OP_8', 'PC_01', 'SEP_1']
    domains.extend(['Timely & Effective Care']*len(m5))
    measures.extend(m5)
    
    rev_measures = ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
                    'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'COMP_HIP_KNEE', 
                    'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6',
                    'PSI_90_SAFETY', 'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN',
                    'OP_32', 'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE', 
                    'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36', 'OP_22',
                    'PC_01', 'OP_3B', 'OP_18B', 'OP_8', 'OP_10','OP_13',
                   ]
    
    higher_better = []
    for m in measures:
        if m in rev_measures:
            higher_better.append('No')
        else:
            higher_better.append('Yes')
    
    tdf = whatif_df.filter(items=measures)
    tdf = tdf.round(3)
    mins = tdf.min().tolist()
    maxs = tdf.max().tolist()

    pnum = re.search(r'\d{6}', hospital)
    pnum = pnum.group()
    
    try:
        tdf = whatif_df[whatif_df['PROVIDER_ID'] == pnum]
        tdf = tdf.filter(items=measures)
        tdf = tdf.round(3)
    except:
        return {}, {}, {}, hospital + ' has no data in the CMS Care Compare release that is used to make predictions.'
    
    cols = ['Domain', 'Measure', 'Higher is better', 'Actual value', 'Min value', 'Max value',
            'What-if value'] 
            
    df_table = pd.DataFrame(columns=cols)
    df_table['Domain'] = domains
    df_table['Measure'] = list(tdf)
    df_table['Higher is better'] = higher_better
    df_table['Actual value'] = tdf.iloc[0].tolist()
    df_table['Min value'] = mins
    df_table['Max value'] = maxs
    df_table['What-if value'] = tdf.iloc[0].tolist()
    
    data=df_table.to_dict('records')
    columns=[
            {'id': c, 'name': c, 'editable': (c == 'What-if value')}  # Make only "What-if value" column editable
            for c in df_table.columns
        ]
    
    name = hospital[:-9]
    name1 = str(name) 
    if name == 'RUSH UNIVERSITY MEDICAL CENTER':
        name1 = 'RUMC'
    elif name == 'RUSH OAK PARK HOSPITAL':
        name1 = 'ROPH'
    elif name == 'RUSH COPLEY':
        name1 = 'RUSH COPLEY'
    else:
        name1 = "hospital " + hospital[-7:-1]
        
    return data, columns, df_table.to_json(), name1   

    



def measure_table(main_df, jd1, feature_dict, hospital, option_hospitals, selected_hosps_btn, stars_peers_btn, domain, score_type, yr, selected_hospitals):

    main_df = pd.read_json(main_df)

    cols = ['Measure', 'Value', 'Delta value', 
            'Percentile', 'Delta percentile'] 
            #'Weight', 'Delta weight']
    
    df_table = pd.DataFrame(columns=cols)
    for i in list(df_table):
        df_table[i] = [np.nan]*4
    
    dashT = dash_table.DataTable(
        data=df_table.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_table.columns],
        export_format="csv",
        page_action='none',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        
        style_table={#'height': '500px', 
                     'overflowY': 'auto',
                     },
        style_cell={'padding':'5px',
                    'minwidth':'140px',
                    'width':'160px',
                    'maxwidth':'160px',
                    'whiteSpace':'normal',
                    'textAlign': 'center',
                    },
        ) 
    
    # No hospital selected
    if hospital is None:
            
        txt = "Please select a focal hospital."
        return dashT, txt
    
    tdf_main = main_df.copy(deep=True)    
    name = hospital[:-9]
    hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
    
    # Selected hospital has no data among release years
    if hosp_df.shape[0] == 0:    
        txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
        return dashT, txt
    
    yrs = sorted(hosp_df['Release year'].unique().tolist())
    
    # There is only one release year for the selected hospital (can't compute deltas)
    if len(yrs) <= 1:
        txt = hospital + ' had only one release year. Cannot compute delta values. Try selecting another hospital.'
        return dashT, txt
    
    yrs = sorted(hosp_df['Release year'].unique().tolist())
    yr1 = int(yr)
    yr2 = int()
    
    if yr1 == 2025:
        yr2 = 2024
        if yr2 in yrs:
            pass
        else:
            yr2 == 2023
            
    if yr1 == 2024:
        yr2 = 2023
        if yr2 in yrs:
            pass
        else:
            yr2 == 2022
        
    if yr1 == 2023:
        yr2 = 2022
        if yr2 in yrs:
            pass
        else:
            yr2 == 2021
            
    elif yr1 == 2022:
        yr2 = 2021
    
    
    set_select = 'Measures group'
    if 'selected_hosps_btn3' in jd1:
        set_select = 'Selected hospitals'
    elif 'stars_peers_btn3' in jd1:
        set_select = 'Measures group'
        
        
    if set_select == 'Measures group':
        
        tdf_main_LY = tdf_main[tdf_main['Release year'] == yr1]
        grp_LY = tdf_main_LY[tdf_main_LY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
        tdf_main_LY = tdf_main_LY[tdf_main_LY['cnt_grp'].isin([grp_LY])]
        
        if yr1 == 2021:
            grp_PY = np.nan
            tdf_main_PY = None
            
        else:
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yr2]
            grp_PY = tdf_main_PY[tdf_main_PY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
            tdf_main_PY = tdf_main_PY[tdf_main_PY['cnt_grp'].isin([grp_PY])]
    
    elif set_select == 'Selected hospitals':
    
        if option_hospitals is None or option_hospitals == []:
                
            txt = ''
            if option_hospitals is None or option_hospitals == []:
                txt += "You either haven't selected any hospitals or the filters you chose left you with no hospitals to analyze."
    
            return dashT, txt
        
    
        if hospital in option_hospitals:
            # Remove focal hospital from options, for use below
            option_hospitals.remove(hospital)
            
        if selected_hospitals is None or selected_hospitals == [] or selected_hospitals == [hospital]:
            #There are options but either none were selected specifically by name or only our focal hospital was selected by name
            selected_hospitals = list(option_hospitals)
        
        if hospital in selected_hospitals:
            #Remove focal hospital from options, for use below
            selected_hospitals.remove(hospital)
        
        tdf_main = main_df[main_df['Name and Num'].isin(selected_hospitals + [hospital])]
        tdf_main_LY = tdf_main[tdf_main['Release year'] == yr1]
        
        if yr1 == 2021:
            tdf_main_PY = None
    
        else:
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yr2]            
            
    ######## GET RESULTS FOR LATEST YEAR ##############
    # Get hospitals
    hosp_ls_LY = tdf_main_LY['Name and Num'].tolist()
    
    i = 0
    try:
        i = hosp_ls_LY.index(hospital)
    except:
        txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
        return dashT, txt
            
    # Get measures
    measure_ls = []
    if score_type == 'Standardized scores':
        measure_ls = feature_dict[domain + ' (std)']
    elif score_type == 'Raw scores':
        measure_ls = feature_dict[domain]
    
    hosp_scors_LY = []
    hosp_percs_LY = []
    hosp_wts_LY = []
    
    # Get values for latest year
    labels_ls = []
    for ii, m in enumerate(measure_ls):
        try:
            ls = tdf_main_LY[m].tolist()
            
            hosp_scors_LY.append(ls[i])
            perc = round(stats.percentileofscore(ls, ls[i]), 1)
            hosp_percs_LY.append(perc)
            
            ls2 = feature_dict[domain + ' labels']
            labels_ls.append(ls2[ii])
            
            if domain == 'Patient Experience':
                pref = 'patient_exp_'   
            elif domain == 'Readmission':
                pref = 'readmission_'
            elif domain == 'Mortality':
                pref = 'mortality_'   
            elif domain == 'Safety of Care':
                pref = 'safety_'   
            elif domain == 'Timely and Effective Care':
                pref = 'process_'   
                        
            ls = tdf_main_LY[pref + 'measure_wt'].tolist()
            hosp_wts_LY.append(ls[i])
        except:
            pass
    
    
    if yr1 == 2021:
        hosp_scors_PY = [np.nan]*len(hosp_scors_LY)
        hosp_percs_PY = [np.nan]*len(hosp_percs_LY)
        hosp_wts_PY = [np.nan] * len(hosp_wts_LY)
        
    else:
        ######## GET RESULTS FOR NEXT LATEST YEAR ##############
            
        # Get hospitals
        hosp_ls_PY = tdf_main_PY['Name and Num'].tolist()
        i = 0
        try:
            i = hosp_ls_PY.index(hospital)
        except:
            txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital"
            return dashT, txt
                
        # Get measures
                
        measure_ls = []
        if score_type == 'Standardized scores':
            measure_ls = feature_dict[domain + ' (std)']
        elif score_type == 'Raw scores':
            measure_ls = feature_dict[domain]
                    
        hosp_scors_PY = []
        hosp_percs_PY = []
        hosp_wts_PY = []
            
        # Get values for next latest year
        labels_ls = []
        for ii, m in enumerate(measure_ls):
            try:
                ls = tdf_main_PY[m].tolist()
                hosp_scors_PY.append(ls[i])
                perc = round(stats.percentileofscore(ls, ls[i]), 1)
                hosp_percs_PY.append(perc)
                
                ls2 = feature_dict[domain + ' labels']
                labels_ls.append(ls2[ii])
                        
                # get individual measure weight ... somehow
                        
                if domain == 'Patient Experience':
                    pref = 'patient_exp_'   
                elif domain == 'Readmission':
                    pref = 'readmission_'
                elif domain == 'Mortality':
                    pref = 'mortality_'   
                elif domain == 'Safety of Care':
                    pref = 'safety_'   
                elif domain == 'Timely and Effective Care':
                    pref = 'process_'   
                            
                ls = tdf_main_PY[pref + 'measure_wt'].tolist()
                hosp_wts_PY.append(ls[i])
            except:
                pass
                    
        #########
        
    # Compute values for columns
            
    delta_value = np.round(np.array(hosp_scors_LY) - np.array(hosp_scors_PY), 3)
    delta_perc = np.round(np.array(hosp_percs_LY) - np.array(hosp_percs_PY), 3)
    #delta_wght = np.round(np.array(hosp_wts_LY) - np.array(hosp_wts_PY), 3)
            
    cols = ['Measure', 'Value', 'Delta value', 
            'Percentile', 'Delta percentile'] 
            #'Weight', 'Delta weight']
            
    df_table = pd.DataFrame(columns=cols)
    df_table['Measure'] = labels_ls
    df_table['Value'] = np.round(hosp_scors_LY, 3)
    df_table['Delta value'] = delta_value.tolist()
    df_table['Percentile'] = hosp_percs_LY
    df_table['Delta percentile'] = delta_perc
    #df_table['Weight'] = np.round(hosp_wts_LY, 4)
    #df_table['Delta weight'] = delta_wght
    
    df_table.dropna(how='all', axis=0, subset=['Value', 'Delta value', 
                                               'Percentile', 'Delta percentile', 
                                               ], inplace=True)
    
    if score_type == 'Standardized scores':        
        dashT = dash_table.DataTable(
            data=df_table.to_dict('records'), columns=[{'id': c, 'name': c} for c in df_table.columns],
            export_format="csv", page_action='none', sort_action="native", sort_mode="multi", #filter_action="native",
                    
            style_table={'overflowY': 'auto'},
            style_cell={'padding':'5px', 'minwidth':'140px',
                        'width':'160px', 'maxwidth':'160px',
                        'whiteSpace':'normal', 'textAlign': 'center'},
            style_cell_conditional=[
                        {'if': {'column_id': 'Measure'},
                         'width': '30%'},
                        #{'if': {'column_id': 'Region'},
                        # 'width': '30%'},
                        ],
            style_data_conditional=[
                {
                    'if': {
                        'column_id': 'Delta value',
                        'filter_query': '{Delta value} > 0'
                        },
                    'backgroundColor': 'green',
                    'color': 'white'
                    },
                {
                    'if': {
                        'column_id': 'Delta value',
                        'filter_query': '{Delta value} < 0'
                        },
                    'backgroundColor': 'red',
                    'color': 'white'
                    },
                {
                    'if': {
                        'column_id': 'Delta percentile',
                        'filter_query': '{Delta percentile} > 0'
                        },
                    'backgroundColor': 'green',
                    'color': 'white'
                    },
                {
                    'if': {
                        'column_id': 'Delta percentile',
                        'filter_query': '{Delta percentile} < 0'
                        },
                    'backgroundColor': 'red',
                    'color': 'white'
                    }]
            ) 
    
    else:        
        dashT = dash_table.DataTable(
            data=df_table.to_dict('records'), columns=[{'id': c, 'name': c} for c in df_table.columns],
            export_format="csv", page_action='none', sort_action="native", sort_mode="multi", #filter_action="native",
                    
            style_table={'overflowY': 'auto'},
            style_cell={'padding':'5px', 'minwidth':'140px',
                        'width':'160px', 'maxwidth':'160px',
                        'whiteSpace':'normal', 'textAlign': 'center'},
            style_cell_conditional=[
                        {'if': {'column_id': 'Measure'},
                         'width': '30%'},
                        #{'if': {'column_id': 'Region'},
                        # 'width': '30%'},
                    ]
            ) 
            
    name1 = str(name) 
    if name == 'RUSH UNIVERSITY MEDICAL CENTER':
        name1 = 'RUMC'
    elif name == 'RUSH OAK PARK HOSPITAL':
        name1 = 'ROPH'
    elif name == 'RUSH COPLEY':
        name1 = 'RUSH COPLEY'
    else:
        name1 = "hospital " + hospital[-7:-1]
    
    if yr1 == 2021:
        txt = "This application does not provide delta values for 2021 because it does not include overall star results from 2020."
    
    else:
        txt = "The latest year of data used was " + str(yr1) + ". "
        txt += "Delta values were computed using the prior year " + str(yr2) + ". "
    
        if set_select == 'Measures group':        
            
            if np.isnan(grp_LY) == True and np.isnan(grp_PY) == True:
                txt += "In both years, " + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_LY) == True:
                txt = "In " + str(yr1) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_PY) == True and yr1 != 2021:
                txt = "In " + str(yr2) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            
            elif grp_LY == grp_PY:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                txt += "In both years, " + name1 + " was in group " + str(int(grp_LY)) + numD_LY + '. '
            else:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                numD_PY = ' (hospitals w/ scores in ' + str(int(grp_PY + 2)) + ' domains)'
                        
                txt += name1 + " was in group " + str(int(grp_PY)) + numD_PY + " in " + str(yr2)
                txt += " and in group " + str(int(grp_LY)) + numD_LY + " in " + str(yr1) + ". "
        
        if score_type == 'Raw scores':
            txt += " Delta's are not color-coded because, unlike standardized scores, greater deltas for raw scores "
            txt += "do not necessarily imply improvement."
        
        else:
            txt += "Delta's are color-coded (green = improved; red = worsened)."
    
    return dashT, txt


    