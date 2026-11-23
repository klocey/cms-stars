#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:22:59 2026

@author: kenlocey
"""
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats
import pandas as pd

def run_whatif(raw_data, pnum, yr):
    
    def adjust_edge_cases(row, centers, col):
        """
        Adjust cluster labels for edge cases, accounting for cases where the distance to the assigned cluster is 0.0.
        
        Args:
        - row: The row of the DataFrame being processed.
        - centers: Array of cluster centers.
        - col: The column name of the measure used for clustering.
        
        Returns:
        - Adjusted cluster label.
        """
        distances = np.abs(centers - row[col])
        closest, second_closest = np.partition(distances, 1)[:2]
        # Check if the closest distance is 0.0; if so, do not adjust the label
        if closest == 0:
            return row['cluster'] + 1
    
        # If the closest and second closest centers are very close, consider adjusting the label
        if np.isclose(closest, second_closest, atol=0.0001):  # 'atol' might need adjustment
            return row['cluster'] + 1  # Increment the cluster label for edge cases
        else:
            return row['cluster']


    def kmeans_clustering(df, n_clusters=5, col='summary_score'):
        # Step 1: Initial Data Preparation - Determine quintile medians as initial seeds
        quintiles = np.percentile(df[col].dropna(),
                                  [20.0, 40.0, 60.0, 80.0],
                                  method='interpolated_inverted_cdf', # this is good
                                  #method='linear',
                                 )
        df['grp'] = pd.cut(df[col], bins=[-np.inf] + quintiles.tolist() + [np.inf], labels=False) + 1
    
        # Step 2: Initial K-Means Clustering - Compute median for initial seeds
        initial_seeds = df.groupby('grp')[col].median().dropna().values.reshape(-1, 1)
    
        kmeans_initial = KMeans(n_clusters=len(initial_seeds), 
                                init=initial_seeds, 
                                n_init=100, 
                                max_iter=1000, 
                                random_state=0,
                                tol=0.000001,
                                #algorithm='auto',
                               )
        kmeans_initial.fit(df[[col]].dropna())
    
        # Use cluster centers from initial k-means as seeds for the main clustering
        main_seeds = kmeans_initial.cluster_centers_
    
        # Step 3: Second K-Means Clustering - Main clustering with refined seeds
        kmeans_main = KMeans(n_clusters=n_clusters, 
                             init=main_seeds, 
                             n_init=100, 
                             max_iter=1000, 
                             random_state=0,
                             tol=0.000001,
                             #algorithm='auto',
                            )
        df['cluster'] = kmeans_main.fit_predict(df[[col]].dropna())
    
        # Post-clustering adjustment for edge cases
        centers = kmeans_main.cluster_centers_.flatten()
        df['cluster'] = df.apply(adjust_edge_cases, centers=centers, col=col, axis=1)
    
        # Step 4: Cluster Ordering and Labeling - Order clusters and assign 'star' ratings
        cluster_means = df.groupby('cluster')[col].mean().sort_values().index
        cluster_mapping = {old: new for new, old in enumerate(cluster_means, 1)}
    
        df['star'] = df['cluster'].map(cluster_mapping)
        df.drop('cluster', axis=1, inplace=True)
    
        return df

    # Define the measures you're interested in
    rev_measures = ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
                    'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'COMP_HIP_KNEE', 
                    'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6',
                    'PSI_90_SAFETY', 'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN',
                    'OP_32', 'READM_30_CABG', 'READM_30_COPD', 
                    'READM_30_HIP_KNEE', 'Hybrid_HWR', 'Hybrid_HWM',
                    'OP_35_ADM', 'OP_35_ED', 'OP_36', 'OP_22',
                    'OP_18B', 'OP_8', 
                    'OP_10','OP_13', 'SAFE_USE_OF_OPIOIDS',
                   ]
    
    if yr == 2027:
        rev_measures = ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
                        'MORT_30_PN', 'MORT_30_STK', 
                        #'PSI_4_SURG_COMP', 
                        'COMP_HIP_KNEE', 
                        'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6',
                        'PSI_90_SAFETY', 'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN',
                        'OP_32', 'READM_30_CABG', 'READM_30_COPD', 
                        'READM_30_HIP_KNEE', 'Hybrid_HWR', 'Hybrid_HWM',
                        'OP_35_ADM', 'OP_35_ED', 'OP_36', 'OP_22',
                        'OP_18B', 
                        #'OP_8', 
                        'OP_10',
                        #'OP_13', 
                        'SAFE_USE_OF_OPIOIDS',
                       ]
        
    measures = [
        'MORT_30_AMI', 'MORT_30_CABG', 
        'MORT_30_COPD', 'MORT_30_HF',
        'MORT_30_PN', 'MORT_30_STK', 
        'PSI_4_SURG_COMP', 
        'COMP_HIP_KNEE',
        'HAI_1', 'HAI_2', 
        'HAI_3', 'HAI_4', 
        'HAI_5', 'HAI_6', 
        'PSI_90_SAFETY', 'EDAC_30_AMI', 
        'EDAC_30_HF', 'EDAC_30_PN', 
        'OP_32', 'READM_30_CABG', 
        'READM_30_COPD', 'READM_30_HIP_KNEE', 
        'Hybrid_HWR', 
        'Hybrid_HWM', # new
        'OP_35_ADM', 
        'OP_35_ED', 'OP_36', 
        'O-COMP-1',
        'O-COMP-2',
        'O-COMP-3',
        'O-PATIENT-RATE',
        'O-PATIENT-REC',
        'H_COMP_1_LINEAR_SCORE',
        'H_COMP_2_LINEAR_SCORE',
        'H_COMP_3_LINEAR_SCORE',
        'H_COMP_5_LINEAR_SCORE',
        'H_COMP_6_LINEAR_SCORE',
        'H_COMP_7_LINEAR_SCORE',
        'H_CLEAN_LINEAR_SCORE',
        'H_QUIET_LINEAR_SCORE',
        'H_RECMND_LINEAR_SCORE',
        'H_HSP_RATING_LINEAR_SCORE',
        'IMM_3', 'OP_10', 
        'OP_13', 'OP_18B', 
        'OP_22', 'OP_23', 
        'OP_29', 'OP_8', 
        'SEP_1', 'SAFE_USE_OF_OPIOIDS',
        # lost PC-01 and HCP COVID-19
       ]

    if yr == 2027:
        measures = [
            'MORT_30_AMI', 'MORT_30_CABG', 
            'MORT_30_COPD', 'MORT_30_HF',
            'MORT_30_PN', 'MORT_30_STK', 
            #'PSI_4_SURG_COMP', 
            'COMP_HIP_KNEE',
            'HAI_1', 'HAI_2', 
            'HAI_3', 'HAI_4', 
            'HAI_5', 'HAI_6', 
            'PSI_90_SAFETY', 'EDAC_30_AMI', 
            'EDAC_30_HF', 'EDAC_30_PN', 
            'OP_32', 'READM_30_CABG', 
            'READM_30_COPD', 'READM_30_HIP_KNEE', 
            'Hybrid_HWR', 
            'Hybrid_HWM', # new
            'OP_35_ADM', 
            'OP_35_ED', 'OP_36', 
            'O-COMP-1',
            'O-COMP-2',
            'O-COMP-3',
            'O-PATIENT-RATE',
            'O-PATIENT-REC',
            'H_COMP_1_LINEAR_SCORE',
            'H_COMP_2_LINEAR_SCORE',
            'H_COMP_3_LINEAR_SCORE',
            'H_COMP_5_LINEAR_SCORE',
            'H_COMP_6_LINEAR_SCORE',
            #'H_COMP_7_LINEAR_SCORE',
            'H_COMP_8_LINEAR_SCORE',
            'H_COMP_9_LINEAR_SCORE',
            'H_SYMP_LINEAR_SCORE',
            'H_CLEAN_LINEAR_SCORE',
            #'H_QUIET_LINEAR_SCORE',
            'H_RECMND_LINEAR_SCORE',
            'H_HSP_RATING_LINEAR_SCORE',
            'IMM_3', 'OP_10', 
            'OP_18B', 
            'OP_22', 'OP_23', 
            'OP_29', 
            #'OP_8', 'OP_13',
            'SEP_1', 'SAFE_USE_OF_OPIOIDS',
           ]
        
    filtered_data = raw_data.filter(items=['PROVIDER_ID']+measures)
    
    # remove any measure (column) that is not reported by at least 100 hospitals
    filtered_data = filtered_data.dropna(axis=1, thresh=101)
    
    ls = filtered_data.columns.drop('PROVIDER_ID').tolist()
    filtered_data = filtered_data.dropna(how='all', subset = ls, axis=0)
    
    filtered_measures = list(filtered_data)
    filtered_measures.remove('PROVIDER_ID')
    
    excluded = [item for item in measures if item not in filtered_measures]
    
    filtered_data.dropna(how='all', subset=filtered_measures, axis=0, inplace=True)
    filtered_data = filtered_data[filtered_data.columns[-1:].tolist() + filtered_data.columns[:-1].tolist()]

    ddof = 1
    zscore_df = filtered_data.copy(deep=True)
    for m in measures:
        if m in excluded:
            continue
            
        zscore_df[m] = stats.zscore(zscore_df[m], ddof=ddof, nan_policy='omit')
    
    for m in rev_measures:
        zscore_df[m] = -1*zscore_df[m]
        
    final_df = pd.DataFrame(columns=['PROVIDER_ID'])
    final_df['PROVIDER_ID'] = zscore_df['PROVIDER_ID']
    
    # Mortality measures
    mort_measures = ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF', 
                     'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'Hybrid_HWM']
    
    if yr == 2027:
        mort_measures.remove('PSI_4_SURG_COMP')
        
    final_df['Std_Outcomes_Mortality_score'] = stats.zscore(zscore_df[mort_measures].mean(axis=1), ddof=ddof, nan_policy='omit')
    final_df['Outcomes_Mortality_cnt'] = zscore_df[mort_measures].apply(lambda row: row.notna().sum(), axis=1)
    
    
    # Readmission measures
    readm_measures = ['EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
                      'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE', 
                      'Hybrid_HWR', 'OP_35_ADM', 'OP_35_ED', 'OP_36']
    
    final_df['Std_Outcomes_Readmission_score'] = stats.zscore(zscore_df[readm_measures].mean(axis=1), ddof=ddof, nan_policy='omit')
    final_df['Outcomes_Readmission_cnt'] = zscore_df[readm_measures].apply(lambda row: row.notna().sum(), axis=1)
    
    
    # SAFETY measures
    safety_measures = ['COMP_HIP_KNEE',  'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 
                       'HAI_5', 'HAI_6', 'PSI_90_SAFETY']
    final_df['Std_Outcomes_Safety_score'] = stats.zscore(zscore_df[safety_measures].mean(axis=1), ddof=ddof, nan_policy='omit')
    final_df['Outcomes_safety_cnt'] = zscore_df[safety_measures].apply(lambda row: row.notna().sum(), axis=1)
    
    
    # Patient experience measures
    patexp_measures = [
        'O-COMP-1',
        'O-COMP-2',
        'O-COMP-3',
        'O-PATIENT-RATE',
        'O-PATIENT-REC',
        'H_COMP_1_LINEAR_SCORE',
        'H_COMP_2_LINEAR_SCORE',
        'H_COMP_3_LINEAR_SCORE',
        'H_COMP_5_LINEAR_SCORE',
        'H_COMP_6_LINEAR_SCORE',
        'H_COMP_7_LINEAR_SCORE',
        'H_CLEAN_LINEAR_SCORE',
        'H_QUIET_LINEAR_SCORE',
        'H_RECMND_LINEAR_SCORE',
        'H_HSP_RATING_LINEAR_SCORE',
    ]
    
    if yr == 2027:
        patexp_measures.remove('H_COMP_7_LINEAR_SCORE') 
        patexp_measures.remove('H_QUIET_LINEAR_SCORE')
        patexp_measures.extend(['H_COMP_8_LINEAR_SCORE', 
                                'H_COMP_9_LINEAR_SCORE', 
                                'H_SYMP_LINEAR_SCORE',
                                ])
        
    final_df['Std_PatientExp_score'] = stats.zscore(zscore_df[patexp_measures].mean(axis=1), ddof=ddof, nan_policy='omit')
    final_df['Patient_Experience_cnt'] = zscore_df[patexp_measures].apply(lambda row: row.notna().sum(), axis=1)
    
    
    # Timely effective care measures
    proc_measures = ['IMM_3', 'OP_10', 'OP_13', 'OP_18B',
                     'SAFE_USE_OF_OPIOIDS', 'OP_22', 
                     'OP_23', 'OP_29', 'OP_8', 'SEP_1']
    
    if yr == 2027:
        proc_measures.remove('OP_13')
        proc_measures.remove('OP_8')
    
    final_df['Std_Process_score'] = stats.zscore(zscore_df[proc_measures].mean(axis=1), ddof=ddof, nan_policy='omit')
    final_df['Process_cnt'] = zscore_df[proc_measures].apply(lambda row: row.notna().sum(), axis=1)
    
    mort_cnts = final_df['Outcomes_Mortality_cnt'].tolist()
    safe_cnts = final_df['Outcomes_safety_cnt'].tolist()
    read_cnts = final_df['Outcomes_Readmission_cnt'].tolist()
    pate_cnts = final_df['Patient_Experience_cnt'].tolist()
    proc_cnts = final_df['Process_cnt'].tolist()
    
    tot_cnts = []
    msg_cnts = []
    grp_cnts = []
    for i, m in enumerate(mort_cnts):
        ct = 0
        ct2 = 0
        if m > 2:
            ct += 1
            ct2 +=1
        if safe_cnts[i] > 2:
            ct += 1
            ct2 += 1
        if read_cnts[i] > 2:
            ct += 1
        if pate_cnts[i] > 2:
            ct += 1
        if proc_cnts[i] > 2:
            ct += 1
            
        tot_cnts.append(ct)
        msg_cnts.append(ct2)
        if ct == 3:
            grp_cnts.append('1) # of groups=3')
        elif ct == 4:
            grp_cnts.append('2) # of groups=4')
        elif ct == 5:
            grp_cnts.append('3) # of groups=5')
        else:
            grp_cnts.append('Not grouped')
    
    final_df['Total_measure_group_cnt'] = tot_cnts
    final_df['MortSafe_Group_cnt'] = msg_cnts
    final_df['cnt_grp'] = grp_cnts
    
    
    # Add standard group measure weights
    weights_info = {}
    if yr == 2026 or yr == 2027: # ALL OF THESE WEIGHTS ARE CORRECT
        final_df['std_weight_PatientExperience'] = 0.22
        final_df['std_weight_Readmission'] = 0.22
        final_df['std_weight_Mortality'] = 0.22
        final_df['std_weight_safety'] = 0.22
        final_df['std_weight_Process'] = 0.12
        
        # Standard weights and their corresponding score columns
        weights_info = {
            'Std_PatientExp_score': ('weight_PatientExperience', 0.22),
            'Std_Outcomes_Readmission_score': ('weight_Outcomes_Readmission', 0.22),
            'Std_Outcomes_Mortality_score': ('weight_Outcomes_Mortality', 0.22),
            'Std_Outcomes_Safety_score': ('weight_Outcomes_Safety', 0.22),
            'Std_Process_score': ('weight_Process', 0.12)
        }
    
    # Function to adjust weights
    def adjust_weights(row):
        # Extract scores and check for NaN
        scores = {score: row[score] for score in weights_info.keys()}
        non_missing_scores = {k: v for k, v in scores.items() if pd.notnull(v)}
        
        # Sum of weights for non-missing scores
        sum_weights = sum(weights_info[k][1] for k in non_missing_scores.keys())
        
        # Assign adjusted weights or 0 if score is missing
        for score, (new_col, weight) in weights_info.items():
            if score in non_missing_scores:
                row[new_col] = weight / sum_weights
            else:
                row[new_col] = 0  # Set weight to 0 if score is missing
        
        return row
    
    # Apply the function to each row
    final_df = final_df.apply(adjust_weights, axis=1)
    
    # Define score columns and their corresponding adjusted weight columns
    score_columns = [
        'Std_PatientExp_score',
        'Std_Outcomes_Readmission_score',
        'Std_Outcomes_Mortality_score',
        'Std_Outcomes_Safety_score',
        'Std_Process_score'
    ]
    weight_columns = [
        'weight_PatientExperience',
        'weight_Outcomes_Readmission',
        'weight_Outcomes_Mortality',
        'weight_Outcomes_Safety',
        'weight_Process'
    ]
    
    # Calculate weighted average for each row
    final_df['summary_score'] = final_df.apply(lambda row: sum(row[score] * row[weight] for score, weight in zip(score_columns, weight_columns) if pd.notnull(row[score])), axis=1)
    
    final_df = final_df[final_df['cnt_grp'] != 'Not grouped']
    final_df = final_df[final_df['MortSafe_Group_cnt'] > 0]
    final_df['report_indicator'] = 1
    
    dfg3 = final_df[final_df['cnt_grp'] == '1) # of groups=3']
    dfg3 = kmeans_clustering(dfg3)
    dfg4 = final_df[final_df['cnt_grp'] == '2) # of groups=4']
    dfg4 = kmeans_clustering(dfg4)
    dfg5 = final_df[final_df['cnt_grp'] == '3) # of groups=5']
    dfg5 = kmeans_clustering(dfg5)
    
    complete_df = pd.concat([dfg3, dfg4, dfg5])
    
    if yr == 2026:
        safety_q1 = complete_df['Std_Outcomes_Safety_score'].quantile(0.25)

        downgrade_mask = (
            (complete_df['star'] == 5) &
            (complete_df['Std_Outcomes_Safety_score'] <= safety_q1)
        )

        complete_df.loc[downgrade_mask, 'star'] = 4

    if yr == 2027:
        safety_q1 = complete_df['Std_Outcomes_Safety_score'].quantile(0.25)

        downgrade_mask = (
            (complete_df['star'] >= 2) &
            (complete_df['Std_Outcomes_Safety_score'] <= safety_q1)
        )

        complete_df.loc[downgrade_mask, 'star'] -= 1
        
    #tdf1 = complete_df[~complete_df['star'].isin([np.nan, float("NaN")])]
    
    return complete_df