#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:51:31 2026

@author: kenlocey
"""

feature_dict = {}
feature_dict['filter categories'] = [
    'State',
    'ZIP Code',
    'County Name',
    'Hospital Type',
    'Hospital Ownership',
    'Emergency Services',
    'Meets criteria for promoting interoperability of EHRs',
    'cnt_grp',
    'star',
    'Beds',
    ]

feature_dict['date categories'] = [
    'file_month',
    'file_year',
    'Release year',
    ]

feature_dict['Standardized scores'] = [
    'Std_Outcomes_Mortality_score',
    'Std_Outcomes_Readmission_score',
    'Std_Outcomes_Safety_score',
    'Std_PatientExp_score',
    'Std_Process_score', 
    'summary_score',
    ]

feature_dict['Stars Domains'] = [
    'Mortality',
    'Readmission',
    'Safety of Care',
    'Patient Experience',
    'Timely and Effective Care',
    ]

feature_dict['Domain weights'] = [
    'std_weight_PatientExperience',
    'std_weight_Readmission',
    'std_weight_Mortality',
    'std_weight_safety',
    'std_weight_Process',
    'weight_PatientExperience',
    'weight_Outcomes_Readmission',
    'weight_Outcomes_Mortality',
    'weight_Outcomes_Safety',
    'weight_Process',
    ]

feature_dict['Domain measure counts'] = [
    'Outcomes_Mortality_cnt',
    'Outcomes_safety_cnt',
    'Outcomes_Readmission_cnt',
    'Patient_Experience_cnt',
    'Process_cnt',
    'Total_measure_group_cnt',
    'MortSafe_Group_cnt',
    ]


feature_dict['Safety of Care'] = [
    'HAI_1', 'HAI_2',
    'HAI_3', 'HAI_4',
    'HAI_5', 'HAI_6',
    'COMP_HIP_KNEE', 
    'PSI_90_SAFETY',
    ]

feature_dict['Safety of Care (std)'] = [
    'std_HAI_1', 'std_HAI_2', 
    'std_HAI_3', 'std_HAI_4', 
    'std_HAI_5', 'std_HAI_6', 
    'std_COMP_HIP_KNEE', 
    'std_PSI_90_SAFETY',
    ]

feature_dict['Safety of Care labels'] = [
    'CLABSI', 
    'CAUTI',
    'SSI Colon', 
    'SSI Abd. Hysterectomy',
    'MRSA Bacteremia', 
    'C. diff. infection',
    'Hip-Knee Complication rate', 
    'PSI-90',
    ]



feature_dict['Readmission'] = [
    'READM_30_HOSP_WIDE',
    'Hybrid_HWR',
    'READM_30_HIP_KNEE',
    'EDAC_30_HF',
    'READM_30_COPD',
    'EDAC_30_AMI',
    'EDAC_30_PN',
    'READM_30_CABG',
    'OP_32',
    'OP_35_ADM',
    'OP_35_ED',
    'OP_36',
    ]

feature_dict['Readmission (std)'] = [
    'std_READM_30_HOSP_WIDE',                                 
    'std_Hybrid_HWR',
    'std_READM_30_HIP_KNEE',
    'std_EDAC_30_HF',
    'std_READM_30_COPD',
    'std_EDAC_30_AMI',
    'std_EDAC_30_PN',
    'std_READM_30_CABG',
    'std_OP_32',
    'std_OP_35_ADM',
    'std_OP_35_ED',
    'std_OP_36',
    ]

feature_dict['Readmission labels'] = [
    '30-Day readmission rate, Hospital-wide',
    '30-Day readmission rate, Hospital-wide', # 30-Day hybrid hospital-wide all-cause readmission rate
    '30-Day readmission rate, HIP KNEE',
    'Excess days in Acute Care, HF',
    '30-Day readmission rate, COPD',
    'Excess days in Acute Care, AMI',
    'Excess days in Acute Care, PN',
    '30-Day readmission rate, CABG',
    '7-Day visit rate after OP colonoscopy',
    'Admissions for Patients Receiving OP Chemo',
    'ED Visits for Patients Receiving OP Chemo',
    'Hospital Visits after OP Surgery',
    ]



feature_dict['Mortality'] = [
    'Hybrid_HWM',
    'MORT_30_STK',
    'MORT_30_PN',
    'MORT_30_HF',
    'MORT_30_COPD',
    'MORT_30_AMI',
    'MORT_30_CABG',
    'PSI_4_SURG_COMP',
    ]
feature_dict['Mortality (std)'] = [
    'std_Hybrid_HWM',
    'std_MORT_30_STK',
    'std_MORT_30_PN',
    'std_MORT_30_HF',
    'std_MORT_30_COPD',
    'std_MORT_30_AMI',
    'std_MORT_30_CABG',
    'std_PSI_4_SURG_COMP',
    ]
feature_dict['Mortality labels'] = [
    'Hospital-Wide All-Cause Risk Standardized Mortality Rate',
    'STK 30-Day Mortality Rate',
    'PN 30-Day Mortality Rate',
    'HF 30-Day Mortality Rate',
    'COPD 30-Day Mortality Rate',
    'AMI 30-Day Mortality Rate',
    'CABG 30-Day Mortality Rate',
    'PSI-04, Death Rate, Surg. Inpatients w/ STCs',
    ]


feature_dict['Patient Experience'] = [
    'H_COMP_1_STAR_RATING',
    'H_COMP_2_STAR_RATING',
    'H_COMP_3_STAR_RATING',
    'H_COMP_5_STAR_RATING',
    'H_COMP_6_STAR_RATING',
    'H_COMP_7_STAR_RATING',
    'H_GLOB_STAR_RATING', # H-HSP-RATING + H-RECMND / 2
    'H_INDI_STAR_RATING', # H-CLEAN-HSP + H-QUIET-HSP / 2
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
    
    'H_COMP_8_LINEAR_SCORE',
    'H_COMP_9_LINEAR_SCORE',
    'H_SYMP_LINEAR_SCORE',
    ]
    
feature_dict['Patient Experience (std)'] = [
    'std_H_COMP_1_STAR_RATING', 
    'std_H_COMP_2_STAR_RATING', 
    'std_H_COMP_3_STAR_RATING', 
    'std_H_COMP_5_STAR_RATING', 
    'std_H_COMP_6_STAR_RATING', 
    'std_H_COMP_7_STAR_RATING', 
    'std_H_GLOB_STAR_RATING', 
    'std_H_INDI_STAR_RATING',
    
    'std_O-COMP-1',
    'std_O-COMP-2',
    'std_O-COMP-3',
    'std_O-PATIENT-RATE',
    'std_O-PATIENT-REC',
    
    'std_H_COMP_1_LINEAR_SCORE',
    'std_H_COMP_2_LINEAR_SCORE',
    'std_H_COMP_3_LINEAR_SCORE',
    'std_H_COMP_5_LINEAR_SCORE',
    'std_H_COMP_6_LINEAR_SCORE',
    'std_H_COMP_7_LINEAR_SCORE',
    'std_H_CLEAN_LINEAR_SCORE',
    'std_H_QUIET_LINEAR_SCORE',
    'std_H_RECMND_LINEAR_SCORE',
    'std_H_HSP_RATING_LINEAR_SCORE',
    
    'std_H_COMP_8_LINEAR_SCORE',
    'std_H_COMP_9_LINEAR_SCORE',
    'std_H_SYMP_LINEAR_SCORE',
    
    ]

feature_dict['Patient Experience labels'] = [
    'Nurse Communication',
    'Doctor Communication',
    'Staff responsiveness',
    'Communication about medicines',
    'Discharge information',
    'Care transition',
    'Hospital rating', # H-HSP-RATING + H-RECMND / 2
    'Cleanliness and Quietness', # H-CLEAN-HSP + H-QUIET-HSP / 2
    
    'Hospital cleanliness (OAS)',
    'Clear and complete information about procedure (OAS)',
    'Written information and preparation for recovery (OAS)',
    'Hospital rating (OAS)',
    'Recommend hospital (OAS)',
    
    'Nurse Communication',
    'Doctor Communication',
    'Staff responsiveness',
    'Communication about medicines',
    'Discharge information',
    'Care transition',
    'Hospital cleanliness',
    'Hospital quietness',
    'Recommend hospital',
    'Hospital rating',
    
    'Restfulness of Hospital Environment',
    'Care Coordination',
    'Info. on symptoms during recovery at home',
    
    ]

                                        
feature_dict['Timely and Effective Care'] = [
    'OP_8',
    'OP_10', 'OP_13', 'OP_18B',
    'OP_22', 'OP_23', 'OP_29',
    'OP_33', 'OP_30', 'IMM_3',
    'PC_01', 'SEP_1', 'ED_2B',
    'HCP_COVID_19',
    'SAFE_USE_OF_OPIOIDS',
    ]

feature_dict['Timely and Effective Care (std)'] = [
    'std_OP_8',
    'std_OP_10', 'std_OP_13', 'std_OP_18B',
    'std_OP_22', 'std_OP_23', 'std_OP_29',
    'std_OP_33', 'std_OP_30', 'std_IMM_3',
    'std_PC_01', 'std_SEP_1', 'std_ED_2B',
    'std_HCP_COVID_19',
    'std_SAFE_USE_OF_OPIOIDS',
    ]

feature_dict['Timely and Effective Care labels'] = [
    'OP-8: MRI Lumbar Spine for Low Back Pain',
    'OP-10: Abdomen CT Use of Contrast Material',
    'OP-13: Cardiac Imaging for Preop Risk for non-cardiac low-risk surg.',
    'OP-18b: Median Time from ED Arrival to ED Departure',
    'OP-22: ED-Patient Left Without Being Seen',
    'OP-23: Received interp. of head CT/MRI for stroke w/in 45 min of arrival',
    'OP-29: Endoscopy/Polyp Surv.: appropriate follow-up int.',
    'OP-33: External Beam Radiotherapy for Bone Metastases',
    'OP-30: Endoscopy/Polyp Surv.: avoidance of inappropriate use',
    'IMM-3: Healthcare Personnel Influenza Vaccination',
    'PC-1: Percent babies elect. del. prior to 39 weeks gestation',
    'SEP-1: Severe Sepsis and Septic Shock',
    'ED-2b: Admit decision time to ED depart time, admitted patients',
    'HCP COVID-19: COVID-19 Vaccination Coverage Among HCP',
    'Safe use of opioids',
    ]