import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
import warnings
import sys
import re
import csv
import math
import random
from datetime import datetime
#import timeit

#import urllib
import numpy as np
#import statsmodels.api as sm
from scipy import stats
#from sklearn.preprocessing import PolynomialFeatures
#from statsmodels.stats.outliers_influence import summary_table

px.set_mapbox_access_token('pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw')

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, 
                external_stylesheets = external_stylesheets,
                )

app.config.suppress_callback_exceptions = True
server = app.server

################################# LOAD DATA ##################################

main_df = pd.read_pickle('dataframe_data/hosp_stars_dat.pkl')
beds_max = np.nanmax(main_df['Beds'])
#for l in list(main_df): print(l)


######################## Create Features Dictionary #####################################

feature_dict = {}
feature_dict['filter categories'] = ['State',
                                     'ZIP Code',
                                     'County Name',
                                     'Hospital Type',
                                     'Hospital Ownership',
                                     'Emergency Services',
                                     'Meets criteria for promoting interoperability of EHRs',
                                     #'Hospital overall rating',  # from previous year
                                     'cnt_grp',
                                     'star',
                                     'Beds',
                                    ]

feature_dict['date categories'] = ['file_month',
                                   'file_year',
                                   'Release year',
                                  ]

feature_dict['Standardized scores'] = ['Std_Outcomes_Mortality_score',
                                       'Std_Outcomes_Readmission_score',
                                       'Std_Outcomes_Safety_score',
                                       'Std_PatientExp_score',
                                       'Std_Process_score', 
                                       'summary_score',
                                      ]

feature_dict['Domain weights'] = ['std_weight_PatientExperience',
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

feature_dict['Domain measure counts'] = ['Outcomes_Mortality_cnt',
                                         'Outcomes_safety_cnt',
                                         'Outcomes_Readmission_cnt',
                                         'Patient_Experience_cnt',
                                         'Process_cnt',
                                         'Total_measure_group_cnt',
                                         'MortSafe_Group_cnt',
                                        ]

feature_dict['Stars Domains'] = ['Patient Experience',
                                 'Readmission',
                                 'Mortality',
                                 'Safety of Care',
                                 'Timely and Effective Care',
                                ]

feature_dict['Safety of Care'] = ['HAI_1_DEN_VOL',
                                  'HAI_2_DEN_VOL',
                                  'HAI_3_DEN_VOL',
                                  'HAI_4_DEN_VOL',
                                  'HAI_5_DEN_VOL',
                                  'HAI_6_DEN_VOL',
                                  'HAI_1_DEN_PRED',
                                  'HAI_2_DEN_PRED',
                                  'HAI_3_DEN_PRED',
                                  'HAI_4_DEN_PRED',
                                  'HAI_5_DEN_PRED',
                                  'HAI_6_DEN_PRED',
                                  'HAI_1',
                                  'HAI_2',
                                  'HAI_3',
                                  'HAI_4',
                                  'HAI_5',
                                  'HAI_6',
                                  'COMP_HIP_KNEE_DEN',
                                  'COMP_HIP_KNEE',
                                 ]

feature_dict['Readmission'] = ['READM_30_HOSP_WIDE',
                               'READM_30_HIP_KNEE',
                               'EDAC_30_HF',
                               'READM_30_COPD',
                               'EDAC_30_AMI',
                               'EDAC_30_PN',
                               'READM_30_CABG',
                               'READM_30_CABG_DEN',
                               'READM_30_HOSP_WIDE_DEN',
                               'READM_30_HIP_KNEE_DEN',
                               'EDAC_30_HF_DEN',
                               'READM_30_COPD_DEN',
                               'EDAC_30_AMI_DEN',
                               'EDAC_30_PN_DEN',
                               'OP_32',
                               'OP_32_DEN',
                               'OP_35_ADM',
                               'OP_35_ADM_DEN',
                               'OP_35_ED',
                               'OP_35_ED_DEN',
                               'OP_36',
                               'OP_36_DEN',
                               'PSI_90_SAFETY',
                               'PSI_90_SAFETY_DEN',
                              ]

feature_dict['Mortality'] = ['MORT_30_STK',
                             'MORT_30_PN',
                             'MORT_30_HF',
                             'MORT_30_COPD',
                             'MORT_30_AMI',
                             'MORT_30_STK_DEN',
                             'MORT_30_PN_DEN',
                             'MORT_30_HF_DEN',
                             'MORT_30_COPD_DEN',
                             'MORT_30_AMI_DEN',
                             'MORT_30_CABG',
                             'MORT_30_CABG_DEN',
                             'PSI_4_SURG_COMP',
                             'PSI_4_SURG_COMP_DEN',
                            ]


feature_dict['Patient Experience'] = ['H_COMP_1_STAR_RATING',
                                      'H_COMP_2_STAR_RATING',
                                      'H_COMP_3_STAR_RATING',
                                      'H_COMP_5_STAR_RATING',
                                      'H_COMP_6_STAR_RATING',
                                      'H_COMP_7_STAR_RATING',
                                      'H_GLOB_STAR_RATING', # H-HSP-RATING + H-RECMND / 2
                                      'H_INDI_STAR_RATING', # H-CLEAN-HSP + H-QUIET-HSP / 2
                                      'H_RESP_RATE_P',
                                      'H_NUMB_COMP',
                                     ]


feature_dict['Timely and Effective Care'] = ['OP_2',
                                             'OP_2_DEN',
                                             'OP_3B',
                                             'OP_3B_DEN',
                                             'OP_8',
                                             'OP_8_DEN',
                                             'OP_10',
                                             'OP_10_DEN',
                                             'OP_13',
                                             'OP_13_DEN',
                                             'OP_18B',
                                             'OP_18B_DEN',
                                             'OP_22',
                                             'OP_22_DEN',
                                             'OP_23',
                                             'OP_23_DEN',
                                             'OP_29',
                                             'OP_29_DEN',
                                             'OP_33',
                                             'OP_33_DEN',
                                             'OP_30',
                                             'OP_30_DEN',
                                             'IMM_3_DEN',
                                             'IMM_3',
                                             'PC_01',
                                             'PC_01_DEN',
                                             'SEP_1',
                                             'SEP_1_DEN',
                                             'ED_2B',
                                             'ED_2B_DEN',
                                            ]



feature_dict['Safety of Care (Std)'] = ['std_COMP_HIP_KNEE', 'std_HAI_1', 'std_HAI_2', 'std_HAI_3', 
                                        'std_HAI_4', 'std_HAI_5', 'std_HAI_6', 'std_PSI_90_SAFETY',
                                 ]

feature_dict['Readmission (Std)'] = ['std_EDAC_30_AMI', 'std_EDAC_30_HF', 'std_EDAC_30_PN', 
                                     'std_OP_32', 'std_READM_30_CABG', 'std_READM_30_COPD', 
                                     'std_READM_30_HIP_KNEE', 'std_READM_30_HOSP_WIDE', 'std_OP_35_ADM',
                                     'std_OP_35_ED', 'std_OP_36',
                              ]

feature_dict['Mortality (Std)'] = ['std_MORT_30_AMI', 'std_MORT_30_CABG', 'std_MORT_30_COPD', 
                                   'std_MORT_30_HF', 'std_MORT_30_PN', 'std_MORT_30_STK', 
                                   'std_PSI_4_SURG_COMP',
                            ]


feature_dict['Patient Experience (Std)'] = ['std_H_COMP_1_STAR_RATING', 'std_H_COMP_2_STAR_RATING', 
                                            'std_H_COMP_3_STAR_RATING', 'std_H_COMP_5_STAR_RATING', 
                                            'std_H_COMP_6_STAR_RATING', 'std_H_COMP_7_STAR_RATING', 
                                            'std_H_GLOB_STAR_RATING', 'std_H_INDI_STAR_RATING',
                                     ]


feature_dict['Timely and Effective Care (Std)'] = ['std_IMM_3', 'std_OP_22', 'std_OP_23', 
                                                   'std_OP_29', 'std_OP_30', 'std_OP_33', 
                                                   'std_PC_01', 'std_SEP_1', 'std_OP_3B', 
                                                   'std_OP_18B', 'std_ED_2B', 'std_OP_8', 
                                                   'std_OP_10', 'std_OP_13',
                                            ]

######################## SELECTION LISTS #####################################
PERFORMANCE_SET = ['None', 'Vizient top performer 2022: Comprehensive academic medical centers']

tdf = main_df[main_df['Vizient top performer 2022: Comprehensive academic medical centers'] == 1]
VIZ_CAMCs = sorted(tdf['Name and Num'].unique().tolist())

HOSPITALS = main_df['Name and Num'].tolist()
beds = main_df['Beds'].tolist()
states = main_df['State'].tolist()
htypes = main_df['Hospital Type'].tolist()
ctypes = main_df['Hospital Ownership'].tolist()
lons = main_df['Lon'].tolist()
lats = main_df['Lat'].tolist()

main_df['Release year'] = main_df['Release year'].astype(int)
latest_yr = np.max(main_df['Release year'])
current_yr = datetime.now().year
current_mo = datetime.now().month

states = ['NaN' if x is np.nan else x for x in states]
htypes = ['NaN' if x is np.nan else x for x in htypes]
ctypes = ['NaN' if x is np.nan else x for x in ctypes]

HOSPITALS_SET = sorted(list(set(HOSPITALS)))

ddfs = "100%"

domains = ['Patient Experience', 'Readmission', 'Mortality', 
           'Safety of Care', 'Timely and Effective Care']

print(len(main_df['Facility ID'].unique()), 'CMS numbers')
print(len(list(set(HOSPITALS))), 'hospitals')

random.seed(42)

COLORS = []
for h in HOSPITALS:
    if 'RUSH UNIVERSITY' in h:
        clr = '#167e04'
    else:
        clr = '#' + "%06x" % random.randint(0, 0xFFFFFF)
    COLORS.append(clr)
    

################# DASH APP CONTROL FUNCTIONS #################################

def description_card1():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("CMS Stars", style={
            'textAlign': 'left',
            }),
           dcc.Markdown("The Centers for Medicare & Medicaid Services (CMS) Overall Hospital Quality Star Ratings " +
                        "provide summary information on existing publicly reported hospital quality data. "),
                        
           
           dcc.Markdown("This first-of-its-kind application enables users to take a deep dive " + 
                        "into CMS Star Ratings by comparing a chosen hospital to its peer group " +
                        "(hospitals it was clustered against) and to a customized set of other hospitals.")
                        
        ],
    )


def generate_control_card1():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card1",
        children=[
            
            html.Br(),
            html.H5("1. Filter on the options below"),
            
            html.Div(id='Filterbeds1'),
            dcc.RangeSlider(
                id='beds1',
                min=0,
                max=2500,
                step=50,
                marks={
                        100: '100',
                        500: '500',
                        1000: '1000',
                        1500: '1500',
                        2000: '2000',
                        2500: 'Max',
                    },
                value=[0, beds_max],
                ),
            
            html.Br(),
            
            dbc.Button("Hospital types",
                       id="open-centered4",
                       style={
                           "background-color": "#2a8cff",
                           'width': '80%',
                               'font-size': 12,
                           'display': 'inline-block',
                           'margin-left': '10%',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                                html.P("Select hospital types",style={'font-size': 16,}),
                                dcc.Dropdown(
                                    id="hospital_type1",
                                    options=[{"label": i, "value": i} for i in sorted(list(set(htypes)))],
                                    multi=True,
                                    value=sorted(list(set(htypes))),
                                    style={
                                        'font-size': 16,
                                        },
                                    ),
                                html.Br(),
                                html.Br(), 
                                ]),
                                dbc.ModalFooter(
                                dbc.Button("Save & Close", id="close-centered4", className="ml-auto", 
                                           style={'font-size': 12,})
                                ),
                        ],
                id="modal-centered4",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            html.Br(),
            html.Br(),
            
            
            dbc.Button("Hospital ownership",
                       id="open-centered1",
                       style={
                           "background-color": "#2a8cff",
                           'width': '80%',
                               'font-size': 12,
                           'display': 'inline-block',
                           'margin-left': '10%',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                                html.P("Select hospital ownership types",style={'font-size': 16,}),
                                dcc.Dropdown(
                                    id="control_type1",
                                    options=[{"label": i, "value": i} for i in sorted(list(set(ctypes)))],
                                    multi=True,
                                    value=sorted(list(set(ctypes))),
                                    style={
                                        #'width': '320px', 
                                        'font-size': 16,
                                        },
                                    ),
                                html.Br(),
                                html.Br(), 
                                ]),
                                dbc.ModalFooter(
                                dbc.Button("Save & Close", id="close-centered1", className="ml-auto",
                                           style={'font-size': 12,})
                                ),
                        ],
                id="modal-centered1",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            html.Br(),
            html.Br(),
            
            
            dbc.Button("US states & territories",
                       id="open-centered3",
                       style={
                           "background-color": "#2a8cff",
                           'width': '80%',
                               'font-size': 12,
                           'display': 'inline-block',
                           'margin-left': '10%',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                                html.P("Select a set of US states and/or territories",style={'font-size': 16,}),
                                dcc.Dropdown(
                                    id="states-select1",
                                    options=[{"label": i, "value": i} for i in sorted(list(set(states)))],
                                    multi=True,
                                    value=sorted(list(set(states))),
                                    style={
                                        'font-size': 16,
                                        }
                                ),
                                html.Br(),
                                html.Br(), 
                                ]),
                                dbc.ModalFooter(
                                dbc.Button("Save & Close", id="close-centered3", className="ml-auto", 
                                           style={'font-size': 12,})
                                ),
                        ],
                id="modal-centered3",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            html.Br(),
            html.Br(),
            
            
            dbc.Button("Vizient & US News",
                       id="open-centered5",
                       style={
                           "background-color": "#2a8cff",
                           'width': '80%',
                               'font-size': 12,
                           'display': 'inline-block',
                           'margin-left': '10%',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                                html.P("Select a performance category. Selecting 'None' will cause the app to ignore this filter. Leaving the selection empty will also cause the app to ignore this filter.",
                                       style={'font-size': 16,}),
                                dcc.Dropdown(
                                    id="performance-select1",
                                    options=[{"label": i, "value": i} for i in PERFORMANCE_SET],
                                    multi=True,
                                    value=None,
                                    optionHeight=50,
                                    style={
                                        'font-size': 14,
                                        }
                                ),
                                html.Br(), 
                                ]),
                                dbc.ModalFooter(
                                dbc.Button("Save & Close", id="close-centered5", className="ml-auto",
                                           style={'font-size': 12,})
                                ),
                        ],
                id="modal-centered5",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            html.Br(),
            html.Br(),
            
            
            dbc.Button("Hospital names & numbers",
                       id="open-centered2",
                       style={
                           "background-color": "#2a8cff",
                           'width': '80%',
                               'font-size': 12,
                           'display': 'inline-block',
                           'margin-left': '10%',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                                html.P("Select a set of hospitals. Not selecting any (i.e., leaving this filter blank) will cause the app to ignore this filter.",
                                       style={'font-size': 16,}),
                                dcc.Dropdown(
                                    id="hospital-select1",
                                    options=[{"label": i, "value": i} for i in HOSPITALS_SET],
                                    multi=True,
                                    value=None,
                                    optionHeight=50,
                                    style={
                                        'font-size': 14,
                                        }
                                ),
                                html.Br(), 
                                ]),
                                dbc.ModalFooter(
                                dbc.Button("Save & Close", id="close-centered2", className="ml-auto",
                                           style={'font-size': 12,})
                                ),
                        ],
                id="modal-centered2",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            html.Br(),
            html.Br(),
            
            
            html.H5("2. Choose a focal hospital", style={
                'display': 'inline-block',
                'width': '400px', 
                
                }),
            dcc.Dropdown(
                id='hospital-select1b',
                options=[{"label": i, "value": i} for i in []],
                value="RUSH UNIVERSITY MEDICAL CENTER (140119)",
                placeholder='Select a focal hospital',
                optionHeight=75,
                style={
                    'width': '265px', 
                    'font-size': 13,
                    #'display': 'inline-block',
                    'border-radius': '15px',
                    'padding': '0px',
                    #'margin-top': '15px',
                    'margin-left': '15px',
                    }
            ),
            html.Br(),
            html.Hr(),
            html.Button("Download data", id="download-btn",
                        style={'width': '80%',
                            'margin-left': '10%',
                            },
                        ),
            dcc.Download(id="data-download"),
            
        ],
    )


#########################################################################################
#############################   DASH APP LAYOUT   #######################################
#########################################################################################    


app.layout = html.Div([
    
    html.Div(
            id='option_hospitals', 
            style={'display': 'none'}
        ),
        
    # Banner
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'})],

        ),
        
    # Left column
    html.Div(
            id="left-column1",
            className="three columns",
            children=[description_card1(), 
                      generate_control_card1(),
                      ],
            style={'width': '24%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
            },
        ),
    
    
    
    # Panel 1 & 2
    html.Div(
            id="right-column1",
            className="eight columns",
            children=[
                
                html.Div(
                    id="map1",
                    children=[
                        html.B("Map of selected hospitals"),
                        html.Hr(),
                        
                        dcc.Loading(
                            id="loading-map1",
                            type="default",
                            fullscreen=False,
                            children=[dcc.Graph(id="map_plot1"),],),
                    ],
                    style={'width': '107%',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
                            },
                ),
                
                
                html.Div(
                    id="data_report1",
                    children=[
                        html.B(id="text9"),
                        
                        dcc.Dropdown(
                            id='year-select1',
                            options=[{"label": i, "value": i} for i in list(range(2021, latest_yr+1))],
                            value=latest_yr,
                            placeholder='Select a Stars year',
                            optionHeight=50,
                            style={
                                'width': '120px', 
                                'font-size': 13,
                                #'display': 'inline-block',
                                'border-radius': '15px',
                                'padding': '0px',
                                #'margin-top': '15px',
                                #'margin-left': '15px',
                                }
                        ),
                        html.Hr(),
                        
                        dcc.Graph(id="figure1"),
                        html.Hr(),
                        html.P(id="text7", 
                               style={'fontSize':16, 
                                      #'display': 'inline-block',
                                      }),
                        html.P(id="text8", 
                               style={'fontSize':16, 
                                      #'display': 'inline-block',
                                      }),
                        
                    ],
                    style={'width': '107%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 #'margin-bottom': '10px',
                                 'height': '551px',
                            },
                ),
                html.Br(),
                html.Br(),
            ],
        ),
    
    
    # Panel 2
    html.Div(
            id="panel-2",
            className="eleven columns",
            children=[
                
                html.Div(
                    id="data_report2",
                    children=[
                        html.H5("Scores across domains for " + str(latest_yr)),
                        dcc.Dropdown(
                            id='set-select1',
                            options=[{"label": i, "value": i} for i in ['Measures group', 'Selected hospitals']],
                            value='Measures group',
                            placeholder='Measures group',
                            optionHeight=75,
                            style={
                                'width': '350px', 
                                'font-size': 16,
                                #'display': 'inline-block',
                                'border-radius': '15px',
                                'padding': '0px',
                                #'margin-top': '15px',
                                #'margin-left': '15px',
                                }
                        ),
                        html.Br(),
                        html.Div(id="data_report_plot2"),
                        html.Hr(),
                        html.B(id="text3", style={'fontSize':16}),
                        html.P(id="text4", style={'fontSize':16}),
                        
                        html.P(id="text1", style={'fontSize':16, 'display': 'inline-block'}),
                        html.P(id="text2", style={'fontSize':16, 'display': 'inline-block'}),
                        
                        ],
                    style={
                        'width': '105%',
                        'display': 'inline-block',
                        'border-radius': '15px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-bottom': '10px',
                        #'height': '780px',
                        },
                ),
                ],
            ),
    
    # Panel 3
    html.Div(
            id="panel-3",
            className="eleven columns",
            children=[
                
                html.Div(
                    id="data_report3",
                    children=[
                        html.H5("Scores within domains for "  + str(latest_yr)),
                        dcc.Dropdown(
                            id='set-select2',
                            options=[{"label": i, "value": i} for i in ['Measures group', 'Selected hospitals']],
                            value='Measures group',
                            placeholder='Measures group',
                            optionHeight=50,
                            style={
                                'width': '250px', 
                                'font-size': 16,
                                'display': 'inline-block',
                                'border-radius': '15px',
                                'padding': '0px',
                                'margin-bottom': '10px',
                                #'margin-left': '15px',
                                }
                        ),
                        
                        dcc.Dropdown(
                            id='domain-select1',
                            options=[{"label": i, "value": i} for i in ['Patient Experience', 'Readmission', 'Mortality', 'Safety of Care', 'Timely and Effective Care']],
                            value='Patient Experience',
                            placeholder='Select a domain',
                            optionHeight=50,
                            style={
                                'width': '250px', 
                                'font-size': 16,
                                'display': 'inline-block',
                                'border-radius': '15px',
                                'padding': '0px',
                                'margin-bottom': '10px',
                                'margin-left': '15px',
                                }
                        ),
                        html.Div(id="data_report_plot3"),
                        html.Hr(),
                        html.B(id="text10", style={'fontSize':16}),
                        html.P(id="text11", style={'fontSize':16}),
                        
                        ],
                    style={
                        'width': '105%',
                        'display': 'inline-block',
                        'border-radius': '15px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-bottom': '10px',
                        #'height': '780px',
                        },
                ),
                ],
            ),
    
    ],
)

  
##############################   Callbacks   ############################################
#########################################################################################

@app.callback(
    Output("modal-centered1", "is_open"),
    [Input("open-centered1", "n_clicks"), Input("close-centered1", "n_clicks")],
    [State("modal-centered1", "is_open")],
)
def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered2", "is_open"),
    [Input("open-centered2", "n_clicks"), Input("close-centered2", "n_clicks")],
    [State("modal-centered2", "is_open")],
)
def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered3", "is_open"),
    [Input("open-centered3", "n_clicks"), Input("close-centered3", "n_clicks")],
    [State("modal-centered3", "is_open")],
)
def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal-centered4", "is_open"),
    [Input("open-centered4", "n_clicks"), Input("close-centered4", "n_clicks")],
    [State("modal-centered4", "is_open")],
)
def toggle_modal4(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered5", "is_open"),
    [Input("open-centered5", "n_clicks"), Input("close-centered5", "n_clicks")],
    [State("modal-centered5", "is_open")],
)
def toggle_modal5(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback( # Updated number of beds text
    Output('Filterbeds1', 'children'),
    [
     Input('beds1', 'value'),
     ],
    )
def update_output1(value):
    
    v1 = value[0]
    v2 = value[1]
    if v2 == 2500:
        v2 = int(beds_max)
    value = [v1, v2]
    
    return 'Number of beds: {}'.format(value)

    
@app.callback(
    [Output("hospital-select1", 'options'),
     Output("hospital-select1b", 'options'),
     Output("option_hospitals", 'children')],
    [Input('beds1', 'value'), 
     Input("close-centered1", "n_clicks"),
     Input("close-centered3", "n_clicks"),
     Input("close-centered4", "n_clicks"),
     Input("close-centered5", "n_clicks"),
     ],
    [
     State('states-select1', 'value'),
     State('hospital_type1', 'value'),
     State('control_type1', 'value'),
     State('performance-select1', 'value'),
     ],
    )
def update_hospitals(bed_range, n_clicks1, n_clicks3, n_clicks4, n_clicks5, states_vals, htype_vals, ctype_vals, perf_types):
    
    #Vizient top performer 2022: Comprehensive academic medical centers
    
    low, high = bed_range
    if high == 2500:
        high = beds_max
        
    hospitals = []
    for i, h in enumerate(HOSPITALS):
        
        b = beds[i]
        s = states[i]
        ht = htypes[i]
        ct = ctypes[i]
        if b >= low and b <= high:
            if s in states_vals and ht in htype_vals:
                if ct in ctype_vals:
                    
                    if perf_types is None or len(perf_types) == 0:
                        hospitals.append(h)
                        
                    elif 'None' in perf_types:
                        hospitals.append(h)
                        
                    elif 'Vizient top performer 2022: Comprehensive academic medical centers' in perf_types:
                        if h in VIZ_CAMCs:
                            hospitals.append(h)
            
    hospitals = sorted(list(set(hospitals)))
    out_ls1 = [{"label": i, "value": i} for i in hospitals]
    
    hospitals_full = sorted(list(set(HOSPITALS)))
    out_ls2 = [{"label": i, "value": i} for i in hospitals_full]
    
    return out_ls1, out_ls2, hospitals



@app.callback(
    Output("data-download", "data"),
    Input("download-btn", "n_clicks"),
    [State('df_tab1', "data")],
    prevent_initial_call=True,
)
def update_download(n_clicks, df):
    
    if df is None:
        return dcc.send_data_frame(main_df.to_csv, "data.csv")
            
    df = pd.read_json(df)
    if df.shape[0] == 0:
        return dcc.send_data_frame(main_df.to_csv, "data.csv")
        
    else:
        tdf = main_df.copy(deep=True)
        cols = list(df)
        
        for i, c in enumerate(cols):
            vals = df[c].tolist()
            
            c = list(eval(c))
            tdf[(c[0], c[1])] = vals
    
    return dcc.send_data_frame(tdf.to_csv, "data.csv")



@app.callback(
     Output("map_plot1", "figure"),
     [Input('beds1', 'value'), 
      Input("close-centered1", "n_clicks"),
      Input("close-centered2", "n_clicks"),
      Input("close-centered3", "n_clicks"),
      Input("close-centered4", "n_clicks"),
      Input("close-centered5", "n_clicks"),
      Input("option_hospitals", 'children'),
      ],
      [State("hospital-select1", "value"),
      ],
    )
def update_map_plot1(beds, n_clicks1, n_clicks2, n_clicks3, n_clicks4, n_clicks5, option_hospitals, selected_hospitals):
    
    figure = go.Figure()
    figure.add_trace(go.Scattermapbox())
    figure.update_layout(
        autosize=True, hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw',
            bearing=0, center=go.layout.mapbox.Center(lat=39, lon=-98),
            pitch=20, zoom=3, style='light',
        )
    )

    figure.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0},)
    
    if option_hospitals is None or option_hospitals == []:
        return figure
    
    if selected_hospitals is None or selected_hospitals == []:
        selected_hospitals = list(option_hospitals)
    
    tdf = main_df[main_df['Name and Num'].isin(selected_hospitals)]
    tdf = tdf.filter(items=['Name and Num', 'Facility ID', 'Lat', 'Lon'], axis=1)
    
    tdf.drop_duplicates(inplace=True)
    
    figure = go.Figure()
    figure.add_trace(go.Scattermapbox(
        lon = tdf['Lon'],
        lat = tdf['Lat'],
        text = tdf['Name and Num'],
        marker = dict(size = 10, color = 'rgb(0, 170, 255)', opacity = 0.8),
        ),)
        
    figure.update_layout(
        autosize=True, hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvY2V5IiwiYSI6ImNrYm9uaWhoYjI0ZDcycW56ZWExODRmYzcifQ.Mb27BYst186G4r5fjju6Pw',
            bearing=0, center=go.layout.mapbox.Center(lat=39, lon=-98),
            pitch=20, zoom=3, style='light',))

    figure.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
    
    return figure




@app.callback(
     [Output("text1", "children"),
      Output("text2", "children"),],
     [Input("hospital-select1b", "value"), 
      Input("close-centered2", "n_clicks"),
      Input("option_hospitals", 'children'),
      ],
     [State("hospital-select1", "value"),
      ],
    )
def update_panel1(hospital, n_clicks, option_hospitals, selected_hospitals):    
    
    tdf_main = []
    
    if hospital is None or option_hospitals is None or option_hospitals == []:
        return '', ''
    
    if hospital in option_hospitals:
        '''Remove focal hospital from options, for use below'''
        option_hospitals.remove(hospital)
        
    if selected_hospitals is None or selected_hospitals == [] or selected_hospitals == [hospital]:
        '''There are options but either none were selected specifically by name or only our 
        focal hospital was selected by name'''
        selected_hospitals = list(option_hospitals)
    
    if hospital in selected_hospitals:
        '''Remove focal hospital from options, for use below'''
        selected_hospitals.remove(hospital)
    
    
    ''' At this point, we can still get results even selected_hospitals and option_hospitals are empty lists. 
        That is, even if we've ended up with nothing to compare our hospital to.
    '''
       
         
    tdf_main = main_df[main_df['Name and Num'].isin(selected_hospitals + [hospital])]
            
    name = hospital[:-9]
    txt = str(name) + " "
        
    tdf = tdf_main[tdf_main['Name and Num'] == hospital]
    tdf2 = tdf[tdf['star'].isin([1, 2, 3, 4, 5])]
        
    if tdf2.shape[0] == 0:
        ''' indicates no star rating for the focal hospital in the data '''
        
        txt += "did not receive a star rating in any year of the CMS Overall Hospital Quality Star Ratings included in this application (2020 to " + str(latest_yr) + "). "
        return txt, ''
    
    if tdf_main['Name and Num'].unique().tolist() == [hospital]:
        ''' indicates that the focal hospital is the only hospital in the data.
            Result of faulty filtering on the user side '''
        return txt, ''
    
    max_yr = np.max(tdf['Release year'])
    tdf3 = tdf[tdf['Release year'] == max_yr]
    latest_rating = tdf3['star'].iloc[0]
        
    max_yr_with_star = np.max(tdf2['Release year'])
    tdf4 = tdf2[tdf2['Release year'] == max_yr_with_star]
    max_yr_star = tdf4['star'].iloc[0]
        
    if max_yr != latest_yr:
        txt += "was not included in the latest available release (" + str(latest_yr) + ") of the CMS Overall Hospital Quality Star Ratings. "
            
        if latest_rating not in [1, 2, 3, 4, 5]:
            txt += name + " was included as recently as " + str(max_yr) + " but did not recieve a star rating. "
            return txt, ''
            
        elif latest_rating in [1, 2, 3, 4, 5]:
            txt += name + " was included as recently as " + str(max_yr) + " where it recieved a star rating of " + str(latest_rating) + ". "
            return txt, ''
        
    elif max_yr_with_star != latest_yr:   
        txt += " was included in the latest available release (" + str(latest_yr) + ") of the CMS Overall Hospital Quality Star Ratings, but did not receive a star rating. "
        txt += "However, " + name + " did recieve a star rating of " + str(max_yr_star) + " in " + str(max_yr_with_star) + ". "
        return txt, ''
            
    elif max_yr_with_star == latest_yr:   

        grp = tdf4['cnt_grp'].iloc[0]
        score = tdf4['summary_score'].iloc[0]
        
        df_chosen_latest = tdf_main[tdf_main['Release year'] == latest_yr]
        perc_chosen = round(stats.percentileofscore(df_chosen_latest['summary_score'], score), 1)
        
        df_all_latest = main_df[main_df['Release year'] == latest_yr]
        perc_all = round(stats.percentileofscore(df_all_latest['summary_score'], score), 1)
        
        df_peer = main_df[main_df['cnt_grp'] == grp]
        df_peer_latest = df_peer[df_peer['Release year'] == latest_yr]
        perc_peer = round(stats.percentileofscore(df_peer_latest['summary_score'], score), 1)
         
        grp = int(grp)
        score = round(score, 4)
            
        name1 = str(name)
        if name == 'RUSH UNIVERSITY MEDICAL CENTER':
            name1 = 'RUMC'
        elif name == 'RUSH OAK PARK HOSPITAL':
            name1 = 'ROPH'
        elif name == 'RUSH COPLEY':
            name1 = 'RUSH COPLEY'
        else:
            name1 = "hospital " + hospital[-7:-1]
        
        numD = ' hospitals w/ scores in ' + str(int(grp + 2)) + ' domains'
        
        txt1 = " In " + str(latest_yr) + ", " + name1 + " received a star rating of " + str(int(max_yr_star)) + " "
        txt1 += " and had an overall standardized score of " + str(score) + ". This score put "
            
        if df_chosen_latest.shape[0] == df_all_latest.shape[0]:
            txt1 += name1 + " in the " + str(perc_peer) + " percentile of its peer group (" + str(int(grp)) + ': ' + numD + ") and in the " + str(perc_all)
            txt1 += " percentile of all hospitals. "
        else:
            txt1 += name1 + " in the " + str(perc_peer) + " percentile of its peer group (" + str(int(grp)) + ': ' + numD + "), in the " + str(perc_all)
            txt1 += " percentile of all hospitals, and in the " + str(perc_chosen) + " percentile of the " + str(len(selected_hospitals)) + " other hospitals you selected. "
            
        
        tdf = main_df[main_df['Release year'] == max_yr_with_star]
        tdf = tdf[tdf['cnt_grp'] == grp]
                      
        hosp_ls = tdf['Name and Num'].tolist()
        mort_ls = tdf['Std_Outcomes_Mortality_score'].tolist()
        safe_ls = tdf['Std_Outcomes_Safety_score'].tolist()
        read_ls = tdf['Std_Outcomes_Readmission_score'].tolist()
        pexp_ls = tdf['Std_PatientExp_score'].tolist()
        proc_ls = tdf['Std_Process_score'].tolist()
        
        i = hosp_ls.index(hospital)
        
        mort_scor = mort_ls[i]
        safe_scor = safe_ls[i]
        read_scor = read_ls[i]
        pexp_scor = pexp_ls[i]
        proc_scor = proc_ls[i]
        
        mort_perc = round(stats.percentileofscore(mort_ls, mort_scor), 1)
        safe_perc = round(stats.percentileofscore(safe_ls, safe_scor), 1)
        read_perc = round(stats.percentileofscore(read_ls, read_scor), 1)
        pexp_perc = round(stats.percentileofscore(pexp_ls, pexp_scor), 1)
        proc_perc = round(stats.percentileofscore(proc_ls, proc_scor), 1)
        
        domains = ['Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
        domain_cols = ['Std_Outcomes_Mortality_score', 'Std_Outcomes_Safety_score',
                       'Std_Outcomes_Readmission_score', 'Std_PatientExp_score',	
                       'Std_Process_score']
        
        perc_ls = [mort_perc, safe_perc, read_perc, pexp_perc, proc_perc]
        scor_ls = [mort_scor, safe_scor, read_scor, pexp_scor, proc_scor]
        
        i = perc_ls.index(max(perc_ls))
        best_domain = domains[i]
        best_domain_score = scor_ls[i]
        best_domain_perc1 = perc_ls[i]
        
        tdf = tdf_main[tdf_main['Release year'] == max_yr]
        scores_ls = tdf[domain_cols[i]].tolist()
        best_domain_perc2 = stats.percentileofscore(scores_ls, best_domain_score)
        
        i = perc_ls.index(min(perc_ls))
        worst_domain = domains[i]
        worst_domain_score = scor_ls[i]
        worst_domain_perc1 = perc_ls[i]
        
        scores_ls = tdf[domain_cols[i]].tolist()
        worst_domain_perc2 = stats.percentileofscore(scores_ls, worst_domain_score)
        
        best_domain_score = str(round(best_domain_score, 3))
        best_domain_perc1 = str(round(best_domain_perc1, 1))
        best_domain_perc2 = str(round(best_domain_perc2, 1))
        
        worst_domain_score = str(round(worst_domain_score, 3))
        worst_domain_perc1 = str(round(worst_domain_perc1, 1))
        worst_domain_perc2 = str(round(worst_domain_perc2, 1))
        
        if df_chosen_latest.shape[0] == df_all_latest.shape[0]:
            txt2 = "In " + str(latest_yr) + ", " + name1 
            txt2 += "'s best performing domain was " + best_domain + " where its standardized score (" 
            txt2 += best_domain_score + ") was in the " + best_domain_perc1 + " percentile of its peer group (" + str(int(grp)) + "). In contrast, "
            txt2 += name1 + "'s worst performing domain was " + worst_domain + " where its standardized score ("
            txt2 += worst_domain_score + ") was in the " + worst_domain_perc1 + " percentile of its peer group." 
    
        else:
            txt2 = "In " + str(latest_yr) + ", " + name1 
            txt2 += "'s best performing domain was " + best_domain + " where its standardized score (" 
            txt2 += best_domain_score + ") was in the " + best_domain_perc1 + " percentile of its peer group (" + str(int(grp)) + "), "
            txt2 += "and in the " + best_domain_perc2 + " percentile of the " + str(len(selected_hospitals)) + " other hospitals you selected. "
            
            txt2 += "In contrast, "
            txt2 += name1 + "'s worst performing domain was " + worst_domain + " where its standardized score ("
            txt2 += worst_domain_score + ") was in the " + worst_domain_perc1 + " percentile of its peer group, " 
            txt2 += "and in the " + worst_domain_perc2 + " percentile of the " + str(len(selected_hospitals)) + " other hospitals you selected."
            
        return txt1, txt2
    


@app.callback(
     [Output("data_report_plot2", "children"),
      Output("text3", "children"),
      Output("text4", "children"),],
     [Input("hospital-select1b", "value"), 
      Input("close-centered2", "n_clicks"),
      Input("option_hospitals", 'children'),
      Input('set-select1', "value"),
      ],
     [State("hospital-select1", "value"),
      ],
    )
def update_panel2(hospital, n_clicks, option_hospitals, set_select, selected_hospitals):    
    
    if set_select == 'Measures group':
        
        cols = ['Domain', 'Value', 'Change in value from previous year', 
                'Group percentile', 'Change in percentile', 
                'Weight', 'Change in weight']
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
        
        if hospital is None:
                
            txt3 = "Please select a focal hospital."
            txt4 = ""
            return dashT, txt3, txt4
        
        tdf_main = main_df.copy(deep=True)
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt3 = hospital + " had no data among the CMS Stars release years"
            txt4 = "Try selecting another hospital"
            return dashT, txt3, txt4
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        
        if len(yrs) > 1:
            tdf_main_LY = tdf_main[tdf_main['Release year'] == yrs[-1]]
            grp_LY = tdf_main_LY[tdf_main_LY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
            tdf_main_LY = tdf_main_LY[tdf_main_LY['cnt_grp'].isin([grp_LY])]
            
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yrs[-2]]
            grp_PY = tdf_main_PY[tdf_main_PY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
            tdf_main_PY = tdf_main_PY[tdf_main_PY['cnt_grp'].isin([grp_PY])]
            
            # Get values for latest year
            summ_ls_LY = tdf_main_LY['summary_score'].tolist()
            hosp_ls_LY = tdf_main_LY['Name and Num'].tolist()
            mort_ls_LY = tdf_main_LY['Std_Outcomes_Mortality_score'].tolist()
            safe_ls_LY = tdf_main_LY['Std_Outcomes_Safety_score'].tolist()
            read_ls_LY = tdf_main_LY['Std_Outcomes_Readmission_score'].tolist()
            pexp_ls_LY = tdf_main_LY['Std_PatientExp_score'].tolist()
            proc_ls_LY = tdf_main_LY['Std_Process_score'].tolist()
            mort_ws_LY = tdf_main_LY['weight_Outcomes_Mortality'].tolist()
            safe_ws_LY = tdf_main_LY['weight_Outcomes_Safety'].tolist()
            read_ws_LY = tdf_main_LY['weight_Outcomes_Readmission'].tolist()
            pexp_ws_LY = tdf_main_LY['weight_PatientExperience'].tolist()
            proc_ws_LY = tdf_main_LY['weight_Process'].tolist()
            	
            i = 0
            try:
                i = hosp_ls_LY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            summ_scor_LY = summ_ls_LY[i]
            mort_scor_LY = mort_ls_LY[i]
            safe_scor_LY = safe_ls_LY[i]
            read_scor_LY = read_ls_LY[i]
            pexp_scor_LY = pexp_ls_LY[i]
            proc_scor_LY = proc_ls_LY[i]
            mort_wt_LY = mort_ws_LY[i]
            safe_wt_LY = safe_ws_LY[i]
            read_wt_LY = read_ws_LY[i]
            pexp_wt_LY = pexp_ws_LY[i]
            proc_wt_LY = proc_ws_LY[i]
            
            summ_perc_LY = round(stats.percentileofscore(summ_ls_LY, summ_scor_LY), 1)
            mort_perc_LY = round(stats.percentileofscore(mort_ls_LY, mort_scor_LY), 1)
            safe_perc_LY = round(stats.percentileofscore(safe_ls_LY, safe_scor_LY), 1)
            read_perc_LY = round(stats.percentileofscore(read_ls_LY, read_scor_LY), 1)
            pexp_perc_LY = round(stats.percentileofscore(pexp_ls_LY, pexp_scor_LY), 1)
            proc_perc_LY = round(stats.percentileofscore(proc_ls_LY, proc_scor_LY), 1)
            
            
            # Get values for next latest year
            summ_ls_PY = tdf_main_PY['summary_score'].tolist()
            hosp_ls_PY = tdf_main_PY['Name and Num'].tolist()
            mort_ls_PY = tdf_main_PY['Std_Outcomes_Mortality_score'].tolist()
            safe_ls_PY = tdf_main_PY['Std_Outcomes_Safety_score'].tolist()
            read_ls_PY = tdf_main_PY['Std_Outcomes_Readmission_score'].tolist()
            pexp_ls_PY = tdf_main_PY['Std_PatientExp_score'].tolist()
            proc_ls_PY = tdf_main_PY['Std_Process_score'].tolist()
            mort_ws_PY = tdf_main_PY['weight_Outcomes_Mortality'].tolist()
            safe_ws_PY = tdf_main_PY['weight_Outcomes_Safety'].tolist()
            read_ws_PY = tdf_main_PY['weight_Outcomes_Readmission'].tolist()
            pexp_ws_PY = tdf_main_PY['weight_PatientExperience'].tolist()
            proc_ws_PY = tdf_main_PY['weight_Process'].tolist()
            
            i = hosp_ls_PY.index(hospital)
            
            summ_scor_PY = summ_ls_PY[i]
            mort_scor_PY = mort_ls_PY[i]
            safe_scor_PY = safe_ls_PY[i]
            read_scor_PY = read_ls_PY[i]
            pexp_scor_PY = pexp_ls_PY[i]
            proc_scor_PY = proc_ls_PY[i]
            mort_wt_PY = mort_ws_PY[i]
            safe_wt_PY = safe_ws_PY[i]
            read_wt_PY = read_ws_PY[i]
            pexp_wt_PY = pexp_ws_PY[i]
            proc_wt_PY = proc_ws_PY[i]
            
            summ_perc_PY = round(stats.percentileofscore(summ_ls_PY, summ_scor_PY), 1)
            mort_perc_PY = round(stats.percentileofscore(mort_ls_PY, mort_scor_PY), 1)
            safe_perc_PY = round(stats.percentileofscore(safe_ls_PY, safe_scor_PY), 1)
            read_perc_PY = round(stats.percentileofscore(read_ls_PY, read_scor_PY), 1)
            pexp_perc_PY = round(stats.percentileofscore(pexp_ls_PY, pexp_scor_PY), 1)
            proc_perc_PY = round(stats.percentileofscore(proc_ls_PY, proc_scor_PY), 1)
            
            
            # compute values for columns
            domains = ['Summary Score', 'Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
            
            values_LY = [summ_scor_LY, mort_scor_LY, safe_scor_LY, read_scor_LY, pexp_scor_LY, proc_scor_LY]
            values_PY = [summ_scor_PY, mort_scor_PY, safe_scor_PY, read_scor_PY, pexp_scor_PY, proc_scor_PY]
            delta_value = np.round(np.array(values_LY) - np.array(values_PY), 4)
            
            perc_LY = [summ_perc_LY, mort_perc_LY, safe_perc_LY, read_perc_LY, pexp_perc_LY, proc_perc_LY]
            perc_PY = [summ_perc_PY, mort_perc_PY, safe_perc_PY, read_perc_PY, pexp_perc_PY, proc_perc_PY]
            delta_perc = np.round(np.array(perc_LY) - np.array(perc_PY), 4)
            
            wght_LY = [mort_wt_LY, safe_wt_LY, read_wt_LY, pexp_wt_LY, proc_wt_LY] 
            wght_PY = [mort_wt_PY, safe_wt_PY, read_wt_PY, pexp_wt_PY, proc_wt_PY] 
            delta_wght = np.round(np.array(wght_LY) - np.array(wght_PY), 4)
            
            cols = ['Domain', 'Value', 'Delta value', 
                    'Group percentile', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Domain'] = domains
            df_table['Value'] = np.round(values_LY, 4)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Group percentile'] = perc_LY
            df_table['Delta percentile'] = delta_perc
            df_table['Weight'] = ['N/A'] + wght_LY
            df_table['Delta weight'] = ['N/A'] + delta_wght.tolist()
            
            
            dashT = dash_table.DataTable(
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                export_format="csv",
                page_action='none',
                sort_action="native",
                sort_mode="multi",
                #filter_action="native",
                
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
            
            name1 = str(name) 
            if name == 'RUSH UNIVERSITY MEDICAL CENTER':
                name1 = 'RUMC'
            elif name == 'RUSH OAK PARK HOSPITAL':
                name1 = 'ROPH'
            elif name == 'RUSH COPLEY':
                name1 = 'RUSH COPLEY'
            else:
                name1 = "hospital " + hospital[-7:-1]
            
            txt3 = "The latest year of data used was " + str(int(yrs[-1])) + ". "
            txt3 += "Delta values were computed using the prior year " + str(int(yrs[-2])) + ". "
            txt4 = ''
            
            if np.isnan(grp_LY) == True and np.isnan(grp_PY) == True:
                txt3 += "In both years, " + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_LY) == True:
                txt3 += "In " + str(int(yrs[-1])) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_PY) == True:
                txt3 += "In " + str(int(yrs[-2])) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            
            elif grp_LY == grp_PY:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                txt3 += "In both years, " + name1 + " was in group " + str(int(grp_LY)) + numD_LY + '.'
            else:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                numD_PY = ' (hospitals w/ scores in ' + str(int(grp_PY + 2)) + ' domains)'
                
                txt3 += name1 + " was in group " + str(int(grp_PY)) + numD_PY + " in " + str(int(yrs[-2]))
                txt3 += " and in group " + str(int(grp_LY)) + numD_LY + " in " + str(int(yrs[-1])) + ". "
                
            return dashT, txt3, txt4
        
        else:
            return dashT, 'Stuff', 'More stuff'
        
    elif set_select == 'Selected hospitals':

        cols = ['Domain', 'Value', 'Change in value from previous year', 
                'Percentile', 'Change in percentile', 
                'Weight', 'Change in weight']
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
        
        if hospital is None or option_hospitals is None or option_hospitals == []:
                
            txt3 = ''
            if hospital is None:
                txt3 = "Please select a focal hospital."
            if option_hospitals is None or option_hospitals == []:
                txt3 += "You either haven't selected any hospitals or the filters you chose left you with no hospitals to analyze."
            
            txt4 = ""
            return dashT, txt3, txt4
        
    
        if hospital in option_hospitals:
            # Remove focal hospital from options, for use below
            option_hospitals.remove(hospital)
            
        if selected_hospitals is None or selected_hospitals == [] or selected_hospitals == [hospital]:
            #There are options but either none were selected specifically by name or only our focal hospital was selected by name
            selected_hospitals = list(option_hospitals)
        
        if hospital in selected_hospitals:
            #Remove focal hospital from options, for use below
            selected_hospitals.remove(hospital)
        
        
        # At this point, we can still get results even selected_hospitals and option_hospitals are empty lists. 
        #    That is, even if we've ended up with nothing to compare our hospital to.
           
        tdf_main = main_df[main_df['Name and Num'].isin(selected_hospitals + [hospital])]
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt3 = hospital + " had no data among the CMS Stars release years"
            txt4 = "Try selecting another hospital"
            return dashT, txt3, txt4
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        
        if len(yrs) > 1:
            tdf_main_LY = tdf_main[tdf_main['Release year'] == yrs[-1]]
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yrs[-2]]
            
            # Get values for latest year
            summ_ls_LY = tdf_main_LY['summary_score'].tolist()
            hosp_ls_LY = tdf_main_LY['Name and Num'].tolist()
            mort_ls_LY = tdf_main_LY['Std_Outcomes_Mortality_score'].tolist()
            safe_ls_LY = tdf_main_LY['Std_Outcomes_Safety_score'].tolist()
            read_ls_LY = tdf_main_LY['Std_Outcomes_Readmission_score'].tolist()
            pexp_ls_LY = tdf_main_LY['Std_PatientExp_score'].tolist()
            proc_ls_LY = tdf_main_LY['Std_Process_score'].tolist()
            mort_ws_LY = tdf_main_LY['weight_Outcomes_Mortality'].tolist()
            safe_ws_LY = tdf_main_LY['weight_Outcomes_Safety'].tolist()
            read_ws_LY = tdf_main_LY['weight_Outcomes_Readmission'].tolist()
            pexp_ws_LY = tdf_main_LY['weight_PatientExperience'].tolist()
            proc_ws_LY = tdf_main_LY['weight_Process'].tolist()
            	
            i = 0
            try:
                i = hosp_ls_LY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            summ_scor_LY = summ_ls_LY[i]
            mort_scor_LY = mort_ls_LY[i]
            safe_scor_LY = safe_ls_LY[i]
            read_scor_LY = read_ls_LY[i]
            pexp_scor_LY = pexp_ls_LY[i]
            proc_scor_LY = proc_ls_LY[i]
            mort_wt_LY = mort_ws_LY[i]
            safe_wt_LY = safe_ws_LY[i]
            read_wt_LY = read_ws_LY[i]
            pexp_wt_LY = pexp_ws_LY[i]
            proc_wt_LY = proc_ws_LY[i]
            
            summ_perc_LY = round(stats.percentileofscore(summ_ls_LY, summ_scor_LY), 1)
            mort_perc_LY = round(stats.percentileofscore(mort_ls_LY, mort_scor_LY), 1)
            safe_perc_LY = round(stats.percentileofscore(safe_ls_LY, safe_scor_LY), 1)
            read_perc_LY = round(stats.percentileofscore(read_ls_LY, read_scor_LY), 1)
            pexp_perc_LY = round(stats.percentileofscore(pexp_ls_LY, pexp_scor_LY), 1)
            proc_perc_LY = round(stats.percentileofscore(proc_ls_LY, proc_scor_LY), 1)
            
            
            # Get values for next latest year
            summ_ls_PY = tdf_main_PY['summary_score'].tolist()
            hosp_ls_PY = tdf_main_PY['Name and Num'].tolist()
            mort_ls_PY = tdf_main_PY['Std_Outcomes_Mortality_score'].tolist()
            safe_ls_PY = tdf_main_PY['Std_Outcomes_Safety_score'].tolist()
            read_ls_PY = tdf_main_PY['Std_Outcomes_Readmission_score'].tolist()
            pexp_ls_PY = tdf_main_PY['Std_PatientExp_score'].tolist()
            proc_ls_PY = tdf_main_PY['Std_Process_score'].tolist()
            mort_ws_PY = tdf_main_PY['weight_Outcomes_Mortality'].tolist()
            safe_ws_PY = tdf_main_PY['weight_Outcomes_Safety'].tolist()
            read_ws_PY = tdf_main_PY['weight_Outcomes_Readmission'].tolist()
            pexp_ws_PY = tdf_main_PY['weight_PatientExperience'].tolist()
            proc_ws_PY = tdf_main_PY['weight_Process'].tolist()
            
            i = hosp_ls_PY.index(hospital)
            
            summ_scor_PY = summ_ls_PY[i]
            mort_scor_PY = mort_ls_PY[i]
            safe_scor_PY = safe_ls_PY[i]
            read_scor_PY = read_ls_PY[i]
            pexp_scor_PY = pexp_ls_PY[i]
            proc_scor_PY = proc_ls_PY[i]
            mort_wt_PY = mort_ws_PY[i]
            safe_wt_PY = safe_ws_PY[i]
            read_wt_PY = read_ws_PY[i]
            pexp_wt_PY = pexp_ws_PY[i]
            proc_wt_PY = proc_ws_PY[i]
            
            summ_perc_PY = round(stats.percentileofscore(summ_ls_PY, summ_scor_PY), 1)
            mort_perc_PY = round(stats.percentileofscore(mort_ls_PY, mort_scor_PY), 1)
            safe_perc_PY = round(stats.percentileofscore(safe_ls_PY, safe_scor_PY), 1)
            read_perc_PY = round(stats.percentileofscore(read_ls_PY, read_scor_PY), 1)
            pexp_perc_PY = round(stats.percentileofscore(pexp_ls_PY, pexp_scor_PY), 1)
            proc_perc_PY = round(stats.percentileofscore(proc_ls_PY, proc_scor_PY), 1)
            
            
            # compute values for columns
            domains = ['Summary Score', 'Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
            
            values_LY = [summ_scor_LY, mort_scor_LY, safe_scor_LY, read_scor_LY, pexp_scor_LY, proc_scor_LY]
            values_PY = [summ_scor_PY, mort_scor_PY, safe_scor_PY, read_scor_PY, pexp_scor_PY, proc_scor_PY]
            delta_value = np.round(np.array(values_LY) - np.array(values_PY), 4)
            
            perc_LY = [summ_perc_LY, mort_perc_LY, safe_perc_LY, read_perc_LY, pexp_perc_LY, proc_perc_LY]
            perc_PY = [summ_perc_PY, mort_perc_PY, safe_perc_PY, read_perc_PY, pexp_perc_PY, proc_perc_PY]
            delta_perc = np.round(np.array(perc_LY) - np.array(perc_PY), 4)
            
            wght_LY = [mort_wt_LY, safe_wt_LY, read_wt_LY, pexp_wt_LY, proc_wt_LY] 
            wght_PY = [mort_wt_PY, safe_wt_PY, read_wt_PY, pexp_wt_PY, proc_wt_PY] 
            delta_wght = np.round(np.array(wght_LY) - np.array(wght_PY), 4)
            
            cols = ['Domain', 'Value', 'Delta value', 
                    'Percentile', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Domain'] = domains
            df_table['Value'] = np.round(values_LY, 4)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Percentile'] = perc_LY
            df_table['Delta percentile'] = delta_perc
            df_table['Weight'] = ['N/A'] + wght_LY
            df_table['Delta weight'] = ['N/A'] + delta_wght.tolist()
            
            
            dashT = dash_table.DataTable(
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                export_format="csv",
                page_action='none',
                sort_action="native",
                sort_mode="multi",
                #filter_action="native",
                
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
            
            name1 = str(name) 
            if name == 'RUSH UNIVERSITY MEDICAL CENTER':
                name1 = 'RUMC'
            elif name == 'RUSH OAK PARK HOSPITAL':
                name1 = 'ROPH'
            elif name == 'RUSH COPLEY':
                name1 = 'RUSH COPLEY'
            else:
                name1 = "Hospital " + hospital[-7:-1]
            
            txt3 = "The latest year of data used was " + str(int(yrs[-1])) + ". "
            txt3 += "Delta values were computed using the prior year " + str(int(yrs[-2])) + ". "
            txt4 = ''
            
            return dashT, txt3, txt4
        
        else:
            return dashT, '', ''


    


@app.callback(
     [Output("figure1", "figure"),
      Output("text7", "children"),
      Output("text8", "children"),
      Output("text9", "children")],
     [Input("hospital-select1b", "value"),
      Input('year-select1', 'value'),
      ],
    )
def update_panel3(hospital, yr):    
    
    fig = go.Figure(data=go.Scatter(x = [0], y = [0]))

    fig.update_yaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))
    fig.update_xaxes(title_font=dict(size=14, 
                                     #family='sans-serif', 
                                     color="rgb(38, 38, 38)"))

    fig.update_layout(title_font=dict(size=14, 
                      color="rgb(38, 38, 38)", 
                      ),
                      showlegend=True,
                      height=439,
                      margin=dict(l=100, r=10, b=10, t=10),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      )
    
    if hospital is None:
        
        return fig, '7', '8', "Please select a focal hospital."
    
        
    tdf_main = main_df.copy(deep=True)
                
    name = hospital[:-9]
    hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
    if hosp_df.shape[0] == 0: 
        name1 = str(name) 
        if name == 'RUSH UNIVERSITY MEDICAL CENTER':
            name1 = 'RUMC'
        elif name == 'RUSH OAK PARK HOSPITAL':
            name1 = 'ROPH'
        else:
            name1 = "Hospital " + hospital[-7:-1]
        return fig, '7', '8', name1 + " did not receive a star rating in any year included in this app (2020 to " + str(latest_yr) + ")"
        
    #yrs = sorted(hosp_df['Release year'].unique().tolist())
    
    tdf_main = tdf_main[tdf_main['Release year'] == yr]
    grp = tdf_main[tdf_main['Name and Num'] == hospital]['cnt_grp'].iloc[0]
    
    if np.isnan(grp) == True:
        name1 = str(name) 
        if name == 'RUSH UNIVERSITY MEDICAL CENTER':
            name1 = 'RUMC'
        elif name == 'RUSH OAK PARK HOSPITAL':
            name1 = 'ROPH'
        else:
            name1 = "Hospital " + hospital[-7:-1]
        return fig, '7', '8', name1 + " did not receive a star rating in any year included in this app (2020 to " + str(latest_yr) + ")"
        
    grp = int(grp)
    
    tdf_main = tdf_main[tdf_main['cnt_grp'].isin([grp])]
            
    # Get values for latest year
    star_ls = tdf_main['star'].tolist()
    summ_ls = tdf_main['summary_score'].tolist()
    hosp_ls = tdf_main['Name and Num'].tolist()
    mort_ls = tdf_main['Std_Outcomes_Mortality_score'].tolist()
    safe_ls = tdf_main['Std_Outcomes_Safety_score'].tolist()
    read_ls = tdf_main['Std_Outcomes_Readmission_score'].tolist()
    pexp_ls = tdf_main['Std_PatientExp_score'].tolist()
    proc_ls = tdf_main['Std_Process_score'].tolist()
          	
    i = hosp_ls.index(hospital)
    star_scor = star_ls[i]
    summ_scor = summ_ls[i]
    mort_scor = mort_ls[i]
    safe_scor = safe_ls[i]
    read_scor = read_ls[i]
    pexp_scor = pexp_ls[i]
    proc_scor = proc_ls[i]
    
    tdf_main['star'] = tdf_main['star'].astype(int)
    
    ################### PLAY CODE

    tdf_main.sort_values(by=['star'], inplace=True, ascending = True)
    
    # Create the histogram figure
    fig = px.histogram(tdf_main, 
                       x="summary_score", 
                       color="star",
                       marginal="box",  # or violin, rug
                       #hover_data=tdf_main.columns,
                       nbins=int(tdf_main.shape[0]/8),
                       )
    
    # Add vertical lines
    line_positions = [summ_scor]  # Specify the x-axis positions for the lines
    line_colors = ['black']  # Specify the colors for each line
    
    counts, bins = np.histogram(tdf_main["summary_score"], bins=int(tdf_main.shape[0]/8), density=False)
    max_count = 1.5*max(counts)

    for pos, color in zip(line_positions, line_colors):
        fig.add_shape(
            type="line",
            x0=pos,
            y0=0,
            x1=pos,
            y1=max_count,  # Set the y-coordinate of the line to the maximum y-value of the histogram
            line=dict(
                color=color,
                width=2,
                dash="dash",  # You can change this to "solid" if you want a solid line
            )
        )
    
    # Update layout
    fig.update_layout(
        #title="Histogram with Vertical Lines",
        xaxis_title="Stars summary score",
        yaxis_title="No. of hospitals",
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        #legend=dict(traceorder="normal"),
        height=425,
        paper_bgcolor="#f0f0f0",
        plot_bgcolor="#f0f0f0",
    )
    
    name1 = str(name) 
    if name == 'RUSH UNIVERSITY MEDICAL CENTER':
        name1 = 'RUMC'
    elif name == 'RUSH OAK PARK HOSPITAL':
        name1 = 'ROPH'
    elif name == 'RUSH COPLEY':
        name1 = 'RUSH COPLEY'
    else:
        name1 = "Hospital " + hospital[-7:-1]
    
    txt_a = name1 + "'s score: " + str(round(summ_scor, 4))
    fig.add_annotation(
        x=summ_scor,  # x-coordinate of the annotation
        y=max_count,  # y-coordinate of the annotation
        text=txt_a,  # text content of the annotation
        showarrow=False,  # hide the arrow
        font=dict(
            family="Arial",  # font family
            size=14,  # font size
            color="black"  # font color
        ),
        bgcolor="#f0f0f0",  # background color (with transparency)
        #bordercolor="black",  # border color
        #borderwidth=1  # border width
    )
        
    txt_a = name1 + "'s score (" + str(round(summ_scor, 4)) + ")"
    
    txt9 = "Distribution of Stars summary scores for " + name1 + " and its measure peer group " 
    txt9 += "(" + str(grp) + ': hospitals w/ scores in ' + str(int(grp + 2)) + ' domains'+ ")"
    
    star_df = tdf_main[tdf_main["star"] == star_scor]
    star_df = star_df[~star_df['summary_score'].isin([np.nan, float("NaN")])]
    
    summ_ls = star_df['summary_score'].tolist()
    summ_star_perc = str(round(stats.percentileofscore(summ_ls, summ_scor), 1))
    
    txt7 = txt_a + " was in the " + summ_star_perc + " percentile of its Stars group ("
    txt7 += str(star_scor) + "). "
    
    return fig, txt7, '', txt9
            
    
    


@app.callback(
     [Output("data_report_plot3", "children"),
      Output("text10", "children"),
      Output("text11", "children"),],
     [Input("hospital-select1b", "value"), 
      Input("close-centered2", "n_clicks"),
      Input("option_hospitals", 'children'),
      Input('set-select2', "value"),
      Input('domain-select1', "value"),
      ],
     [State("hospital-select1", "value"),
      ],
    )
def update_panel4(hospital, n_clicks, option_hospitals, set_select, domain, selected_hospitals):    
    
    if set_select == 'Measures group':
        
        cols = ['Measure', 'Value', 'Delta value', 
                'Group percentile', 'Delta percentile', 
                'Weight', 'Delta weight']
        
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
        
        if hospital is None:
                
            txt3 = "Please select a focal hospital."
            txt4 = ""
            return dashT, txt3, txt4
        
        tdf_main = main_df.copy(deep=True)
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt3 = hospital + " had no data among the CMS Stars release years"
            txt4 = "Try selecting another hospital"
            return dashT, txt3, txt4
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        
        if len(yrs) > 1:
            tdf_main_LY = tdf_main[tdf_main['Release year'] == yrs[-1]]
            grp_LY = tdf_main_LY[tdf_main_LY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
            tdf_main_LY = tdf_main_LY[tdf_main_LY['cnt_grp'].isin([grp_LY])]
            
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yrs[-2]]
            grp_PY = tdf_main_PY[tdf_main_PY['Name and Num'] == hospital]['cnt_grp'].iloc[0]
            tdf_main_PY = tdf_main_PY[tdf_main_PY['cnt_grp'].isin([grp_PY])]
            
            ######## GET RESULTS FOR LATEST YEAR ##############
            # Get hospitals
            hosp_ls_LY = tdf_main_LY['Name and Num'].tolist()
            i = 0
            try:
                i = hosp_ls_LY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            # Get measures
            
            measure_ls = feature_dict[domain + ' (Std)']
            hosp_scors_LY = []
            hosp_percs_LY = []
            hosp_wts_LY = []
            
            # Get values for latest year
            for m in measure_ls:
                ls = tdf_main_LY[m].tolist()
                hosp_scors_LY.append(ls[i])
                perc = round(stats.percentileofscore(ls, ls[i]), 1)
                hosp_percs_LY.append(perc)
            
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
                    
                ls = tdf_main_LY[pref + 'measure_wt'].tolist()
                hosp_wts_LY.append(ls[i])
                
                
            ######## GET RESULTS FOR NEXT LATEST YEAR ##############
            
            # Get hospitals
            hosp_ls_PY = tdf_main_PY['Name and Num'].tolist()
            i = 0
            try:
                i = hosp_ls_PY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            # Get measures
            
            measure_ls = feature_dict[domain + ' (Std)']
            hosp_scors_PY = []
            hosp_percs_PY = []
            hosp_wts_PY = []
            
            # Get values for latest year
            for m in measure_ls:
                ls = tdf_main_PY[m].tolist()
                hosp_scors_PY.append(ls[i])
                perc = round(stats.percentileofscore(ls, ls[i]), 1)
                hosp_percs_PY.append(perc)
                
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
                
                
            #########
            
            # Compute values for columns
            
            delta_value = np.round(np.array(hosp_scors_LY) - np.array(hosp_scors_PY), 4)
            delta_perc = np.round(np.array(hosp_percs_LY) - np.array(hosp_percs_PY), 4)
            delta_wght = np.round(np.array(hosp_wts_LY) - np.array(hosp_wts_PY), 4)
            
            cols = ['Measure', 'Value', 'Delta value', 
                    'Group percentile', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Measure'] = measure_ls
            df_table['Value'] = np.round(hosp_scors_LY, 4)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Group percentile'] = hosp_percs_LY
            df_table['Delta percentile'] = delta_perc
            df_table['Weight'] = hosp_wts_LY
            df_table['Delta weight'] = delta_wght
            
            df_table.dropna(how='all', axis=0, subset=['Value', 'Delta value', 
                                                       'Group percentile', 'Delta percentile', 
                                                       ], inplace=True)
            
            dashT = dash_table.DataTable(
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                export_format="csv",
                page_action='none',
                sort_action="native",
                sort_mode="multi",
                #filter_action="native",
                
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
            
            name1 = str(name) 
            if name == 'RUSH UNIVERSITY MEDICAL CENTER':
                name1 = 'RUMC'
            elif name == 'RUSH OAK PARK HOSPITAL':
                name1 = 'ROPH'
            elif name == 'RUSH COPLEY':
                name1 = 'RUSH COPLEY'
            else:
                name1 = "hospital " + hospital[-7:-1]
            
            txt3 = "The latest year of data used was " + str(int(yrs[-1])) + ". "
            txt3 += "Delta values were computed using the prior year " + str(int(yrs[-2])) + ". "
            txt4 = ''
            
            if np.isnan(grp_LY) == True and np.isnan(grp_PY) == True:
                txt3 += "In both years, " + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_LY) == True:
                txt3 += "In " + str(int(yrs[-1])) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_PY) == True:
                txt3 += "In " + str(int(yrs[-2])) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            
            elif grp_LY == grp_PY:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                txt3 += "In both years, " + name1 + " was in group " + str(int(grp_LY)) + numD_LY + '.'
            else:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                numD_PY = ' (hospitals w/ scores in ' + str(int(grp_PY + 2)) + ' domains)'
                
                txt3 += name1 + " was in group " + str(int(grp_PY)) + numD_PY + " in " + str(int(yrs[-2]))
                txt3 += " and in group " + str(int(grp_LY)) + numD_LY + " in " + str(int(yrs[-1])) + ". "
                
            return dashT, txt3, txt4
        
        else:
            return dashT, '', ''
        
        
    elif set_select == 'Selected hospitals':

        cols = ['Measure', 'Value', 'Change in value from previous year', 
                'Percentile', 'Change in percentile', 
                'Weight', 'Change in weight']
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
        
        if hospital is None or option_hospitals is None or option_hospitals == []:
                
            txt3 = ''
            if hospital is None:
                txt3 = "Please select a focal hospital."
            if option_hospitals is None or option_hospitals == []:
                txt3 += "You either haven't selected any hospitals or the filters you chose left you with no hospitals to analyze."
            
            txt4 = ""
            return dashT, txt3, txt4
        
    
        if hospital in option_hospitals:
            # Remove focal hospital from options, for use below
            option_hospitals.remove(hospital)
            
        if selected_hospitals is None or selected_hospitals == [] or selected_hospitals == [hospital]:
            #There are options but either none were selected specifically by name or only our focal hospital was selected by name
            selected_hospitals = list(option_hospitals)
        
        if hospital in selected_hospitals:
            #Remove focal hospital from options, for use below
            selected_hospitals.remove(hospital)
        
        
        # At this point, we can still get results even selected_hospitals and option_hospitals are empty lists. 
        #    That is, even if we've ended up with nothing to compare our hospital to.
           
        tdf_main = main_df[main_df['Name and Num'].isin(selected_hospitals + [hospital])]
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt3 = hospital + " had no data among the CMS Stars release years"
            txt4 = "Try selecting another hospital"
            return dashT, txt3, txt4
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        
        if len(yrs) > 1:
            tdf_main_LY = tdf_main[tdf_main['Release year'] == yrs[-1]]
            tdf_main_PY = tdf_main[tdf_main['Release year'] == yrs[-2]]
            
            ######## GET RESULTS FOR LATEST YEAR ##############
            # Get hospitals
            hosp_ls_LY = tdf_main_LY['Name and Num'].tolist()
            i = 0
            try:
                i = hosp_ls_LY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            # Get measures
            
            measure_ls = feature_dict[domain + ' (Std)']
            hosp_scors_LY = []
            hosp_percs_LY = []
            hosp_wts_LY = []
            
            # Get values for latest year
            for m in measure_ls:
                ls = tdf_main_LY[m].tolist()
                hosp_scors_LY.append(ls[i])
                perc = round(stats.percentileofscore(ls, ls[i]), 1)
                hosp_percs_LY.append(perc)
                
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
                    
                ls = tdf_main_LY[pref + 'measure_wt'].tolist()
                hosp_wts_LY.append(ls[i])
            
            ######## GET RESULTS FOR NEXT LATEST YEAR ##############
            
            # Get hospitals
            hosp_ls_PY = tdf_main_PY['Name and Num'].tolist()
            i = 0
            try:
                i = hosp_ls_PY.index(hospital)
            except:
                txt3 = hospital + " had no data among the CMS Stars release years"
                txt4 = "Try selecting another hospital"
                return dashT, txt3, txt4
            
            # Get measures
            
            measure_ls = feature_dict[domain + ' (Std)']
            hosp_scors_PY = []
            hosp_percs_PY = []
            hosp_wts_PY = []
            
            # Get values for latest year
            for m in measure_ls:
                ls = tdf_main_PY[m].tolist()
                hosp_scors_PY.append(ls[i])
                perc = round(stats.percentileofscore(ls, ls[i]), 1)
                hosp_percs_PY.append(perc)
                
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
                
            #########
            
            # Compute values for columns
            
            
            delta_value = np.round(np.array(hosp_scors_LY) - np.array(hosp_scors_PY), 4)
            delta_perc = np.round(np.array(hosp_percs_LY) - np.array(hosp_percs_PY), 4)
            delta_wght = np.round(np.array(hosp_wts_LY) - np.array(hosp_wts_PY), 4)
            
            cols = ['Measure', 'Value', 'Delta value', 
                    'Percentile', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Measure'] = measure_ls
            df_table['Value'] = np.round(hosp_scors_LY, 4)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Percentile'] = hosp_percs_LY
            df_table['Delta percentile'] = delta_perc
            df_table['Weight'] = hosp_wts_LY
            df_table['Delta weight'] = delta_wght
            
            df_table.dropna(how='all', axis=0, subset=['Value', 'Delta value', 
                                                       'Percentile', 'Delta percentile', 
                                                       ], inplace=True)
            
            
            dashT = dash_table.DataTable(
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                export_format="csv",
                page_action='none',
                sort_action="native",
                sort_mode="multi",
                #filter_action="native",
                
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
            
            name1 = str(name) 
            if name == 'RUSH UNIVERSITY MEDICAL CENTER':
                name1 = 'RUMC'
            elif name == 'RUSH OAK PARK HOSPITAL':
                name1 = 'ROPH'
            elif name == 'RUSH COPLEY':
                name1 = 'RUSH COPLEY'
            else:
                name1 = "Hospital " + hospital[-7:-1]
            
            txt3 = "The latest year of data used was " + str(int(yrs[-1])) + ". "
            txt3 += "Delta values were computed using the prior year " + str(int(yrs[-2])) + ". "
            txt4 = ''
            
            return dashT, txt3, txt4
        
        else:
            return dashT, '', ''
    
#########################################################################################


# Run the server
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug = True) # modified to run on linux server

