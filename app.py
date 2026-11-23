import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.exceptions import PreventUpdate

import numpy as np
from scipy.stats import percentileofscore

import pandas as pd
#import plotly
import plotly.graph_objects as go
import plotly.express as px
import warnings
#import sys
import re
#import csv
#import math
import random
import json
from datetime import datetime
import textwrap

import what_if_fxn
import features

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


# ---------------------------------------------------------------------------
# Layout geometry for the measure time-series figure.
#
# The plot area (the rectangle inside the axes) is held at a fixed
# TS_PLOT_VW vw tall. Only the pixels reserved ABOVE it for the legend change
# with the number of traces. The legend is laid out as a fixed grid of
# TS_LEGEND_COLS columns. Long labels wrap onto extra lines, and each legend
# row is as tall as its most-wrapped entry.
# ---------------------------------------------------------------------------
TS_PLOT_VW          = 30   # fixed plot-area height, in vw (1 vw = 1% of window width)
TS_LEGEND_COLS      = 3    # legend columns
TS_LEGEND_ROW_PX    = 22   # px per legend row at Plotly's default 12-px font; tune by eye
TS_LEGEND_PAD_PX    = 20   # gap between the legend and the top of the plot area
TS_LEGEND_WRAP_CHARS = 32  # legend text wraps onto a new line after about this many characters
TS_LEGEND_LINE_PX   = 16   # px added to a legend row for each extra wrapped line; tune by eye
TS_MARGIN_TOP_BASE  = 10
TS_MARGIN_BOTTOM    = 60   # x-axis ticks + title; margins no longer auto-expand, so be generous
TS_MARGIN_LEFT      = 80
TS_MARGIN_RIGHT     = 40

TS_YLABELS = {
    'Standardized scores': "Standardized score",
    'Raw scores':          "Raw score",
    'Percentiles':         "Percentile",   # relative to all hospitals in the same release year
}

################################# LOAD DATA ##################################
month = 'Nov'
year = '2026'
main_df = pd.read_pickle('dataframe_data/hosp_stars_dat_from_'+month+'_'+year+'.pkl')
beds_max = np.nanmax(main_df['Beds'])

what_if_df_2026 = pd.read_pickle('dataframe_data/2026_data_for_whatifs.pkl')
what_if_df_2027 = pd.read_pickle('dataframe_data/2027_data_for_whatifs.pkl')

##############################################################################
######################## Create Features Dictionary #####################################





HOSPITALS = main_df['Name and Num'].tolist()

beds = main_df['Beds'].tolist()
states = main_df['State'].tolist()
htypes = main_df['Hospital Type'].tolist()
ctypes = main_df['Hospital Ownership'].tolist()
lons = main_df['Lon'].tolist()
lats = main_df['Lat'].tolist()

main_df['Release year'] = main_df['Release year'].astype(int)

latest_yr = np.max(main_df['Release year'])
prediction_yr = int(latest_yr)
latest_release_yr = latest_yr - 1

current_yr = datetime.now().year
current_mo = datetime.now().month

states = ['NaN' if x is np.nan else x for x in states]
htypes = ['NaN' if x is np.nan else x for x in htypes]
ctypes = ['NaN' if x is np.nan else x for x in ctypes]

HOSPITALS_SET = sorted(list(set(HOSPITALS)))

ddfs = "100%"

domains = ['Patient Experience', 'Readmission', 'Mortality', 
           'Safety of Care', 'Timely and Effective Care']


random.seed(42)
COLORS = []
for h in HOSPITALS:
    if 'RUSH UNIVERSITY' in h:
        clr = '#167e04'
    else:
        clr = '#' + "%06x" % random.randint(0, 0xFFFFFF)
    COLORS.append(clr)
    

##############################################################################

'''
domain = 'Patient Experience'
measures = ['H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 
               'H_COMP_3_STAR_RATING', 'H_COMP_5_STAR_RATING', 
               'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING', 
               'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING']
'''

##############################################################################


################# DASH APP CONTROL FUNCTIONS #################################

def description_card1():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("Hospital Quality Star Ratings", style={
            'textAlign': 'left',
            }),
           dcc.Markdown("The Centers for Medicare & Medicaid Services (CMS) Overall Hospital " +
                        "Quality Star Ratings are summary measures of hospital quality " +
                        "based on safety, mortality, readmissions, patient ratings, " +
                        "and timely and effective care. This app enables deep dives into " +
                        "the past, present, and future ratings of all CMS certified hospitals."),
                        
        ],
    )


def generate_control_card1():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card1",
        children=[
            
            html.H5("Choose a hospital", style={
                'display': 'inline-block',
                #'width': '100%', 
                
                }),
            
            dcc.Dropdown(
                id="hospital-select1b",
                options=[{"label": i, "value": i} for i in ["RUSH UNIVERSITY MEDICAL CENTER (140119)"]],
                value="RUSH UNIVERSITY MEDICAL CENTER (140119)",
                #placeholder='Select a focal hospital',
                optionHeight=75,
                style={
                    'width': '99%', 
                    'font-size': 13,
                    #'display': 'inline-block',
                    'border-radius': '15px',
                    'padding': '0px',
                    #'margin-top': '15px',
                    'margin-left': '1px',
                    'margin-bottom': '20px',
                    }
            ),
            
            html.H5("Set filters"),
            html.P("Your chosen hospital can be compared to a subset. All hospitals are chosen by default."),
            
            dbc.Button("Hospital types",
                        id="open-centered4",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'display': 'inline-block',
                            'margin-left': '10%',
                            #'margin-bottom': '1%',
                            'width': '80%',
                            'font-size': 12,
                            'margin-bottom': '5px',
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
            
            dbc.Button("Hospital ownership",
                        id="open-centered1",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'display': 'inline-block',
                            'margin-left': '10%',
                            'margin-bottom': '5px',
                            'width': '80%',
                            'font-size': 12,
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
            
            dbc.Button("US states & territories",
                        id="open-centered3",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'display': 'inline-block',
                            'margin-left': '10%',
                            #'margin-bottom': '1%',
                            'width': '80%',
                            'font-size': 12,
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
                        2700: '2700',
                    },
                value=[0, 2700],
                ),
            
            html.Br(),
        ],
    )


#########################################################################################
#############################   DASH APP LAYOUT   #######################################
#########################################################################################    


app.layout = html.Div([
    
    html.Div(
            id="option_hospitals", 
            style={'display': 'none'}
        ),
    
    dcc.Store(id="whatif_df"),
        
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
            },
        ),
    
    
    
    # Panel 1
    html.Div(
            id="panel-1",
            className="eight columns",
            children=[
                html.Div(
                    children=[
                        html.H5(id="header-text", 
                                style={'text-align': 'center'},
                                ),
                        ],
                    style={'width': '107%', 
                           'display': 'inline-block',
                           'border-radius': '15px',
                           'box-shadow': '1px 1px 1px grey',
                           'background-color': '#f0f0f0',
                           'padding': '10px',
                           'margin-bottom': '10px',
                           },
                    ),

                html.Div(
                    #className="one columns",
                    children=[
                        html.H5(id="box-header-text", 
                                style={'text-align': 'center'},
                                ),
                        
                        dcc.Dropdown(
                            id='year-select1',
                            options=[{"label": str(i)+'  ', "value": i} for i in list(range(2021, latest_yr+1))],
                            value=latest_yr,
                            placeholder='Select a Stars year',
                            optionHeight=50,
                            style={
                                #'width': '30%', 
                                #'font-size': 13,
                                'display': 'inline-block',
                                #'border-radius': '15px',
                                'padding': '0px 30px 0px 20px',
                                #"background-color": "#2a8cff",
                                'verticalAlign': 'bottom',
                                #'margin-top': '15px',
                                'margin-left': '1%',
                                },
                        ),
                        
                        dbc.Button("Compare to Filtered Hospitals", 
                                    id="selected_hosps_btn1",
                                    n_clicks=0,
                                    className="custom-button",
                                    style={
                                       'display': 'inline-block',
                                       'margin-left': '8%',
                                       "fontSize": "12px",
                                       },
                                   ),
                        
                        dbc.Button("Compare to Stars Peer Group",
                                    id="stars_peers_btn1",
                                    n_clicks=0,
                                    className="custom-button",
                                    style={
                                       'display': 'inline-block',
                                       'margin-left': '4%',
                                       "fontSize": "12px",
                                       },
                                   ),
                        html.Hr(),
                        
                        dcc.Graph(id="figure1"),
                        
                        ],
                    
                    style={'width': '107%',
                           'horizontal-align': 'center',
                           'display': 'inline-block',
                           'margin-bottom': '1%',
                           'border-radius': '15px',
                           'box-shadow': '1px 1px 1px grey',
                           'background-color': '#f0f0f0',
                           'padding': '10px',
                           },
                    ),
                
                
                html.Div(
                    children=[
                        html.Div(
                            id="box1",
                            className="two columns",
                            children=[
                                dbc.Container([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(
                                                id="boxtext1",
                                                style={
                                                    "backgroundColor": "LightSkyBlue",
                                                    "color": "RoyalBlue",
                                                    "textAlign": "center",
                                                    "padding": "10px",
                                                    "border": "2px solid RoyalBlue",
                                                    "borderRadius": "5px",
                                                    "fontSize": "16px"},
                                                ),
                                            ],
                                            width=2,
                                            ),
                                        
                                        dbc.Col([
                                            html.Div(
                                                id="boxtext2",
                                                style={
                                                    "backgroundColor": "LightSkyBlue",
                                                    "color": "RoyalBlue",
                                                    "textAlign": "center",
                                                    "padding": "10px",
                                                    "border": "2px solid RoyalBlue",
                                                    "borderRadius": "5px",
                                                    "fontSize": "16px"},
                                                ),
                                            ],
                                            width=4,
                                            ),
                                        
                                        dbc.Col([
                                            html.Div(
                                                id="boxtext3",
                                                style={
                                                    "backgroundColor": "LightSkyBlue",
                                                    "color": "RoyalBlue",
                                                    "textAlign": "center",
                                                    "padding": "10px",
                                                    "border": "2px solid RoyalBlue",
                                                    "borderRadius": "5px",
                                                    "fontSize": "16px"},
                                                ),
                                            ], 
                                            width=6,
                                            ),
                                        ],
                                        ),
                                    
                                    
                                    html.Hr(),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(
                                                id="boxtext4",
                                                style={
                                                    "backgroundColor": "LightSkyBlue",
                                                    "color": "RoyalBlue",
                                                    "textAlign": "center",
                                                    "padding": "10px",
                                                    "border": "2px solid RoyalBlue",
                                                    "borderRadius": "5px",
                                                    "fontSize": "16px"},
                                                ),
                                            ], 
                                            width=6,
                                            ),
                                        
                                        dbc.Col([
                                            html.Div(
                                                id="boxtext5",
                                                style={
                                                    "backgroundColor": "LightSkyBlue",
                                                    "color": "RoyalBlue",
                                                    "textAlign": "center",
                                                    "padding": "10px",
                                                    "border": "2px solid RoyalBlue",
                                                    "borderRadius": "5px",
                                                    "fontSize": "16px"},
                                                ),
                                            ], 
                                            width=6,
                                            ),
                                        
                                        ],
                                        ),
                                    ],
                                    fluid=True,
                                    ),
                                
                                ],
                                style={
                                    'width': '105.5%',
                                    #'display': 'inline-block',
                                    'horizontal-align': 'center',
                                    #'border-radius': '15px',
                                    #'box-shadow': '1px 1px 1px grey',
                                    #'background-color': '#f0f0f0',
                                    #'padding': '10px',
                                    #'margin-bottom': '10px',
                                    },
                                ),
                        
                        ],
                    ),
                
                ],
            ),
    
    html.Div(
        children=[
            
            dbc.Button("Examine measure domains",
                        id="open-that",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'font-size': 14,
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '2%',
                            'margin-top': '0.5%',
                            'width': '30%',
                            #'white-space': 'normal',  # Allow text to wrap
                            #'overflow': 'hidden',  # Optional: Hide any overflow
                            #'word-wrap': 'break-word',  # Optional: Break long words
                            },
                       ),
            
            dbc.Modal(
                [dbc.ModalBody([
                    html.Div(
                        children=[
                            html.H5("Scores across domains"),
                            dcc.Dropdown(
                                id='year-select2',
                                options=[{"label": i, "value": i} for i in list(range(2021, latest_yr+1))],
                                value=latest_yr,
                                placeholder='Select a Stars year',
                                optionHeight=50,
                                style={
                                    'display': 'inline-block',
                                    'padding': '0px 30px 0px 20px',
                                    'verticalAlign': 'bottom',
                                    'margin-left': '0%',
                                    },
                                ),
                            
                            dbc.Button("Compare to Filtered Hospitals",
                                        id="selected_hosps_btn2",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'font-size': 12,
                                            'display': 'inline-block',
                                            'vertical-align': 'top',
                                            'margin-left': '3%',
                                            },
                                       ),
                            
                            
                            dbc.Button("Compare to Stars Peer Group",
                                        id="stars_peers_btn2",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'font-size': 12,
                                            'display': 'inline-block',
                                            'vertical-align': 'top',
                                            'margin-left': '3%',
                                            },
                                       ),
                            
                            html.Hr(),
                            html.Br(),
                            html.Div(id="data_report_plot2"),
                            html.Hr(),
                            html.B(id="text3",
                                   style={'fontSize':16, 
                                          'margin-left':'1%',
                                          },
                                   ),
                            
                            html.Hr(),
                            dcc.Dropdown(
                                id="score-select2",
                                options=[{"label": i, "value": i} for i in ['Percentiles', 'Domain scores']],
                                value='Percentiles',
                                placeholder='Select a score type',
                                optionHeight=50,
                                style={
                                    'width': '250px', 
                                    'font-size': 16,
                                    'display': 'inline-block',
                                    'padding': '0px 30px 0px 20px',
                                    'margin-left': '2%',
                                    'margin-top': '1%',
                                    'margin-bottom': '1%',
                                    }
                                ),
                            
                            
                            html.Div(id="data_report_plot6"),
                            
                            ],
                        style={
                            'width': '100%',
                            #'display': 'inline-block',
                            #'border-radius': '15px',
                            #'box-shadow': '1px 1px 1px grey',
                            'background-color': '#f0f0f0',
                            #'padding': '10px',
                            #'margin-bottom': '10px',
                            },
                        ),
                    html.Br(), 
                    ],
                    style={
                        'background-color': '#f0f0f0',
                        },
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", 
                                   id="close-that",
                                   className="ml-auto",
                                   style={"background-color": "#2a8cff",
                                          'width': '30%',
                                          'font-size': 14,
                                          },
                                   ),
                        style={"background-color": "#696969",
                               "display": "flex",
                               "justify-content": "center",
                               "align-items": "center",
                               },
                        ),
                    ],
                id="modal-choose_that",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            
            dbc.Button("Examine individual measures",
                        id="open-theother",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'font-size': 14,
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '3%',
                            'margin-top': '0.5%',
                            'width': '30%',
                            #'white-space': 'normal',  # Allow text to wrap
                            #'overflow': 'hidden',  # Optional: Hide any overflow
                            #'word-wrap': 'break-word',  # Optional: Break long words
                            },
                       ),
            dbc.Modal(
                [dbc.ModalBody([
                    html.Div(
                        children=[
                            html.H5("Examine individual measures"),
                            
                            dbc.Button("Compare to Filtered Hospitals",
                                        id="selected_hosps_btn3",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'display': 'inline-block',
                                            },
                                       ),
                            
                            dbc.Button("Compare to Stars Peer Group",
                                       id="stars_peers_btn3",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'display': 'inline-block',
                                            'margin-left': '2%',
                                            },
                                       ),
                            
                            html.Br(),
                            dcc.Dropdown(
                                id='year-select3',
                                options=[{"label": i, "value": i} for i in list(range(2021, latest_yr+1))],
                                value=latest_yr,
                                placeholder='Select a Stars year',
                                optionHeight=50,
                                style={
                                    'display': 'inline-block',
                                    #'padding': '0px 30px 0px 20px',
                                    #'verticalAlign': 'top',
                                    'width': '7vw', 
                                    'margin-left': '0%',
                                    },
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
                                    'padding': '0px 30px 0px 20px',
                                    'margin-top': '1%',
                                    }
                                ),
                            
                            dcc.Dropdown(
                                id='score-type1',
                                options=[{"label": i, "value": i} for i in ['Standardized scores', 'Raw scores']],
                                value='Standardized scores',
                                placeholder='Select a score type',
                                optionHeight=50,
                                style={
                                    'width': '250px', 
                                    'font-size': 16,
                                    'display': 'inline-block',
                                    'padding': '0px 30px 0px 20px',
                                    'margin-left': '2%',
                                    'margin-top': '1%',
                                    'margin-bottom': '1%',
                                    }
                                ),
                            #html.Br(),
                            html.Div(id="data_report_plot3"),
                            html.Hr(),
                            html.B(id="text10", style={'fontSize':16}),
                            html.P(id="text11", style={'fontSize':16}),
                            html.Hr(),
                            dcc.Dropdown(
                                id="score-select1",
                                options=[{"label": i, "value": i} for i in ['Percentiles', 'Standardized scores', 'Raw scores']],
                                value='Percentiles',
                                placeholder='Select a score type',
                                optionHeight=50,
                                style={
                                    'width': '250px', 
                                    'font-size': 16,
                                    'display': 'inline-block',
                                    'padding': '0px 30px 0px 20px',
                                    'margin-left': '2%',
                                    'margin-top': '1%',
                                    'margin-bottom': '1%',
                                    }
                                ),
                            
                            
                            html.Div(id="data_report_plot5",
                                             ),
                            
                            ],
                        style={
                            'width': '100%',
                            'background-color': '#f0f0f0',
                            },
                        ),
                    html.Br(),
                    ],
                    style={
                        'background-color': '#f0f0f0',
                        },
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close",
                                   id="close-theother",
                                   className="ml-auto",
                                   style={"background-color": "#2a8cff",
                                          'width': '30%',
                                          'font-size': 14,
                                          },
                                   ),
                        style={"background-color": "#696969",
                               "display": "flex",
                               "justify-content": "center",
                               "align-items": "center",
                               },
                        ),
                    ],
                id="modal-choose_theother",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            
            dbc.Button("Run what-if scenarios",
                       id="open-what-if",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'font-size': 14,
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '3%',
                            'margin-top': '0.5%',
                            'width': '30%',
                            #'white-space': 'normal',  # Allow text to wrap
                            #'overflow': 'hidden',  # Optional: Hide any overflow
                            #'word-wrap': 'break-word',  # Optional: Break long words
                            },
                       ),
            dbc.Modal(
                [dbc.ModalBody([
                    html.Div(
                        children=[
                            html.H5(id="what-if-header"),
                            
                            dcc.Markdown("**Instructions:**\n* The table below provides the actual " +
                                         "value for your chosen hospital, the min " +
                                         "and max among hospitals, and whether a higher " +
                                         "score means a better score.\n* Simply modify the " +
                                         "values in one or more cells of the 'What-if value' " +
                                         "column, then click RUN WHAT-IF." +
                                         "\n* Avoid invalid results by staying within the min " +
                                         "and the max."),
                            
                            html.Hr(),
                            html.B(id="text-what-if", style={'fontSize':16}),
                            html.Br(),
                            
                            dcc.Dropdown(
                                id="year-select4",
                                options=[{"label": str(i)+'  ', "value": i} for i in list(range(latest_release_yr, latest_yr+1))],
                                value=latest_yr,
                                #placeholder='Select a year',
                                optionHeight=50,
                                style={
                                    #'width': '20%', 
                                    #'font-size': 13,
                                    'display': 'inline-block',
                                    #'border-radius': '15px',
                                    'width': '7vw',
                                    #"background-color": "#2a8cff",
                                    'verticalAlign': 'top',
                                    #'margin-top': '15px',
                                    #'margin-left': '3%',
                                    },
                            ),
                            
                            dbc.Button("Run What-If",
                                        id="whatif_button",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'display': 'inline-block',
                                            'margin-left': '2%',
                                            'margin-bottom': '1%',
                                            'width': '20%',
                                            'font-size': 12,
                                            },
                                       ),
                            
                            dbc.Button("View what-if results",
                                       id="open-what-if-results",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'width': '20%',
                                            'font-size': 12,
                                            #'display': 'inline-block',
                                            'margin-left': '2%',
                                            'margin-bottom': '1%',
                                            #'verticalAlign': 'top',
                                            },
                                       ),
                            
                            dbc.Button("Reset the table",
                                        id="reset-table",
                                        n_clicks=0,
                                        className="custom-button",
                                        style={
                                            'width': '20%',
                                            'font-size': 12,
                                            #'display': 'inline-block',
                                            'margin-left': '2%',
                                            'margin-bottom': '1%',
                                            #'verticalAlign': 'top',
                                            },
                                       ),
                            
                            dbc.Modal(
                                [dbc.ModalBody([
                                    
                                    html.H5("What-if Results", 
                                            style={'text-align': 'center',
                                                   'color': '#ffffff'},
                                            ),
                                    
                                    #modal boxes
                                    dcc.Loading(
                                        id="loading-spinner",
                                        type="circle",  # "circle", "dot", or "default"
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        id="modal-box1",
                                                        className="two columns",
                                                        children=[
                                                            dbc.Container([
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Div(
                                                                            id="modal-boxtext1",
                                                                            style={
                                                                                "backgroundColor": "LightSkyBlue",
                                                                                "color": "RoyalBlue",
                                                                                "textAlign": "center",
                                                                                "padding": "10px",
                                                                                "border": "2px solid RoyalBlue",
                                                                                "borderRadius": "5px",
                                                                                "fontSize": "16px"},
                                                                            ),
                                                                        ],
                                                                        width=2,
                                                                        ),
                                                                    
                                                                    dbc.Col([
                                                                        html.Div(
                                                                            id="modal-boxtext2",
                                                                            style={
                                                                                "backgroundColor": "LightSkyBlue",
                                                                                "color": "RoyalBlue",
                                                                                "textAlign": "center",
                                                                                "padding": "10px",
                                                                                "border": "2px solid RoyalBlue",
                                                                                "borderRadius": "5px",
                                                                                "fontSize": "16px"},
                                                                            ),
                                                                        ],
                                                                        width=4,
                                                                        ),
                                                                    
                                                                    dbc.Col([
                                                                        html.Div(
                                                                            id="modal-boxtext3",
                                                                            style={
                                                                                "backgroundColor": "LightSkyBlue",
                                                                                "color": "RoyalBlue",
                                                                                "textAlign": "center",
                                                                                "padding": "10px",
                                                                                "border": "2px solid RoyalBlue",
                                                                                "borderRadius": "5px",
                                                                                "fontSize": "16px"},
                                                                            ),
                                                                        ], 
                                                                        width=6,
                                                                        ),
                                                                    ],
                                                                    ),
                                                                
                                                                
                                                                html.Br(),
                                                                
                                                                dbc.Row([
                                                                    dbc.Col([
                                                                        html.Div(
                                                                            id="modal-boxtext4",
                                                                            style={
                                                                                "backgroundColor": "LightSkyBlue",
                                                                                "color": "RoyalBlue",
                                                                                "textAlign": "center",
                                                                                "padding": "10px",
                                                                                "border": "2px solid RoyalBlue",
                                                                                "borderRadius": "5px",
                                                                                "fontSize": "16px"},
                                                                            ),
                                                                        ], 
                                                                        width=6,
                                                                        ),
                                                                    
                                                                    dbc.Col([
                                                                        html.Div(
                                                                            id="modal-boxtext5",
                                                                            style={
                                                                                "backgroundColor": "LightSkyBlue",
                                                                                "color": "RoyalBlue",
                                                                                "textAlign": "center",
                                                                                "padding": "10px",
                                                                                "border": "2px solid RoyalBlue",
                                                                                "borderRadius": "5px",
                                                                                "fontSize": "16px"},
                                                                            ),
                                                                        ], 
                                                                        width=6,
                                                                        ),
                                                                    
                                                                    ],
                                                                    ),
                                                                ],
                                                                fluid=True,
                                                                ),
                                                            
                                                            ],
                                                            style={
                                                                'width': '98%',
                                                                'horizontal-align': 'center',
                                                                'background-color': '#696969',
                                                                },
                                                            ),
                                                    
                                                    ],
                                                ),
                                            ],
                                        ),
                                    
                                    html.Br(),
                                    ],
                                    style={
                                        'background-color': '#696969',
                                        },
                                    ),
                                dbc.ModalFooter(
                                        dbc.Button("Close", 
                                                   id="close-what-if-results", 
                                                   className="ml-auto",
                                                   style={"background-color": "#2a8cff",
                                                          'width': '30%',
                                                          'font-size': 14,
                                                          },
                                                   ),
                                        style={"background-color": "#696969",
                                            "display": "flex",
                                            "justify-content": "center",  # Center horizontally
                                            "align-items": "center",  # Center vertically)
                                            },
                                        ),
                                ],
                                id="modal-what-if-results",
                                is_open=False,
                                centered=True,
                                autoFocus=True,
                                size="xl",
                                keyboard=True,
                                fade=True,
                                backdrop=True,
                                ),
                            
                            
                            dash_table.DataTable(
                                id="data_report_plot4",
                                page_action='none',
                                sort_action="native",
                                sort_mode="multi",
                                style_table={
                                    'overflowY': 'auto',   # vertical scroll
                                    'overflowX': 'auto',   # horizontal scroll
                                    'minWidth': '100%'     # ensures table can expand horizontally
                                },
                                style_cell={
                                    'padding': '10px',
                                    'whiteSpace': 'normal',
                                    'textAlign': 'center',
                                    'minWidth': '100px',  # can tweak per column
                                    'width': '200px',
                                    'maxWidth': '400px',
                                },
                            ),

                            
                            ]   ,
                        style={
                            'width': '100%',
                            'background-color': '#f0f0f0',
                            },
                        ),
                    html.Br(),
                    ],
                    style={
                        'background-color': '#f0f0f0',
                        },
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close",
                                   id="close-what-if",
                                   className="ml-auto",
                                   style={"background-color": "#2a8cff",
                                          'width': '30%',
                                          'font-size': 14,
                                          },
                                   ),
                        style={"background-color": "#696969",
                               "display": "flex",
                               "justify-content": "center",
                               "align-items": "center",
                               },
                        ),
                    ],
                id="modal-choose_what-if",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            
            dbc.Button("Download data", 
                        id="download-btn",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'font-size': 14,
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '2%',
                            'margin-top': '1%',
                            'width': '30%',
                            #'white-space': 'normal',  # Allow text to wrap
                            #'overflow': 'hidden',  # Optional: Hide any overflow
                            #'word-wrap': 'break-word',  # Optional: Break long words
                            },
                       ),
            dcc.Download(id="data-download",
                         ),
            
            
            dbc.Button("View map of filtered hospitals",
                       id="open-this",
                        n_clicks=0,
                        className="custom-button",
                        style={
                            'font-size': 14,
                            'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-left': '3%',
                            'margin-top': '1%',
                            'width': '30%',
                            #'white-space': 'normal',  # Allow text to wrap
                            #'overflow': 'hidden',  # Optional: Hide any overflow
                            #'word-wrap': 'break-word',  # Optional: Break long words
                            },
                       ),
            dbc.Modal(
                [dbc.ModalBody([
                    
                    html.Div(
                        id="map1",
                        children=[
                            html.B(id="map-header"),
                            html.Hr(),
                            
                            dcc.Loading(
                                id="loading-map1",
                                type="default",
                                fullscreen=False,
                                children=[dcc.Graph(id="map_plot1"),],),
                        ],
                        style={'width': '100%',
                                     #'border-radius': '15px',
                                     #'box-shadow': '1px 1px 1px grey',
                                     #'background-color': '#f0f0f0',
                                     #'padding': '10px',
                                     #'margin-bottom': '10px',
                                     "align-items": "center",
                                },
                    ),
                    
                    html.Br(),
                    ],
                    ),
                dbc.ModalFooter(
                        dbc.Button("Close", 
                                   id="close-this", 
                                   className="ml-auto",
                                   style={"background-color": "#2a8cff",
                                          'width': '30%',
                                          'font-size': 14,
                                          },
                                   ),
                        style={ "background-color": "#696969",
                            "display": "flex",
                            "justify-content": "center",  # Center horizontally
                            "align-items": "center",  # Center vertically)
                            },
                        ),
                ],
                id="modal-choose_this",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
            
            ],
        style={'width': '96%', 
               'display': 'inline-block',
               'border-radius': '15px',
               'box-shadow': '1px 1px 1px grey',
               'background-color': '#f0f0f0',
               'padding': '10px',
               'margin-left': '2%',
               'margin-bottom': '100px',
               'margin-top': '0.5%',
               'height': '100%',
               },
        ),
    ],
)

  
##############################   Callbacks   ############################################
#########################################################################################


@app.callback(
    Output("modal-choose_this", "is_open"),
    [Input("open-this", "n_clicks"), 
     Input("close-this", "n_clicks")],
    [State("modal-choose_this", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-choose_that", "is_open"),
    [Input("open-that", "n_clicks"), 
     Input("close-that", "n_clicks")],
    [State("modal-choose_that", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-choose_theother", "is_open"),
    [Input("open-theother", "n_clicks"), 
     Input("close-theother", "n_clicks")],
    [State("modal-choose_theother", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-choose_what-if", "is_open"),
    [Input("open-what-if", "n_clicks"), 
     Input("close-what-if", "n_clicks")],
    [State("modal-choose_what-if", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_4(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered1", "is_open"),
    [Input("open-centered1", "n_clicks"), Input("close-centered1", "n_clicks")],
    [State("modal-centered1", "is_open")],
)
def toggle_modal_5(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-what-if-results", "is_open"),
    [Input("open-what-if-results", "n_clicks"), Input("close-what-if-results", "n_clicks")],
    [State("modal-what-if-results", "is_open")],
)
def toggle_modal_6(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

'''
@app.callback(
    Output("modal-centered2", "is_open"),
    [Input("open-centered2", "n_clicks"), Input("close-centered2", "n_clicks")],
    [State("modal-centered2", "is_open")],
)
def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
'''

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




@app.callback([Output("header-text", 'children'),
               Output("what-if-header", 'children'),
               ],
              [Input("hospital-select1b", 'value'),
               ],
            )
def update_header_text(hospital):
    
    if hospital is None:
        raise PreventUpdate
    else:
        t1 = "Overall Star Rating Results for " + hospital
        t2 = "Explore improvements in the overall star rating for " + hospital
        return t1, t2
        
  
    
@app.callback(Output("map-header", 'children'),
              [Input("option_hospitals", 'children'),
              ],
            )
def update_map_header(filtered_hospitals):
    
    if filtered_hospitals is None:
        raise PreventUpdate
    
    else:
        return "Your filters produced " + str(len(filtered_hospitals)) + " hospitals. Not all hospitals will have data or stars results for each year."

        

@app.callback([Output('year-select1', 'options'),
               Output('year-select1', 'value'),
               Output('year-select2', 'options'),
               Output('year-select2', 'value'),
               Output('year-select3', 'options'),
               Output('year-select3', 'value'),
               ],
              [Input("hospital-select1b", 'value'),
               ],
            )
def update_yrs(hospital):
    
    if hospital is None:
        raise PreventUpdate
        
    else:
        yrs = main_df[main_df['Name and Num'] == hospital]['Release year'].unique()
        yrs = sorted(yrs, reverse=True)
        
        options = [{"label": str(i)+'  ', "value": i} for i in yrs]
        return options, yrs[0], options, yrs[0], options, yrs[0]


@app.callback( # Updated number of beds text
    Output('Filterbeds1', 'children'),
    [
     Input('beds1', 'value'),
     ],
    )
def update_output1(value):
    
    v1 = int(value[0])
    v2 = int(value[1])
    if v2 > 2700:
        v2 = 2700
    value = [v1, v2]
    
    return 'Number of beds: {}'.format(value)

    
@app.callback(
    [Output("hospital-select1b", 'options'),
     Output("option_hospitals", 'children'),
     ],
    [Input('beds1', 'value'), 
     Input("close-centered1", "n_clicks"),
     Input("close-centered3", "n_clicks"),
     Input("close-centered4", "n_clicks"),
     ],
    [
     State('states-select1', 'value'),
     State('hospital_type1', 'value'),
     State('control_type1', 'value'),
     ],
    )
def update_hospitals(bed_range, n_clicks1, n_clicks3, n_clicks4, states_vals, htype_vals, ctype_vals):
    
    low, high = bed_range
    
    if high == 2700:
        high = beds_max
    
    hospitals = []
    for i, h in enumerate(HOSPITALS):
        
        b = beds[i]
        s = states[i]
        ht = htypes[i]
        ct = ctypes[i]
        if (b >= low and b <= high) or np.isnan(b):
            if s in states_vals and ht in htype_vals and ct in ctype_vals:
                    hospitals.append(h)
                    
    hospitals = sorted(list(set(hospitals)))
    hospitals_full = sorted(list(set(HOSPITALS)))
    out_ls2 = [{"label": i, "value": i} for i in hospitals_full]
    
    return out_ls2, hospitals



@app.callback(
    Output("data-download", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_download(n_clicks):
    return dcc.send_data_frame(main_df.to_csv, "data.csv")
    

@app.callback(
     Output("map_plot1", "figure"),
     [Input('beds1', 'value'), 
      Input("close-centered1", "n_clicks"),
      Input("close-centered3", "n_clicks"),
      Input("close-centered4", "n_clicks"),
      Input("option_hospitals", 'children'),
      ],
    )
def update_map_plot1(beds, n_clicks1, n_clicks3, n_clicks4, filtered_hospitals):
    
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
    
    if filtered_hospitals is None or filtered_hospitals == []:
        return figure
    
    tdf = main_df[main_df['Name and Num'].isin(filtered_hospitals)]
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
     [Output("boxtext1", "children"),
      Output("boxtext2", "children"),
      Output("boxtext3", "children"),
      Output("boxtext4", "children"),
      Output("boxtext5", "children"),
      Output("box-header-text", "children"),
      ],
     [Input("hospital-select1b", "value"), 
      Input("option_hospitals", 'children'),
      Input('year-select1', 'value'),
      ],
    )
def update_boxes(hospital, filtered_hospitals, year):    
    
    if hospital is None or filtered_hospitals is None or filtered_hospitals == []:
        return '', '', '', '', '', ''
    
    if hospital in filtered_hospitals:
        '''Remove focal hospital from options, for use below'''
        filtered_hospitals.remove(hospital)
        
    name = hospital[:-9]
    txt = str(name) + " "
    
    name1 = str(name)
    if name == 'RUSH UNIVERSITY MEDICAL CENTER':
        name1 = 'RUMC'
    elif name == 'RUSH OAK PARK HOSPITAL':
        name1 = 'ROPH'
    elif name == 'RUSH COPLEY':
        name1 = 'RUSH COPLEY'
    else:
        name1 = "hospital " + hospital[-7:-1]
        
    
    filtered_hosps_df = main_df[main_df['Name and Num'].isin(filtered_hospitals + [hospital])]
    if filtered_hosps_df['Name and Num'].unique().tolist() == [hospital]:
        ''' indicates that the focal hospital is the only hospital in the data.
            Result of faulty filtering on the user side '''
        
        txt += " is the only hospital in the data."
        return txt, '', '', '', '', '',
    
    hosp_df = filtered_hosps_df[filtered_hosps_df['Name and Num'] == hospital]
    hosp_df = hosp_df[hosp_df['star'].isin([1, 2, 3, 4, 5])]
    if hosp_df.shape[0] == 0:
        ''' indicates no star rating for the focal hospital in the data '''
        
        txt = name1 + " did not receive a star rating in any year of the CMS Overall Hospital Quality Star Ratings included in this application (2020 to " + str(latest_yr) + "). "
        return txt, '', '', '', '', ''
    
    
    txt1 = ''
    txt2 = ''
    txt3 = ''
    txt4 = ''
    txt5 = ''
    txt6 = ''
    
    hosp_yrs = hosp_df['Release year'].unique().tolist()
    
    if latest_release_yr in hosp_yrs:
        
        tdf = hosp_df[hosp_df['Release year'] == year]
        star = tdf['star'].iloc[0]
        txt1 = str(star) + ' Star'
        
        score = tdf['summary_score'].iloc[0]
        score = round(score, 4)
        txt2 = str(score) + ' Summary Score'
        
        grp = tdf['cnt_grp'].iloc[0]
        grp = int(grp)
        
        if grp == 3:
            txt3 = 'Peer Group 3: At least 3 measures in each of 3 domains'
        elif grp == 2:
            txt3 = 'Peer Group 2: At least 3 measures in each of 2 domains'
        elif grp == 1:
            txt3 = 'Peer Group 1: At least 3 measures in 1 domain'
            
        tdf_peer = main_df[(main_df['cnt_grp'] == grp) & (main_df['Release year'] == year)]
        tdf_peer = tdf_peer[tdf_peer['star'] == star]
        perc_of_peer = round(percentileofscore(tdf_peer['summary_score'], score, nan_policy='omit'), 1)
        txt4 = str(perc_of_peer) + " percentile of " + str(star) + " Star group " + str(grp) + " hospitals"
        
        tdf_filtered = filtered_hosps_df[filtered_hosps_df['Release year'] == year]
        tdf_filtered = tdf_filtered[~tdf_filtered['summary_score'].isin([np.nan, float("NaN")])]
        #selected_hospitals = tdf_filtered.shape[0] - 1
        perc_of_chosen = round(percentileofscore(tdf_filtered['summary_score'], score, nan_policy='omit'), 1)
        txt5 = str(perc_of_chosen) + " percentile of hospitals in your filters"
        
        #numD = ' hospitals w/ scores in ' + str(int(grp)) + ' domains'
        
        hosp_ls = tdf_peer['Name and Num'].tolist()
        mort_ls = tdf_peer['Std_Outcomes_Mortality_score'].tolist()
        safe_ls = tdf_peer['Std_Outcomes_Safety_score'].tolist()
        read_ls = tdf_peer['Std_Outcomes_Readmission_score'].tolist()
        pexp_ls = tdf_peer['Std_PatientExp_score'].tolist()
        proc_ls = tdf_peer['Std_Process_score'].tolist()
        
        i = hosp_ls.index(hospital)
        
        mort_scor = mort_ls[i]
        safe_scor = safe_ls[i]
        read_scor = read_ls[i]
        pexp_scor = pexp_ls[i]
        proc_scor = proc_ls[i]
        
        mort_perc = round(percentileofscore(mort_ls, mort_scor, nan_policy='omit'), 1)
        safe_perc = round(percentileofscore(safe_ls, safe_scor, nan_policy='omit'), 1)
        read_perc = round(percentileofscore(read_ls, read_scor, nan_policy='omit'), 1)
        pexp_perc = round(percentileofscore(pexp_ls, pexp_scor, nan_policy='omit'), 1)
        proc_perc = round(percentileofscore(proc_ls, proc_scor, nan_policy='omit'), 1)
        
        #domains = ['Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
        domain_cols = ['Std_Outcomes_Mortality_score', 'Std_Outcomes_Safety_score', 'Std_Outcomes_Readmission_score', 
                       'Std_PatientExp_score', 'Std_Process_score']
        
        perc_ls = [mort_perc, safe_perc, read_perc, pexp_perc, proc_perc]
        scor_ls = [mort_scor, safe_scor, read_scor, pexp_scor, proc_scor]
        
        i = perc_ls.index(max(perc_ls))
        #best_domain = domains[i]
        best_domain_score = scor_ls[i]
        best_domain_perc1 = perc_ls[i]
        
        
        tdf = tdf_filtered[tdf_filtered['Release year'] == year]
        scores_ls = tdf[domain_cols[i]].tolist()
        best_domain_perc2 = percentileofscore(scores_ls, best_domain_score, nan_policy='omit')
        
        i = perc_ls.index(min(perc_ls))
        #worst_domain = domains[i]
        worst_domain_score = scor_ls[i]
        worst_domain_perc1 = perc_ls[i]
        
        scores_ls = tdf[domain_cols[i]].tolist()
        worst_domain_perc2 = percentileofscore(scores_ls, worst_domain_score, nan_policy='omit')
        
        best_domain_score = str(round(best_domain_score, 3))
        best_domain_perc1 = str(round(best_domain_perc1, 1))
        best_domain_perc2 = str(round(best_domain_perc2, 1))
        
        worst_domain_score = str(round(worst_domain_score, 3))
        worst_domain_perc1 = str(round(worst_domain_perc1, 1))
        worst_domain_perc2 = str(round(worst_domain_perc2, 1))
        
    else:
        txt1 = name1 + " did not receive a star rating in " + str(year)
    
    if year == 2027:
        txt6 = 'Predictions for 2027: Distribution of Stars summary scores'
    else:
        txt6 = 'Results for ' + str(year) + ": Distribution of Stars summary scores"
        
    return txt1, txt2, txt3, txt4, txt5, txt6
    
    

@app.callback(
     Output("figure1", "figure"),
     [Input("hospital-select1b", "value"),
      Input('year-select1', 'value'),
      Input("selected_hosps_btn1", "n_clicks"),
      Input("stars_peers_btn1", "n_clicks"),
      Input("option_hospitals", 'children'),
      ],
    )
def update_figure1(hospital, yr, selected_hosps_btn, stars_peers_btn, filtered_hospitals):    
    
    
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
    
    if hospital is None or filtered_hospitals is None or filtered_hospitals == []:
        return fig
    
    if hospital in filtered_hospitals:
        '''Remove focal hospital from options, for use below'''
        filtered_hospitals.remove(hospital)
        
    tdf_main = main_df[main_df['Release year'] == yr]
    hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
    if hosp_df.shape[0] == 0:
        return fig
    
    grp = int(hosp_df['cnt_grp'].iloc[0])
    if np.isnan(grp) == True:
        return fig
    
    name = hospital[:-9]
    name1 = str(name) 
    if name == 'RUSH UNIVERSITY MEDICAL CENTER':
        name1 = 'RUMC'
    elif name == 'RUSH OAK PARK HOSPITAL':
        name1 = 'ROPH'
    else:
        name1 = "Hospital " + hospital[-7:-1]
        
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:150]
    
    #if 'all_hosps_btn1' in jd1:
    #    tdf_main = tdf_main[~tdf_main['summary_score'].isin([np.nan, float("NaN")])]
    if 'selected_hosps_btn1' in jd1:
        tdf_main = tdf_main[tdf_main['Name and Num'].isin(filtered_hospitals+[hospital])]
        tdf_main = tdf_main[~tdf_main['summary_score'].isin([np.nan, float("NaN")])]
    elif 'stars_peers_btn1' in jd1:
        tdf_main = tdf_main[tdf_main['cnt_grp'].isin([grp])]
    else:
        tdf_main = tdf_main[tdf_main['cnt_grp'].isin([grp])]
        
    
    # Get values for latest year
    summ_ls = tdf_main['summary_score'].tolist()
    hosp_ls = tdf_main['Name and Num'].tolist()
    
          	
    i = hosp_ls.index(hospital)
    summ_scor = summ_ls[i]
    
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
        xaxis_title="Stars summary score",
        yaxis_title="No. of hospitals",
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        height=410,
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
        
    
    return fig



@app.callback(
     [Output("data_report_plot2", "children"),
      Output("text3", "children"),
      ],
     [Input("hospital-select1b", "value"), 
      Input("option_hospitals", 'children'),
      Input("selected_hosps_btn2", "n_clicks"),
      Input("stars_peers_btn2", "n_clicks"),
      Input("year-select2", "value"),
      ],
    )
def update_domains_table(hospital, option_hospitals, selected_hosps_btn, stars_peers_btn, yr):    
    
    set_select = 'Measures group'
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:150]
    
    if 'selected_hosps_btn2' in jd1:
        set_select = 'Selected hospitals'
    elif 'stars_peers_btn2' in jd1:
        set_select = 'Measures group'
    
    if set_select == 'Measures group':
        
        cols = ['Domain', 'Value', 'Change in value from previous year', 
                'Percentile of Peer Group', 'Change in percentile', 
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
                
            txt = "Please select a focal hospital."
            return dashT, txt
        
        tdf_main = main_df.copy(deep=True)
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
            return dashT, txt
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        yr1 = int(yr)
        yr2 = None
        
        if yr1 == 2027:
            yr2 = 2026
            if yr2 in yrs:
                pass
            else:
                yr2 = 2025
                
        if yr1 == 2026:
            yr2 = 2025
            if yr2 in yrs:
                pass
            else:
                yr2 = 2024
                
        if yr1 == 2025:
            yr2 = 2024
            if yr2 in yrs:
                pass
            else:
                yr2 = 2023
                
        if yr1 == 2024:
            yr2 = 2023
            if yr2 in yrs:
                pass
            else:
                yr2 = 2022
                
        if yr1 == 2023:
            yr2 = 2022
            if yr2 in yrs:
                pass
            else:
                yr2 = 2021
                
        elif yr1 == 2022:
            yr2 = 2021
        
        elif yr1 == 2021:
            yr2 = 2021
            
        if yr1 in yrs and yr2 in yrs:
            
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
                txt3 = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
                return dashT, txt3
            
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
            
            summ_perc_LY = round(percentileofscore(summ_ls_LY, summ_scor_LY, nan_policy='omit'), 1)
            mort_perc_LY = round(percentileofscore(mort_ls_LY, mort_scor_LY, nan_policy='omit'), 1)
            safe_perc_LY = round(percentileofscore(safe_ls_LY, safe_scor_LY, nan_policy='omit'), 1)
            read_perc_LY = round(percentileofscore(read_ls_LY, read_scor_LY, nan_policy='omit'), 1)
            pexp_perc_LY = round(percentileofscore(pexp_ls_LY, pexp_scor_LY, nan_policy='omit'), 1)
            proc_perc_LY = round(percentileofscore(proc_ls_LY, proc_scor_LY, nan_policy='omit'), 1)
            
            
            # Get values for next latest year
            if yr1 != 2021:
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
                
                summ_perc_PY = round(percentileofscore(summ_ls_PY, summ_scor_PY, nan_policy='omit'), 1)
                mort_perc_PY = round(percentileofscore(mort_ls_PY, mort_scor_PY, nan_policy='omit'), 1)
                safe_perc_PY = round(percentileofscore(safe_ls_PY, safe_scor_PY, nan_policy='omit'), 1)
                read_perc_PY = round(percentileofscore(read_ls_PY, read_scor_PY, nan_policy='omit'), 1)
                pexp_perc_PY = round(percentileofscore(pexp_ls_PY, pexp_scor_PY, nan_policy='omit'), 1)
                proc_perc_PY = round(percentileofscore(proc_ls_PY, proc_scor_PY, nan_policy='omit'), 1)
                
            elif yr1 == 2021:
                summ_scor_PY = np.nan
                mort_scor_PY = np.nan
                safe_scor_PY = np.nan
                read_scor_PY = np.nan
                pexp_scor_PY = np.nan
                proc_scor_PY = np.nan
                mort_wt_PY = np.nan
                safe_wt_PY = np.nan
                read_wt_PY = np.nan
                pexp_wt_PY = np.nan
                proc_wt_PY = np.nan
                summ_perc_PY = np.nan
                mort_perc_PY = np.nan
                safe_perc_PY = np.nan
                read_perc_PY = np.nan
                pexp_perc_PY = np.nan
                proc_perc_PY = np.nan
                
            # compute values for columns
            domains = ['Summary Score', 'Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
            
            values_LY = [summ_scor_LY, mort_scor_LY, safe_scor_LY, read_scor_LY, pexp_scor_LY, proc_scor_LY]
            values_PY = [summ_scor_PY, mort_scor_PY, safe_scor_PY, read_scor_PY, pexp_scor_PY, proc_scor_PY]
            delta_value = np.round(np.array(values_LY) - np.array(values_PY), 3)
            
            perc_LY = [summ_perc_LY, mort_perc_LY, safe_perc_LY, read_perc_LY, pexp_perc_LY, proc_perc_LY]
            perc_PY = [summ_perc_PY, mort_perc_PY, safe_perc_PY, read_perc_PY, pexp_perc_PY, proc_perc_PY]
            delta_perc = np.round(np.array(perc_LY) - np.array(perc_PY), 3)
            
            wght_LY = [mort_wt_LY, safe_wt_LY, read_wt_LY, pexp_wt_LY, proc_wt_LY] 
            wght_PY = [mort_wt_PY, safe_wt_PY, read_wt_PY, pexp_wt_PY, proc_wt_PY] 
            delta_wght = np.round(np.array(wght_LY) - np.array(wght_PY), 3)
            
            cols = ['Domain', 'Value', 'Delta value', 
                    'Percentile of Peer Group', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Domain'] = domains
            df_table['Value'] = np.round(values_LY, 3)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Percentile of Peer Group'] = perc_LY
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
            
            txt = "The latest year of data used was " + str(yr1) + ". "
            if yr1 != 2021:
                txt += "Delta values were computed using the year " + str(yr2) + ". "
            elif yr1 == 2021:
                txt += "Delta values are not available for 2021 because this app does not provide overall star rating results for 2020."
            
            
            if np.isnan(grp_LY) == True and np.isnan(grp_PY) == True:
                txt += "In both years, " + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_LY) == True:
                txt += "In " + str(yr1) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            elif np.isnan(grp_PY) == True and yr1 != 2021:
                txt += "In " + str(yr2) + ', '  + name1 + " was not assigned to a peer group and did not receive a star rating."
            
            elif grp_LY == grp_PY and yr1 != 2021:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                txt += "In both years, " + name1 + " was in group " + str(int(grp_LY)) + numD_LY + '.'
            elif grp_LY != grp_PY and yr1 != 2021:
                numD_LY = ' (hospitals w/ scores in ' + str(int(grp_LY + 2)) + ' domains)'
                numD_PY = ' (hospitals w/ scores in ' + str(int(grp_PY + 2)) + ' domains)'
                
                txt += name1 + " was in group " + str(int(grp_PY)) + numD_PY + " in " + str(yr2)
                txt += " and in group " + str(int(grp_LY)) + numD_LY + " in " + str(yr1) + ". "
                
            return dashT, txt
        
        else:
            return dashT, ''
        
    elif set_select == 'Selected hospitals':

        cols = ['Domain', 'Value', 'Change in value from previous year', 
                'Percentile of Selected Hospitals', 'Change in percentile', 
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
                
            txt = ''
            if hospital is None:
                txt = "Please select a hospital."
            if option_hospitals is None or option_hospitals == []:
                txt += "You either haven't selected any hospitals or the filters you chose left you with no hospitals to analyze."
            
            return dashT, txt
        
    
        if hospital in option_hospitals:
            # Remove focal hospital from options, for use below
            option_hospitals.remove(hospital)
            
        # At this point, we can still get results even selected_hospitals and option_hospitals are empty lists. 
        #    That is, even if we've ended up with nothing to compare our hospital to.
           
        tdf_main = main_df[main_df['Name and Num'].isin(option_hospitals + [hospital])]
                
        name = hospital[:-9]
        hosp_df = tdf_main[tdf_main['Name and Num'] == hospital] 
        
        if hosp_df.shape[0] == 0:    
            txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
            return dashT, txt
        
        yrs = sorted(hosp_df['Release year'].unique().tolist())
        yr1 = int(yr)
        
        if yr1 == 2027:
            yr2 = 2026
            if yr2 in yrs:
                pass
            else:
                yr2 = 2025
                
        if yr1 == 2026:
            yr2 = 2025
            if yr2 in yrs:
                pass
            else:
                yr2 = 2024
                
        if yr1 == 2025:
            yr2 = 2024
            if yr2 in yrs:
                pass
            else:
                yr2 = 2023
                
        if yr1 == 2024:
            yr2 = 2023
            if yr2 in yrs:
                pass
            else:
                yr2 = 2022
                
        elif yr1 == 2023:
            yr2 = 2022
            if yr2 in yrs:
                pass
            else:
                yr2 = 2021
                
        elif yr1 == 2022:
            yr2 = 2021
            
        elif yr1 == 2021:
            yr2 = 2021
        
        if yr1 in yrs and yr2 in yrs:
            
            tdf_main_LY = tdf_main[tdf_main['Release year'] == yr1]
            
            if yr1 == 2021:
                tdf_main_PY = None
            
            if yr1 != 2021:
                tdf_main_PY = tdf_main[tdf_main['Release year'] == yr2]
                
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
                txt = hospital + " had no data among the CMS Stars release years. Try selecting another hospital."
                return dashT, txt
            
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
            
            summ_perc_LY = round(percentileofscore(summ_ls_LY, summ_scor_LY, nan_policy='omit'), 1)
            mort_perc_LY = round(percentileofscore(mort_ls_LY, mort_scor_LY, nan_policy='omit'), 1)
            safe_perc_LY = round(percentileofscore(safe_ls_LY, safe_scor_LY, nan_policy='omit'), 1)
            read_perc_LY = round(percentileofscore(read_ls_LY, read_scor_LY, nan_policy='omit'), 1)
            pexp_perc_LY = round(percentileofscore(pexp_ls_LY, pexp_scor_LY, nan_policy='omit'), 1)
            proc_perc_LY = round(percentileofscore(proc_ls_LY, proc_scor_LY, nan_policy='omit'), 1)
            
            if yr1 != 2021:
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
                
                summ_perc_PY = round(percentileofscore(summ_ls_PY, summ_scor_PY, nan_policy='omit'), 1)
                mort_perc_PY = round(percentileofscore(mort_ls_PY, mort_scor_PY, nan_policy='omit'), 1)
                safe_perc_PY = round(percentileofscore(safe_ls_PY, safe_scor_PY, nan_policy='omit'), 1)
                read_perc_PY = round(percentileofscore(read_ls_PY, read_scor_PY, nan_policy='omit'), 1)
                pexp_perc_PY = round(percentileofscore(pexp_ls_PY, pexp_scor_PY, nan_policy='omit'), 1)
                proc_perc_PY = round(percentileofscore(proc_ls_PY, proc_scor_PY, nan_policy='omit'), 1)
            
            elif yr1 == 2021:
                summ_scor_PY = np.nan
                mort_scor_PY = np.nan
                safe_scor_PY = np.nan
                read_scor_PY = np.nan
                pexp_scor_PY = np.nan
                proc_scor_PY = np.nan
                mort_wt_PY = np.nan
                safe_wt_PY = np.nan
                read_wt_PY = np.nan
                pexp_wt_PY = np.nan
                proc_wt_PY = np.nan
                summ_perc_PY = np.nan
                mort_perc_PY = np.nan
                safe_perc_PY = np.nan
                read_perc_PY = np.nan
                pexp_perc_PY = np.nan
                proc_perc_PY = np.nan
                
            # compute values for columns
            domains = ['Summary Score', 'Mortality', 'Safety of Care', 'Readmission', 'Patient Experience', 'Timely and Effective Care']
            
            values_LY = [summ_scor_LY, mort_scor_LY, safe_scor_LY, read_scor_LY, pexp_scor_LY, proc_scor_LY]
            values_PY = [summ_scor_PY, mort_scor_PY, safe_scor_PY, read_scor_PY, pexp_scor_PY, proc_scor_PY]
            delta_value = np.round(np.array(values_LY) - np.array(values_PY), 3)
            
            perc_LY = [summ_perc_LY, mort_perc_LY, safe_perc_LY, read_perc_LY, pexp_perc_LY, proc_perc_LY]
            perc_PY = [summ_perc_PY, mort_perc_PY, safe_perc_PY, read_perc_PY, pexp_perc_PY, proc_perc_PY]
            delta_perc = np.round(np.array(perc_LY) - np.array(perc_PY), 3)
            
            wght_LY = [mort_wt_LY, safe_wt_LY, read_wt_LY, pexp_wt_LY, proc_wt_LY] 
            wght_PY = [mort_wt_PY, safe_wt_PY, read_wt_PY, pexp_wt_PY, proc_wt_PY] 
            delta_wght = np.round(np.array(wght_LY) - np.array(wght_PY), 3)
            
            cols = ['Domain', 'Value', 'Delta value', 
                    'Percentile of Selected Hospitals', 'Delta percentile', 
                    'Weight', 'Delta weight']
            
            df_table = pd.DataFrame(columns=cols)
            df_table['Domain'] = domains
            df_table['Value'] = np.round(values_LY, 3)
            df_table['Delta value'] = delta_value.tolist()
            df_table['Percentile of Selected Hospitals'] = perc_LY
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
            
            txt = "The latest year of data used was " + str(yr1) + ". "
            
            if yr1 != 2021:
                txt += "Delta values were computed using the year " + str(yr2) + ". "
                txt += "Delta's are color-coded (green = improved; red = worsened)."
            elif yr1 == 2021:
                txt = "Delta values are not available for 2021 because this app does not provide overall star rating results for 2020."
            
            return dashT, txt
        
        else:
            return dashT, ''


    
    


@app.callback(
     [Output("data_report_plot3", "children"),
      Output("text10", "children"),
      ],
     [Input("hospital-select1b", "value"),
      Input("option_hospitals", 'children'),
      Input("selected_hosps_btn3", "n_clicks"),
      Input("stars_peers_btn3", "n_clicks"),
      Input('domain-select1', "value"),
      Input('score-type1', 'value'),
      Input('year-select3', "value"),
      ],
    )
def update_panel4(hospital, option_hospitals, selected_hosps_btn, stars_peers_btn, domain, score_type, yr):    
    
    cols = ['Measure', 'Value', 'Delta value', 
            'Percentile', 'Delta percentile']
    
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
    
    if yr1 == 2027:
        yr2 = 2026
        if yr2 in yrs:
            pass
        else:
            yr2 = 2025
            
    if yr1 == 2026:
        yr2 = 2025
        if yr2 in yrs:
            pass
        else:
            yr2 = 2024
            
    if yr1 == 2025:
        yr2 = 2024
        if yr2 in yrs:
            pass
        else:
            yr2 = 2023
            
    if yr1 == 2024:
        yr2 = 2023
        if yr2 in yrs:
            pass
        else:
            yr2 = 2022
        
    if yr1 == 2023:
        yr2 = 2022
        if yr2 in yrs:
            pass
        else:
            yr2 = 2021
            
    elif yr1 == 2022:
        yr2 = 2021
    
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:150]
    
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
            
        
        tdf_main = main_df.copy() #[main_df['Name and Num'].isin(option_hospitals + [hospital])]
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
        measure_ls = features.feature_dict[domain + ' (std)']
    elif score_type == 'Raw scores':
        measure_ls = features.feature_dict[domain]
    
    hosp_scors_LY = []
    hosp_percs_LY = []
    hosp_wts_LY = []
    
    # Get values for latest year
    labels_ls = []
    for ii, m in enumerate(measure_ls):
        try:
            ls = tdf_main_LY[m].tolist()
            i_score = ls[i]
            hosp_scors_LY.append(i_score)
            ls = [x for x in ls if x == x]
            
            perc = round(percentileofscore(ls, i_score, nan_policy='omit'), 1)
            hosp_percs_LY.append(perc)
            
            ls2 = features.feature_dict[domain + ' labels']
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
        #hosp_wts_PY = [np.nan] * len(hosp_wts_LY)
        
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
            measure_ls = features.feature_dict[domain + ' (std)']
        elif score_type == 'Raw scores':
            measure_ls = features.feature_dict[domain]
                    
        hosp_scors_PY = []
        hosp_percs_PY = []
        #hosp_wts_PY = []
            
        # Get values for next latest year
        labels_ls = []
        for ii, m in enumerate(measure_ls):
            try:
                ls = tdf_main_PY[m].tolist()
                if ls is None or ls == [] or len(ls) == 0:
                    continue
                
                i_score = ls[i]
                hosp_scors_PY.append(i_score)
                ls = [x for x in ls if x == x]
                
                perc = round(percentileofscore(ls, i_score, nan_policy='omit'), 1)
                hosp_percs_PY.append(perc)
                
                ls2 = features.feature_dict[domain + ' labels']
                labels_ls.append(ls2[ii])
                        
                # get individual measure weight ... somehow
                  
                '''
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
                '''
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
    


def _wrap(label, width=TS_LEGEND_WRAP_CHARS):
    """
    Wrap a legend label at word boundaries so it fits its fixed-width legend
    column without being cut off. Returns (wrapped_text, n_lines).
    """
    lines = textwrap.wrap(str(label), width=width) or ['']
    return '<br>'.join(lines), len(lines)


def _legend_height_px(line_counts, n_cols=TS_LEGEND_COLS):
    """
    Height of the legend band. Entries fill rows of n_cols in trace order;
    each row is as tall as its most-wrapped entry.
    """
    if not line_counts:
        return TS_LEGEND_ROW_PX
    total = 0
    for i in range(0, len(line_counts), n_cols):
        max_lines = max(line_counts[i:i + n_cols])
        total += TS_LEGEND_ROW_PX + (max_lines - 1) * TS_LEGEND_LINE_PX
    return total


def _measure_label_pairs(domain, score_type):
    """
    Return [(column, label), ...] for the selected domain and score type,
    keeping only columns that exist in main_df.
    Percentiles are computed from the standardized-score columns.
    """
    if score_type == 'Raw scores':
        key = domain
    elif score_type in ('Standardized scores', 'Percentiles'):
        key = domain + ' (std)'
    else:
        return []

    measures = features.feature_dict[key]
    labels = features.feature_dict[domain + ' labels']
    return [(m, lab) for m, lab in zip(measures, labels) if m in main_df.columns]


def _group_by_label(pairs):
    """
    Collapse [(column, label), ...] into {label: [column, ...]}, keeping the
    order in which each label first appears. Columns sharing a label (e.g. the
    pre- and post-2026 versions of 'Nurse Communication') end up in one list,
    so they are drawn as a single trace.
    """
    groups = {}
    for m, lab in pairs:
        groups.setdefault(lab, []).append(m)
    return groups


def _percentile_series(measure, hospital):
    """
    Percentile (0-100) of `hospital` on `measure`, ranked against all hospitals
    with a non-missing value in the same release year.
    Returns a DataFrame with columns ['Release year', 'y'], sorted by year.
    """
    tmp = (
        main_df[['Release year', 'Name and Num', measure]]
        .dropna(subset=[measure])
    )
    tmp = tmp.assign(
        y=tmp.groupby('Release year')[measure].rank(pct=True) * 100
    )
    return (
        tmp.loc[tmp['Name and Num'] == hospital, ['Release year', 'y']]
        .sort_values('Release year')
    )


def _value_series(hosp_df, measure):
    """
    Raw or standardized values of `measure` for one hospital.
    Returns a DataFrame with columns ['Release year', 'y'], sorted by year.
    """
    return (
        hosp_df[['Release year', measure]]
        .dropna(subset=[measure])
        .rename(columns={measure: 'y'})
        .sort_values('Release year')
    )


@app.callback(
    Output("data_report_plot5", "children"),
    [
        Input("hospital-select1b", "value"),
        Input("selected_hosps_btn3", "n_clicks"),
        Input("stars_peers_btn3", "n_clicks"),
        Input("domain-select1", "value"),
        Input("score-select1", "value"),
    ],
)
def update_measure_timeseries(hospital, n_clicks1, n_clicks2, domain, score_type):

    if hospital is None or domain is None or score_type is None:
        return None

    # ---------- WHICH BUTTON FIRED ----------
    # Reads the triggering component ids directly; works across Dash versions.
    triggered_ids = {
        t['prop_id'].split('.')[0] for t in dash.callback_context.triggered
    }
    if 'selected_hosps_btn3' in triggered_ids:
        set_select = 'Selected hospitals'
    else:
        set_select = 'Measures group'
    # NOTE: set_select is not yet used below (the original branch was `pass`);
    # it is retained so the selection logic is ready when you need it.

    hosp_df = main_df[main_df['Name and Num'] == hospital]
    if hosp_df.empty:
        return None

    groups = _group_by_label(_measure_label_pairs(domain, score_type))
    is_pct = (score_type == 'Percentiles')
    y_hover = "Percentile: %{y:.1f}" if is_pct else "Score: %{y:.3f}"

    # ---------- TRACES (one per label) ----------
    fig = go.Figure()
    legend_line_counts = []     # wrapped line count per trace, in trace order

    for lab, cols in groups.items():

        # Build each column's series separately (percentiles are ranked within
        # each column and release year), then stitch them into one series.
        parts = [
            _percentile_series(m, hospital) if is_pct else _value_series(hosp_df, m)
            for m in cols
        ]
        ts = (
            pd.concat(parts, ignore_index=True)
            .sort_values('Release year', kind='stable')
            # Columns sharing a label are expected to cover different years.
            # If a year ever appears in more than one, keep the first column's value.
            .drop_duplicates(subset='Release year', keep='first')
        )

        hover = (
            f"<b>{lab}</b><br>"
            "Release year: %{x}<br>"
            f"{y_hover}"
            "<extra></extra>"
        )

        legend_text, n_lines = _wrap(lab)
        legend_line_counts.append(n_lines)

        fig.add_trace(
            go.Scatter(
                x=ts['Release year'].astype(int),
                y=ts['y'],
                mode='markers' if ts.shape[0] < 2 else 'lines+markers',
                name=legend_text,           # full label, wrapped onto lines
                connectgaps=False,
                hovertemplate=hover,        # full label on one line
            )
        )

    # ---------- GEOMETRY: fixed plot area, variable legend band ----------
    margin_top = (
        TS_MARGIN_TOP_BASE
        + _legend_height_px(legend_line_counts)
        + TS_LEGEND_PAD_PX
    )

    fig.update_layout(
        paper_bgcolor='#4d4d4d',
        template="plotly_dark",
        xaxis=dict(
            title="Release year",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title=TS_YLABELS.get(score_type, "Score")),
        margin=dict(
            l=TS_MARGIN_LEFT,
            r=TS_MARGIN_RIGHT,
            t=margin_top,
            b=TS_MARGIN_BOTTOM,
            autoexpand=False,       # stop Plotly from growing margins on its own
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            # Each entry gets a fixed fraction of the plot width -> fixed column grid.
            # Slightly under 1/N so that exactly N entries fit per row.
            entrywidth=1 / TS_LEGEND_COLS - 0.005,
            entrywidthmode="fraction",
            valign="top",   # align each swatch with the first line of a wrapped label
        ),
    )

    # Reference line for standardized scores
    if score_type == 'Standardized scores':
        fig.add_hline(y=0, line_dash="dot", line_color="white")

    # Plot area = TS_PLOT_VW vw exactly; the px term covers legend band + margins.
    height = f"calc({TS_PLOT_VW}vw + {margin_top + TS_MARGIN_BOTTOM}px)"

    return dcc.Graph(
        figure=fig,
        config={"responsive": True},
        style={"height": height},
    )


@app.callback(
    Output("data_report_plot6", "children"),
    [
        Input("hospital-select1b", "value"),
        Input("score-select2", "value"),
        Input("open-that", "n_clicks"),
    ],
    prevent_initial_call=True
)
def update_domains_timeseries(hospital, score_type, n_clicks):

    if hospital is None:
        return None

    hosp_df = main_df[main_df['Name and Num'] == hospital].copy()
    if hosp_df.empty:
        return None

    labels = features.feature_dict['Stars Domains'] + ['Stars Summary Score']
    measures = features.feature_dict['Standardized scores']
    
    fig = go.Figure()

    for m, lab in zip(measures, labels):

        if m not in main_df.columns:
            continue

        # ---------- BUILD TIME SERIES ----------
        if score_type == 'Percentiles':

            # Compute percentiles relative to all hospitals in each release year
            tmp = (
                main_df[['Release year', 'Name and Num', m]]
                .dropna(subset=[m])
                .copy()
            )

            tmp['percentile'] = (
                tmp
                .groupby('Release year')[m]
                .rank(pct=True) * 100
            )

            ts = (
                tmp[tmp['Name and Num'] == hospital][['Release year', 'percentile']]
                .sort_values('Release year')
            )

            yvalues = ts['percentile']
            hover = (
                f"<b>{lab}</b><br>"
                "Release year: %{x}<br>"
                "Percentile: %{y:.1f}"
                "<extra></extra>"
            )

        else:
            # Raw or standardized value
            ts = (
                hosp_df[['Release year', m]]
                .dropna(subset=[m])
                .sort_values('Release year')
            )

            yvalues = ts[m]
            hover = (
                f"<b>{lab}</b><br>"
                "Release year: %{x}<br>"
                "Score: %{y:.3f}"
                "<extra></extra>"
            )

        if ts.shape[0] < 2:
            mode = 'markers'
        else:
            mode = 'lines+markers'

        fig.add_trace(
            go.Scatter(
                x=ts['Release year'].astype(int),
                y=yvalues,
                mode=mode,
                name=lab,
                connectgaps=False,
                hovertemplate=hover,
            )
        )

    # Axis label
    if score_type == 'Standardized scores':
        ylab = "Standardized score"
    elif score_type == 'Percentiles':
        ylab = "Percentile"
        # (relative to all hospitals in same release year)
    else:
        ylab = "Raw score"

    fig.update_layout(
        paper_bgcolor='#4d4d4d',
        xaxis=dict(
            title="Release year",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(title=ylab),
        template="plotly_dark",
        margin=dict(l=80, r=40, t=60 + 5*len(labels), b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    # Reference line for standardized scores
    if score_type == 'Standardized scores':
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="white",
        )

    height = str(30 + 1.5*len(labels)) + "vw"

    return dcc.Graph(
        figure=fig,
        config={"responsive": True},
        #config={"displayModeBar": False},
        style={"height": height, 
               #"width": "50vw",
               },
    )


@app.callback(
     [Output("data_report_plot4", "data"),
      Output("data_report_plot4", "columns"),
      Output("whatif_df", "data"),
      Output("text-what-if", "children"),
      ],
     [Input("hospital-select1b", "value"),
      Input("reset-table", "n_clicks"),
      Input("year-select4", "value"),
      ],
    )
def update_whatif_table(hospital, n_clicks, yr):    
    
    if hospital is None:
        raise PreventUpdate
    
    measures = []
    domains = []
    
    m1 = [
        'H_CLEAN_LINEAR_SCORE', 
        'H_COMP_1_LINEAR_SCORE', 
        'H_COMP_2_LINEAR_SCORE', 
        'H_COMP_3_LINEAR_SCORE', 
        'H_COMP_5_LINEAR_SCORE', 
        'H_COMP_6_LINEAR_SCORE', 
        'H_COMP_7_LINEAR_SCORE', 
        'H_HSP_RATING_LINEAR_SCORE', 
        'H_QUIET_LINEAR_SCORE', 
        'H_RECMND_LINEAR_SCORE',
        'O-COMP-1',
        'O-COMP-2',
        'O-COMP-3',
        'O-PATIENT-RATE',
        'O-PATIENT-REC',
        ]
    if yr == 2027:
        m1 = [
            'H_CLEAN_LINEAR_SCORE', 
            'H_COMP_1_LINEAR_SCORE', 
            'H_COMP_2_LINEAR_SCORE', 
            #'H_COMP_3_LINEAR_SCORE', 
            'H_COMP_5_LINEAR_SCORE', 
            'H_COMP_6_LINEAR_SCORE', 
            #'H_COMP_7_LINEAR_SCORE', 
            'H_HSP_RATING_LINEAR_SCORE', 
            'H_QUIET_LINEAR_SCORE', 
            'H_RECMND_LINEAR_SCORE',
            'O-COMP-1',
            'O-COMP-2',
            'O-COMP-3',
            'O-PATIENT-RATE',
            'O-PATIENT-REC',
            ]
        
        
    domains.extend(['Patient Experience']*len(m1))
    measures.extend(m1)
    
    m2 = []
    if yr == 2026:
        m2 = [
            'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
            'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
            'Hybrid_HWR', 'OP_35_ADM', 'OP_35_ED', 'OP_36',
            ]
    if yr == 2027:
        m2 = [
            'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
            'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
            'Hybrid_HWR', 'OP_35_ADM', 'OP_35_ED', 'OP_36',
            ]
        
    domains.extend(['Readmission']*len(m2))
    measures.extend(m2)
    
    m3 = [
        'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF', 
        'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'Hybrid_HWM',
        ]
    domains.extend(['Mortality']*len(m3))
    measures.extend(m3)
    
    m4 = ['COMP_HIP_KNEE',  'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 
          'HAI_5', 'HAI_6', 'PSI_90_SAFETY',
          ]
    domains.extend(['Safety of Care']*len(m4))
    measures.extend(m4)
    
    m5 = [
        'IMM_3', 'OP_10', 'OP_13', 'OP_18B',
        'OP_22', 'OP_23', 'OP_29',
        'OP_8', 
        'SEP_1', 
        'SAFE_USE_OF_OPIOIDS',
        ]
    if yr == 2027:
        m5 = [
            'IMM_3', 'OP_10', 
            #'OP_13', 
            'OP_18B',
            'OP_22', 'OP_23', 'OP_29',
            #'OP_8', 
            'SEP_1', 
            'SAFE_USE_OF_OPIOIDS',
            ]
    
    rev_measures = [
        'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
        'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'COMP_HIP_KNEE', 
        'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6',
        'PSI_90_SAFETY', 'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN',
        'OP_32', 'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE', 
        'Hybrid_HWR', 
        'Hybrid_HWM',
        'OP_35_ADM', 'OP_35_ED', 'OP_36', 'OP_22',
        'OP_18B', 'OP_8', 'OP_10','OP_13', 
        'SAFE_USE_OF_OPIOIDS',
        ]
    
    if yr == 2027:
        rev_measures = [
            'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
            'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP', 'COMP_HIP_KNEE', 
            'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6',
            'PSI_90_SAFETY', 'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN',
            'OP_32', 'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE', 
            'Hybrid_HWR', 
            'Hybrid_HWM',
            'OP_35_ADM', 'OP_35_ED', 'OP_36', 'OP_22',
            'OP_18B', 
            #'OP_8', 
            'OP_10',
            #'OP_13', 
            'SAFE_USE_OF_OPIOIDS',
            ]
        
    domains.extend(['Timely & Effective Care']*len(m5))
    measures.extend(m5)
    
    higher_better = []
    for m in measures:
        if m in rev_measures:
            higher_better.append('No')
        else:
            higher_better.append('Yes')
    
    pnum = re.search(r'\d{6}', hospital)
    pnum = pnum.group()
    
    tdf = main_df[main_df['Release year'] == yr]

    # Keep Facility ID for filtering, but compute stats only on measures
    tdf = tdf[['Facility ID'] + measures]
    
    try:
        tdf = tdf[tdf['Facility ID'] == pnum]
    except Exception:
        return {}, {}, {}, hospital + ' has no data in the CMS Care Compare release that is used to make predictions.'
    
    if tdf.empty:
        return {}, {}, {}, hospital + ' has no data in the CMS Care Compare release that is used to make predictions.'
    
    # --- Extract measure values ONLY (exclude Facility ID) ---
    #actual_vals = tdf.loc[:, measures].iloc[0].tolist()
    actual_vals = tdf.loc[:, measures].iloc[0].round(6).tolist()
    
    # --- Min / Max across all hospitals, measures only ---
    mins = main_df.loc[
        main_df['Release year'] == yr, measures
    ].min().round(3).tolist()
    
    maxs = main_df.loc[
        main_df['Release year'] == yr, measures
    ].max().round(3).tolist()
    
    # Map Measure-ID to Measure Name using features.feature_dict
    measure_names = []
    for m in measures:
        # Check each domain for the measure
        found = False
        for domain in features.feature_dict['Stars Domains']:
            if m in features.feature_dict[domain]:
                idx = features.feature_dict[domain].index(m)
                measure_names.append(features.feature_dict[domain + ' labels'][idx])
                found = True
                break
            elif m in features.feature_dict.get(domain + ' (std)', []):
                idx = features.feature_dict[domain + ' (std)'].index(m)
                measure_names.append(features.feature_dict[domain + ' labels'][idx])
                found = True
                break
        if not found:
            measure_names.append(m)  # fallback to ID if no label found
        
    df_table = pd.DataFrame({
        'Domain': domains,
        'Measure-ID': measures,
        'Measure Name': measure_names,
        'Higher is better': higher_better,
        'Actual value': actual_vals,
        'Min value': mins,
        'Max value': maxs,
        'What-if value': actual_vals
    })
    
    data = df_table.to_dict('records')
    
    columns = [
        {'id': c, 'name': c, 'editable': (c == 'What-if value')}
        for c in df_table.columns
    ]
    
    return data, columns, df_table.to_json(), ''

    



@app.callback(
    [Output("modal-boxtext1", "children"),
     Output("modal-boxtext2", "children"),
     Output("modal-boxtext3", "children"),
     Output("modal-boxtext4", "children"),
     Output("modal-boxtext5", "children"),
     ],
    [
     Input("whatif_button", "n_clicks"),
      ],
     [State("data_report_plot4", "data"),
      State("data_report_plot4", "columns"),
      State("hospital-select1b", "value"), 
      State("option_hospitals", 'children'),
      State("year-select4", "value"),
      ],
     prevent_initial_call=True,
    )
def update_whatif_analysis(n_clicks, data, columns, hospital, filtered_hospitals, yr):   
    
    column_names = [col['id'] for col in columns]
    df = pd.DataFrame(data, columns=column_names)
    
    if df is None:
        return ["Yikes! There's no data!"]
    
    df['What-if value'] = df['What-if value'].astype(float)
    
    pnum = re.search(r'\d{6}', hospital)
    pnum = pnum.group()
    
    tdf = None
    if yr == 2026:
        tdf = what_if_df_2026.copy()
    elif yr == 2027:
        tdf = what_if_df_2027.copy()
        
    measures = df['Measure-ID'].tolist()
    vals = df['What-if value'].tolist()
    for i, m in enumerate(measures):
        tdf.loc[tdf['PROVIDER_ID'] == pnum, m] = vals[i]
        
    stars_output_df = what_if_fxn.run_whatif(tdf, pnum, yr)
    
    tdf = stars_output_df[stars_output_df['PROVIDER_ID'] == pnum]
    star = tdf['star'].iloc[0]
    score = tdf['summary_score'].iloc[0]
    grp = tdf['cnt_grp'].iloc[0]
    
    stars_output_df.sort_values(by='PROVIDER_ID', ascending=True, inplace=True)
    prvdrs1 = stars_output_df['PROVIDER_ID'].tolist()
    
    ct = 0
    for p in prvdrs1:
        if 'F' in p:
            ct += 1
    
    results_yr_df = main_df[main_df['Release year'] == yr]
    results_yr_df.drop_duplicates(inplace=True)
    results_yr_df = results_yr_df[results_yr_df['Facility ID'].isin(prvdrs1)]
    
    results_yr_df.sort_values(by='Facility ID', ascending=True, inplace=True)
    prvdrs2 = results_yr_df['Facility ID'].tolist()
    
    ct = 0
    for p in prvdrs2:
        if 'F' in p:
            ct += 1
    
    if prvdrs1 == prvdrs2:
        stars_output_df['Name and Num'] = results_yr_df['Name and Num'].tolist()
        
    else:
        for i, p1 in enumerate(prvdrs1):
            p2 = prvdrs2[i]
            if p1 != p2:
                pass
                
                
    txt1 = ''
    txt2 = ''
    txt3 = ''
    txt4 = ''
    txt5 = ''
    
    if hospital in filtered_hospitals:
        'Remove focal hospital from options, for use below'
        filtered_hospitals.remove(hospital)
        
    
    filtered_hosps_df = stars_output_df[stars_output_df['Name and Num'].isin(filtered_hospitals + [hospital])]
    if filtered_hosps_df['Name and Num'].unique().tolist() == [hospital]:
        txt1 += " is the only hospital in the data."
        return txt1, '', '', '', ''
    
    #for i, m in enumerate(measures):
    #    tdf.loc[tdf['Facility ID'] == pnum, m] = vals[i]
    
    
    txt1 = str(star) + ' Star'
    score = round(score, 4)
    txt2 = str(score) + ' Summary Score'
    grp_num = int(grp[0])
    
    if grp_num == 3:
        txt3 = 'Peer Group 3: At least 3 measures in each of 3 domains'
    elif grp_num == 2:
        txt3 = 'Peer Group 2: At least 3 measures in each of 2 domains'
    elif grp_num == 1:
        txt3 = 'Peer Group 1: At least 3 measures in 1 domain'
            
    tdf_peer = stars_output_df[stars_output_df['cnt_grp'] == grp]
    tdf_peer = tdf_peer[tdf_peer['star'] == star]
    perc_of_peer = round(percentileofscore(tdf_peer['summary_score'], score, nan_policy='omit'), 1)
    txt4 = str(perc_of_peer) + " percentile of " + str(star) + " Star group " + str(grp_num) + " hospitals"
        
    tdf_filtered = filtered_hosps_df[~filtered_hosps_df['summary_score'].isin([np.nan, float("NaN")])]
    perc_of_chosen = round(percentileofscore(tdf_filtered['summary_score'], score, nan_policy='omit'), 1)
    txt5 = str(perc_of_chosen) + " percentile of hospitals in your filters"

    
    return txt1, txt2, txt3, txt4, txt5


#########################################################################################


# Run the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True) # modified to run on linux server
