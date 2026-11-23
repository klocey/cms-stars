"""Page layout for the app: the sidebar controls and the main panels."""

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table

from app_data import (
    states, htypes, ctypes, latest_yr, latest_release_yr,
)


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


def create_layout(app):
    """Return the page layout. `app` is used only to build the image (asset) URLs."""
    return html.Div([
    
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
