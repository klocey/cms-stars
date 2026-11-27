"""Page layout for the app: the sidebar controls and the main panels."""

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table

from app_data import (
    states, htypes, ctypes, latest_yr, latest_release_yr, year_label,
)


def comparison_switch(switch_id, style=None):
    """Two-option switch that sets what the focal hospital is compared to.
    The selected option stays highlighted and is kept when the hospital or year changes."""
    return dbc.RadioItems(
        id=switch_id,
        className="btn-group compare-switch",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-primary",
        labelCheckedClassName="active",
        options=[
            {"label": "Compare to Stars Peer Group", "value": "Measures group"},
            {"label": "Compare to Filtered Hospitals", "value": "Selected hospitals"},
        ],
        value="Measures group",
        style=style or {},
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
                keyboard=False,   # Esc does not close it; use "Save & Close"
                fade=True,
                backdrop="static",   # clicking outside does not close it
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
                keyboard=False,   # Esc does not close it; use "Save & Close"
                fade=True,
                backdrop="static",   # clicking outside does not close it
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
                keyboard=False,   # Esc does not close it; use "Save & Close"
                fade=True,
                backdrop="static",   # clicking outside does not close it
                ),
            html.Br(),
            html.Br(),
            
            html.Div(id='Filterbeds1'),
            dcc.RangeSlider(
                id='beds1',
                min=0,
                max=2700,   # matches the 2700 mark and the 'all beds' value
                step=50,
                marks={
                        100: '100',
                        500: '500',
                        1000: '1000',
                        1500: '1500',
                        2000: '2000',
                        2700: 'Max',   # the top of the range includes every hospital
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
                                options=[{"label": year_label(i)+'  ', "value": i} for i in list(range(2021, latest_yr+1))],
                                value=latest_yr,
                                placeholder='Select a Stars year',
                                optionHeight=50,
                                style={
                                    'min-width': '170px',   # room for the '(predicted)' label
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
                        
                            comparison_switch("compare-select1",
                                              style={'display': 'inline-block', 'margin-left': '8%', 'verticalAlign': 'bottom'}),
                            
                            # Which filters are in use (filled in by a callback). The outer Div is
                            # shown only when the switch is on "Compare to Filtered Hospitals".
                            html.Div(id="active-filters-box",
                                     style={'display': 'none'},
                                     children=html.Div(id="active-filters",
                                                       style={'fontSize': '13px', 'color': '#555555', 'margin': '10px 0px 0px 1%'}),
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
                                html.H5("Scores across domains", style={'color': 'white'}),
                                dcc.Dropdown(
                                    id='year-select2',
                                    clearable=False,
                                    className='dark-dropdown',
                                    options=[{"label": year_label(i), "value": i} for i in list(range(2021, latest_yr+1))],
                                    value=latest_yr,
                                    placeholder='Select a Stars year',
                                    optionHeight=50,
                                    style={
                                        'min-width': '170px',   # room for the '(predicted)' label
                                        'display': 'inline-block',
                                        'padding': '0px 30px 0px 20px',
                                        'verticalAlign': 'bottom',
                                        'margin-left': '0%',
                                        },
                                    ),
                            
                                comparison_switch("compare-select2",
                                                  style={'display': 'inline-block', 'margin-left': '3%', 'vertical-align': 'top'}),
                            
                                html.Hr(),
                                html.Br(),
                                html.Div(id="data_report_plot2"),
                                html.Hr(),
                                html.B(id="text3",
                                       style={'fontSize':16, 'color': 'white',
                                              'margin-left':'1%',
                                              },
                                       ),
                            
                                html.Hr(),
                                # Domain time series, with its header and score-type buttons
                                # in the same dark container.
                                html.Div([
                                    html.H5("Time series",
                                            style={'color': 'white', 'display': 'block',
                                                   'padding': '15px 20px 0 20px'}),
                                    dbc.RadioItems(
                                        id="score-select2",
                                        className="btn-group compare-switch",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[{"label": i, "value": i} for i in ['Percentiles', 'Domain scores']],
                                        value='Percentiles',
                                        style={'margin': '10px 0 0 20px'},
                                        ),
                                    html.Div(id="data_report_plot6"),
                                    ],
                                    style={'backgroundColor': '#4d4d4d'},
                                    ),
                            
                                ],
                            style={
                                'width': '100%',
                                #'display': 'inline-block',
                                #'border-radius': '15px',
                                #'box-shadow': '1px 1px 1px grey',
                                'background-color': '#303030',  # dark theme
                                #'padding': '10px',
                                #'margin-bottom': '10px',
                                },
                            ),
                        html.Br(), 
                        ],
                        style={
                            'background-color': '#303030',  # dark theme
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
                                html.H5("Examine individual measures", style={'color': 'white'}),
                            
                                comparison_switch("compare-select3", style={'display': 'inline-block'}),
                            
                                # Year, domain and score type in one row, vertically centered,
                                # with a fixed gap between them.
                                html.Div([
                                    dcc.Dropdown(
                                        id='year-select3',
                                        clearable=False,
                                        className='dark-dropdown',
                                        options=[{"label": year_label(i), "value": i} for i in list(range(2021, latest_yr+1))],
                                        value=latest_yr,
                                        placeholder='Select a Stars year',
                                        optionHeight=50,
                                        style={'width': '170px', 'font-size': 16},  # room for '(predicted)'
                                        ),
                                    dcc.Dropdown(
                                        id='domain-select1',
                                        clearable=False,
                                        className='dark-dropdown',
                                        options=[{"label": i, "value": i} for i in ['Patient Experience', 'Readmission', 'Mortality', 'Safety of Care', 'Timely and Effective Care']],
                                        value='Patient Experience',
                                        placeholder='Select a domain',
                                        optionHeight=50,
                                        style={'width': '250px', 'font-size': 16},
                                        ),
                                    dcc.Dropdown(
                                        id='score-type1',
                                        clearable=False,
                                        className='dark-dropdown',
                                        options=[{"label": i, "value": i} for i in ['Standardized scores', 'Raw scores']],
                                        value='Standardized scores',
                                        placeholder='Select a score type',
                                        optionHeight=50,
                                        style={'width': '250px', 'font-size': 16},
                                        ),
                                    ],
                                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px',
                                           'alignItems': 'center', 'margin': '1% 0'},
                                    ),
                                html.Div(id="data_report_plot3"),
                                html.Hr(),
                                html.B(id="text10", style={'fontSize':16, 'color': 'white'}),
                                html.P(id="text11", style={'fontSize':16, 'color': 'white'}),
                                html.Hr(),

                                # Main line plot, with its score-type buttons in the same dark container.
                                html.Div([
                                    html.H5("Time series",
                                           style={'color': 'white', 'display': 'block',
                                                  'padding': '15px 20px 0 20px'}),
                                    dbc.RadioItems(
                                        id="score-select1",
                                        className="btn-group compare-switch",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[{"label": i, "value": i} for i in ['Percentiles', 'Standardized scores', 'Raw scores']],
                                        value='Percentiles',
                                        style={'margin': '10px 0 0 20px'},
                                        ),
                                    html.Div(id="data_report_plot5"),
                                    ],
                                    style={'backgroundColor': '#4d4d4d'},
                                    ),

                                # ---------- Score and volume over time (one measure) ----------
                                html.Hr(),
                                html.Div([
                                    html.H5("Score and volume over time",
                                            style={'color': 'white', 'display': 'block',
                                                   'padding': '15px 20px 0 20px'}),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='trend-measure',
                                            clearable=False,
                                            className='dark-dropdown',
                                            options=[],
                                            placeholder='Select a measure',
                                            optionHeight=50,
                                            style={'width': '450px', 'font-size': 16},
                                            ),
                                        dbc.RadioItems(
                                            id='trend-score',
                                            className="btn-group compare-switch",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[{"label": i, "value": i} for i in
                                                     ['Percentiles', 'Standardized scores', 'Raw scores']],
                                            value='Percentiles',
                                            ),
                                        dcc.Dropdown(
                                            id='trend-x',
                                            options=[{'label': 'Predicted infections', 'value': 'pred'},
                                                     {'label': 'Exposure count', 'value': 'vol'}],
                                            value='pred',
                                            clearable=False,
                                            className='dark-dropdown',
                                            optionHeight=50,
                                            style={'display': 'none'},  # shown for infection measures only
                                            ),
                                        ],
                                        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px',
                                               'alignItems': 'center', 'padding': '10px 20px 0 20px'},
                                        ),
                                    html.Div(id='trend-plot'),
                                    html.P(id='trend-note',
                                           style={'fontSize': 14, 'color': 'white',
                                                  'padding': '0 20px 15px 20px', 'margin': 0}),
                                    ],
                                    style={'backgroundColor': '#4d4d4d'},
                                    ),

                                # ---------- Funnel plot: score vs. denominator ----------
                                html.Hr(),
                                # Header, controls, plot and note in one dark container. The
                                # dropdowns sit in one row (wrapping if the window is narrow),
                                # with a fixed gap so they cannot overlap.
                                html.Div([
                                html.H5("Score vs. volume",
                                       style={'color': 'white', 'display': 'block',
                                              'padding': '15px 20px 0 20px'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='funnel-measure',
                                        clearable=False,
                                        className='dark-dropdown',
                                        options=[],
                                        placeholder='Select a measure',
                                        optionHeight=50,
                                        style={'width': '450px', 'font-size': 16},
                                        ),
                                    dcc.Dropdown(
                                        id='funnel-year',
                                        clearable=False,
                                        className='dark-dropdown',
                                        options=[],
                                        placeholder='Select a year',
                                        optionHeight=50,
                                        style={'width': '170px', 'font-size': 16},  # room for '(predicted)'
                                        ),
                                    dcc.Dropdown(
                                        id='funnel-score',
                                        options=[{"label": i, "value": i} for i in
                                                 ['Percentiles', 'Standardized scores', 'Raw scores']],
                                        value='Percentiles',
                                        clearable=False,
                                        className='dark-dropdown',
                                        optionHeight=50,
                                        style={'width': '220px', 'font-size': 16},
                                        ),
                                    dcc.Dropdown(
                                        id='funnel-x',
                                        options=[{'label': 'Predicted infections', 'value': 'pred'},
                                                 {'label': 'Exposure count', 'value': 'vol'}],
                                        value='pred',
                                        clearable=False,
                                        className='dark-dropdown',
                                        optionHeight=50,
                                        style={'display': 'none'},  # shown for infection measures only
                                        ),
                                    ],
                                    style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px',
                                           'alignItems': 'center', 'padding': '10px 20px 0 20px'},
                                    ),
                                html.Div(id='funnel-plot'),
                                html.P(id='funnel-note',
                                       style={'fontSize': 14, 'color': 'white',
                                              'padding': '0 20px 15px 20px', 'margin': 0}),
                                    ],
                                    style={'backgroundColor': '#4d4d4d'},
                                    ),

                                ],
                            style={
                                'width': '100%',
                                'background-color': '#303030',  # dark theme
                                },
                            ),
                        html.Br(),
                        ],
                        style={
                            'background-color': '#303030',  # dark theme
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
                                    options=[{"label": year_label(i)+'  ', "value": i} for i in list(range(latest_release_yr, latest_yr+1))],
                                    value=latest_yr,
                                    #placeholder='Select a year',
                                    optionHeight=50,
                                    style={
                                        'min-width': '170px',   # room for the '(predicted)' label
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
                                    fixed_rows={'headers': True},   # column labels stay visible when scrolling
                                    style_table={
                                        'overflowY': 'auto',   # vertical scroll
                                        'overflowX': 'auto',   # horizontal scroll
                                        'maxHeight': '60vh',   # the table scrolls within this height
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

                # Improvement path: the path of least resistance to a target star rating
                dbc.Button("Improvement path",
                           id="open-leverage",
                            n_clicks=0,
                            className="custom-button",
                            style={
                                'font-size': 14,
                                'display': 'inline-block',
                                'vertical-align': 'top',
                                'margin-left': '3%',
                                'margin-top': '1%',
                                'width': '30%',
                                },
                           ),
                dbc.Modal(
                    [dbc.ModalBody([
                        html.Div(
                            children=[
                                html.H5(id="leverage-header"),
                                dcc.Markdown(
                                    "**Choose a target star rating and click FIND PATH.** The app finds a balanced set of "
                                    "measure improvements that would reach it: similar-sized moves spread across measures, "
                                    "starting with those that help the score most and that are supported by evidence.\n"
                                    "* **Size of a move:** the larger of its percentile gain and its change in standard "
                                    "deviations (1 SD = 34 percentile points), so moves into the extremes count as large. Each "
                                    "move is at least 5 points and at most 33, or up to 50 where the hospital has held that "
                                    "percentile before.\n"
                                    "* The path uses as few measures as it can while keeping moves to about 20 points, and at "
                                    "most 10 measures unless more are needed.\n"
                                    "* **Evidence (realistic room)** comes from three references, each shown in this year's "
                                    "values: the hospital's best standing (percentile) in any year so far; what similar "
                                    "hospitals (same type, ownership and size) typically achieve; and the standing that the "
                                    "better-performing 10% of hospitals starting near the same level reached one year later.\n"
                                    "* Every target is a value hospitals actually reported this year. Click CONFIRM WITH THE "
                                    "WHAT-IF ENGINE to recompute the star rating with the full methodology (this takes a few seconds)."),
                                html.Hr(),
                                html.B(id="leverage-current", style={'fontSize': 16}),
                                html.Br(),
                                html.Br(),
                                dcc.Dropdown(
                                    id="year-select-lev",
                                    options=[{"label": year_label(i), "value": i} for i in list(range(latest_release_yr, latest_yr+1))],
                                    value=latest_yr,
                                    clearable=False,
                                    style={'min-width': '170px', 'width': '7vw', 'display': 'inline-block',
                                           'verticalAlign': 'top'},
                                ),
                                dcc.Dropdown(
                                    id="target-star-lev",
                                    options=[],
                                    placeholder='Target star rating',
                                    clearable=False,
                                    style={'min-width': '200px', 'display': 'inline-block',
                                           'verticalAlign': 'top', 'margin-left': '2%'},
                                ),
                                dbc.Button("Find path",
                                           id="leverage-run",
                                           n_clicks=0,
                                           className="custom-button",
                                           style={'display': 'inline-block', 'margin-left': '2%',
                                                  'verticalAlign': 'top'},
                                           ),
                                html.Br(),
                                html.Br(),
                                dcc.Loading(
                                    id="loading-leverage",
                                    type="default",
                                    children=[
                                        html.Div(id="leverage-summary", style={'fontSize': 15, 'margin-bottom': '10px'}),
                                        dash_table.DataTable(
                                            id="leverage-table",
                                            page_action='none',
                                            fixed_rows={'headers': True},
                                            style_table={'overflowY': 'auto', 'overflowX': 'auto',
                                                         'maxHeight': '60vh', 'minWidth': '100%'},
                                            style_cell={'padding': '10px', 'whiteSpace': 'normal',
                                                        'textAlign': 'center', 'minWidth': '100px',
                                                        'width': '140px', 'maxWidth': '300px'},
                                        ),
                                    ],
                                ),
                                html.Br(),
                                dbc.Button("Confirm with the what-if engine",
                                           id="leverage-confirm",
                                           n_clicks=0,
                                           className="custom-button",
                                           style={'display': 'inline-block'},
                                           ),
                                dcc.Loading(
                                    id="loading-leverage-confirm",
                                    type="default",
                                    children=[html.Div(id="leverage-confirm-text",
                                                       style={'fontSize': 15, 'margin-top': '10px'})],
                                ),
                                dcc.Store(id="leverage-path-store"),
                            ],
                            style={'width': '100%', 'background-color': '#f0f0f0', 'padding': '10px'},
                        ),
                        html.Br(),
                        ],
                        ),
                    dbc.ModalFooter(
                            dbc.Button("Close",
                                       id="close-leverage",
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
                    id="modal-leverage",
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
