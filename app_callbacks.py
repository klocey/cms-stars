"""Callbacks for the app (the functions that update the page when a user
interacts with it). They are registered with Dash through the dash.callback
decorator, so importing this module is enough to register them."""

from dash import callback, ctx
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash.exceptions import PreventUpdate

import numpy as np
from scipy.stats import percentileofscore

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
#import sys
import re
#import csv
#import math
#import random
#from datetime import datetime
import textwrap

import what_if_fxn
import features
import leverage

leverage.warm_up()   # compute the engine baselines for the improvement path in the background

from app_data import (
    main_df, beds_max, what_if_df_2026, what_if_df_2027, HOSPITALS, beds, states, htypes, ctypes, latest_yr, latest_release_yr,
    year_label,
)


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


##############################   Helper functions   #####################################

def missing_years_message(hospital, yr1, yr2, yrs):
    """Return a message when the hospital has no data for the chosen year (yr1) or,
    after 2021, for the earlier year (yr2) that delta values are computed against.
    Return None when both years are available."""
    if yr1 not in yrs:
        return hospital + " has no data for " + str(yr1) + ". Try selecting another year."
    if yr1 != 2021 and yr2 not in yrs:
        missing = " or ".join(str(y) for y in sorted({yr1 - 1, yr2}, reverse=True))
        return (hospital + " has no data for " + missing + ", so delta values cannot be "
                "computed for " + str(yr1) + ". Try selecting another year.")
    return None


def facility_id(hospital):
    """Return the Facility ID from a 'NAME (ID)' label, e.g. '140119' or '01014F'."""
    m = re.search(r'\(([^()]*)\)\s*$', str(hospital))
    return m.group(1).strip() if m else None


def prior_release_year(yr1, yrs):
    """Return the release year that yr1 is compared against when computing deltas.

    This is the year before yr1 if the hospital has data for it (yrs), and
    otherwise the year before that. 2022 is always compared against 2021, and
    2021 (the first year in the app) is paired with itself.
    """
    if yr1 <= 2022:
        return 2021
    if yr1 - 1 in yrs:
        return yr1 - 1
    return yr1 - 2


##############################   Callbacks   ############################################
#########################################################################################


# Help pop-ups (modals). Each one opens with its "open-..." button and closes
# with its "close-..." button. Columns: modal id, open-button id,
# close-button id, and whether to skip the call when the page first loads.
MODALS = [
    ("modal-choose_this",     "open-this",            "close-this",            True),
    ("modal-choose_that",     "open-that",            "close-that",            True),
    ("modal-choose_theother", "open-theother",        "close-theother",        True),
    ("modal-choose_what-if",  "open-what-if",         "close-what-if",         True),
    ("modal-centered1",       "open-centered1",       "close-centered1",       False),
    ("modal-what-if-results", "whatif_button",        "close-what-if-results", False),
    ("modal-centered3",       "open-centered3",       "close-centered3",       False),
    ("modal-centered4",       "open-centered4",       "close-centered4",       False),
    ("modal-leverage",        "open-leverage",        "close-leverage",        True),
]


def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


for modal_id, open_id, close_id, prevent_initial in MODALS:
    callback(
        Output(modal_id, "is_open"),
        [Input(open_id, "n_clicks"), Input(close_id, "n_clicks")],
        [State(modal_id, "is_open")],
        prevent_initial_call=prevent_initial,
    )(toggle_modal)




@callback([Output("header-text", 'children'),
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
        
  
    
@callback(Output("map-header", 'children'),
              [Input("option_hospitals", 'children'),
              ],
            )
def update_map_header(filtered_hospitals):
    
    if filtered_hospitals is None:
        raise PreventUpdate
    
    else:
        return "Your filters produced " + str(len(filtered_hospitals)) + " hospitals. Not all hospitals will have data or stars results for each year."



def _describe_selection(values, full, max_listed=4):
    """Short description of a multi-select filter: 'all', 'none', the values, or 'k of n'."""
    full = set(full)
    values = list(values or [])
    if not values:
        return 'none'
    if full.issubset(values):
        return 'all'
    if len(values) <= max_listed:
        return ', '.join(sorted(str(v) for v in values))
    return str(len(values)) + ' of ' + str(len(full))


@callback(
    Output("active-filters", "children"),
    Output("active-filters-box", "style"),
    Input("option_hospitals", "children"),
    Input("compare-select1", "value"),
    State("beds1", "value"),
    State("states-select1", "value"),
    State("hospital_type1", "value"),
    State("control_type1", "value"),
)
def update_active_filters(filtered_hospitals, compare, bed_range, states_vals, htype_vals, ctype_vals):
    """One line describing the filters that define the 'Filtered Hospitals' comparison set.
    It is shown only while the main switch is on 'Compare to Filtered Hospitals'."""
    if filtered_hospitals is None:
        raise PreventUpdate
    box_style = {'display': 'block'} if compare == 'Selected hospitals' else {'display': 'none'}
    low, high = int(bed_range[0]), int(bed_range[1])
    beds_txt = 'all' if (low <= 0 and high >= 2700) else '{:,}\u2013{:,}'.format(low, high)
    parts = [
        'Filtered hospitals: {:,}'.format(len(filtered_hospitals)),
        'States: ' + _describe_selection(states_vals, states),
        'Hospital types: ' + _describe_selection(htype_vals, htypes),
        'Ownership: ' + _describe_selection(ctype_vals, ctypes),
        'Beds: ' + beds_txt,
    ]
    return '  \u00b7  '.join(parts), box_style

        

@callback([Output('year-select1', 'options'),
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
        
        options = [{"label": year_label(i)+'  ', "value": i} for i in yrs]
        return options, yrs[0], options, yrs[0], options, yrs[0]


@callback( # Updated number of beds text
    Output('Filterbeds1', 'children'),
    [
     Input('beds1', 'value'),
     ],
    )
def update_output1(value):
    
    v1 = int(value[0])
    v2 = int(value[1])
    if v2 >= 2700:   # the top of the slider includes every hospital
        return 'Number of beds: [{}, Max]'.format(v1)
    value = [v1, v2]
    
    return 'Number of beds: {}'.format(value)

    
@callback(
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



@callback(
    Output("data-download", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_download(n_clicks):
    return dcc.send_data_frame(main_df.to_csv, "data.csv")
    

@callback(
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




@callback(
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
    
    

@callback(
     Output("figure1", "figure"),
     [Input("hospital-select1b", "value"),
      Input('year-select1', 'value'),
      Input("compare-select1", "value"),
      Input("option_hospitals", 'children'),
      ],
    )
def update_figure1(hospital, yr, compare, filtered_hospitals):    
    
    
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
        
    
    #if 'all_hosps_btn1' in triggered:
    #    tdf_main = tdf_main[~tdf_main['summary_score'].isin([np.nan, float("NaN")])]
    if compare == 'Selected hospitals':
        tdf_main = tdf_main[tdf_main['Name and Num'].isin(filtered_hospitals+[hospital])]
        tdf_main = tdf_main[~tdf_main['summary_score'].isin([np.nan, float("NaN")])]
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



@callback(
     [Output("data_report_plot2", "children"),
      Output("text3", "children"),
      ],
     [Input("hospital-select1b", "value"), 
      Input("option_hospitals", 'children'),
      Input("compare-select2", "value"),
      Input("year-select2", "value"),
      ],
    )
def update_domains_table(hospital, option_hospitals, compare, yr):    
    
    set_select = 'Selected hospitals' if compare == 'Selected hospitals' else 'Measures group'
    
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
        yr2 = prior_release_year(yr1, yrs)
            
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
            return dashT, missing_years_message(hospital, yr1, yr2, yrs)
        
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
        
        yr2 = prior_release_year(yr1, yrs)
        
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
            return dashT, missing_years_message(hospital, yr1, yr2, yrs)


    
    


@callback(
     [Output("data_report_plot3", "children"),
      Output("text10", "children"),
      ],
     [Input("hospital-select1b", "value"),
      Input("option_hospitals", 'children'),
      Input("compare-select3", "value"),
      Input('domain-select1', "value"),
      Input('score-type1', 'value'),
      Input('year-select3', "value"),
      ],
    )
def update_panel4(hospital, option_hospitals, compare, domain, score_type, yr):    
    
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
    yr2 = prior_release_year(yr1, yrs)

    # The hospital has no data for the chosen year, or for the earlier year that
    # delta values are computed against (for example, a gap in its release years)
    txt = missing_years_message(hospital, yr1, yr2, yrs)
    if txt:
        return dashT, txt
    
    
    set_select = 'Selected hospitals' if compare == 'Selected hospitals' else 'Measures group'
        
        
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


@callback(
    Output("data_report_plot5", "children"),
    [
        Input("hospital-select1b", "value"),
        Input("compare-select3", "value"),
        Input("domain-select1", "value"),
        Input("score-select1", "value"),
    ],
)
def update_measure_timeseries(hospital, compare, domain, score_type):

    if hospital is None or domain is None or score_type is None:
        return None

    # ---------- COMPARISON SET (from the switch) ----------
    set_select = 'Selected hospitals' if compare == 'Selected hospitals' else 'Measures group'
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


@callback(
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


@callback(
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
            'H_COMP_3_LINEAR_SCORE', 
            'H_COMP_5_LINEAR_SCORE', 
            'H_COMP_6_LINEAR_SCORE', 
            #'H_COMP_7_LINEAR_SCORE', 
            'H_COMP_8_LINEAR_SCORE', 
            'H_COMP_9_LINEAR_SCORE', 
            'H_SYMP_LINEAR_SCORE', 
            'H_HSP_RATING_LINEAR_SCORE', 
            #'H_QUIET_LINEAR_SCORE',   # dropped by the engine for 2027
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
    if yr == 2027:   # dropped by the engine (what_if_fxn.py) for 2027
        m3.remove('PSI_4_SURG_COMP')
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
    
    pnum = facility_id(hospital)
    
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
    
    # --- The hospital's all-time best value for each measure, across all release
    # years in the data: the highest value if higher is better, otherwise the lowest ---
    hosp_all_yrs = main_df.loc[main_df['Facility ID'] == pnum, measures]
    best_vals = []
    for m, hb in zip(measures, higher_better):
        v = hosp_all_yrs[m].max() if hb == 'Yes' else hosp_all_yrs[m].min()
        best_vals.append(None if pd.isna(v) else round(float(v), 6))
    
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
        'Min value': mins,
        'Max value': maxs,
        'Actual value': actual_vals,
        'What-if value': actual_vals,
        'All-time best': best_vals,
    })
    
    data = df_table.to_dict('records')
    
    columns = [
        {'id': c, 'name': c, 'editable': (c == 'What-if value')}
        for c in df_table.columns
    ]
    
    return data, columns, df_table.to_json(), ''

    



@callback(
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
    
    # Nothing to analyze: the what-if table is empty because the hospital has no
    # data in the release used for this year's what-if analysis
    if not data or not columns or 'What-if value' not in [col['id'] for col in columns]:
        txt = (str(hospital) + " has no data in the CMS Care Compare release that is used "
               "to make predictions, so there are no what-if results to show.")
        return txt, '', '', '', ''
    
    column_names = [col['id'] for col in columns]
    df = pd.DataFrame(data, columns=column_names)
    
    if df is None:
        return ["Yikes! There's no data!"]
    
    df['What-if value'] = df['What-if value'].astype(float)
    
    pnum = facility_id(hospital)
    
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


##############################   Improvement path   #####################################

@callback(
    Output("leverage-header", "children"),
    Output("leverage-current", "children"),
    Output("target-star-lev", "options"),
    Output("target-star-lev", "value"),
    Input("hospital-select1b", "value"),
    Input("year-select-lev", "value"),
)
def update_leverage_targets(hospital, yr):
    """Header, current rating, and the target ratings above it."""
    if hospital is None or yr is None:
        raise PreventUpdate
    header = "Improvement path: " + hospital
    pnum = facility_id(hospital)
    rec = main_df[(main_df['Release year'] == yr) & (main_df['Facility ID'] == pnum)]
    star = rec['star'].iloc[0] if not rec.empty else np.nan
    if pd.isna(star):
        return header, hospital + " has no star rating for " + year_label(yr) + ".", [], None
    star = int(star)
    if star >= 5:
        return header, year_label(yr) + " star rating: 5. There is no higher rating to aim for.", [], None
    options = [{"label": str(s) + " stars", "value": s} for s in range(star + 1, 6)]
    return header, year_label(yr) + " star rating: " + str(star), options, star + 1


@callback(
    Output("leverage-summary", "children"),
    Output("leverage-table", "data"),
    Output("leverage-table", "columns"),
    Output("leverage-path-store", "data"),
    Output("leverage-confirm-text", "children"),
    Input("leverage-run", "n_clicks"),
    Input("hospital-select1b", "value"),
    Input("year-select-lev", "value"),
    Input("target-star-lev", "value"),
    prevent_initial_call=True,
)
def update_leverage_path(n_clicks, hospital, yr, target):
    """Find the path when FIND PATH is clicked; a new hospital, year or target clears the old one."""
    if ctx.triggered_id != "leverage-run":
        return '', [], [], None, ''
    if not hospital or yr is None or target is None:
        return 'Choose a target star rating first.', [], [], None, ''
    if not facility_id(hospital):
        return 'The improvement path is not available for ' + hospital + '.', [], [], None, ''
    data, columns, _, msg = update_whatif_table(hospital, None, yr)
    if msg:
        return msg, [], [], None, ''
    pnum = facility_id(hospital)
    res = leverage.find_path(hospital, pnum, yr, target, data)
    if not res['ok']:
        return res['message'], [], [], None, ''
    rows = res['rows']
    summary = ("To reach " + str(res['reached']) + " stars for " + year_label(yr) + ", this path changes " +
               str(len(rows)) + (" measure" if len(rows) == 1 else " measures") +
               " (largest move: " + str(res['max_gain']) + " points). " + res['note'])
    columns = [{'id': c, 'name': c} for c in rows[0].keys()]
    store = {'hospital': hospital, 'yr': yr, 'target': target, 'targets': res['targets']}
    return summary, rows, columns, store, ''


@callback(
    Output("leverage-confirm-text", "children", allow_duplicate=True),
    Input("leverage-confirm", "n_clicks"),
    State("leverage-path-store", "data"),
    State("option_hospitals", "children"),
    prevent_initial_call=True,
)
def confirm_leverage_path(n_clicks, store, filtered_hospitals):
    """Recompute the star rating with the full what-if engine, using the path's target values."""
    if not store:
        return 'Find a path first.'
    data, columns, _, msg = update_whatif_table(store['hospital'], None, store['yr'])
    if msg:
        return msg
    for row in data:
        if row['Measure-ID'] in store['targets']:
            row['What-if value'] = store['targets'][row['Measure-ID']]
    res = update_whatif_analysis(n_clicks, data, columns, store['hospital'], filtered_hospitals or [], store['yr'])
    star_txt, score_txt = str(res[0]), str(res[1])
    found = re.match(r'\s*([0-9.]+)\s*Star', star_txt)
    verdict = ''
    if found:
        verdict = (" The target of " + str(store['target']) + " stars is " +
                   ("reached." if float(found.group(1)) >= store['target'] else "not reached."))
    return "What-if engine result: " + star_txt + ", " + score_txt + "." + verdict
