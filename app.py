"""Entry point for the CMS Overall Hospital Quality Star Ratings app.

The source is split across these modules:
    app_data.py       loads the data and derives the lists used by the controls
    app_layout.py     builds the page layout
    app_callbacks.py  defines the callbacks (the functions that update the page)

Heroku serves `server` (see Procfile: web: gunicorn app:server).
"""

import dash
import pandas as pd
#import plotly
import warnings

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# All stylesheets are served from assets/, which Dash loads in alphabetical order:
# 00_bootstrap.min.css, then 01_codepen_bWLwgP.css, then base.css and the rest.
# No stylesheets come from outside sites.
external_stylesheets=[]
app = dash.Dash(__name__, 
                external_stylesheets = external_stylesheets,
                )

app.config.suppress_callback_exceptions = True
server = app.server

# These imports come after the settings above so that the warning filter is in
# place before the data are loaded (app_layout imports app_data).
from app_layout import create_layout
import app_callbacks  # noqa: F401  (importing this registers the callbacks)

app.layout = create_layout(app)


# Run the server
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True) # modified to run on linux server
