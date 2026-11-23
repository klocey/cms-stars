# cms-stars

Python source code for an open-source [Plotly Dash](https://dash.plotly.com/) application for examining the Centers for Medicare & Medicaid Services (CMS) Overall Hospital Quality Star Ratings. The app supports retrospective analysis of past ratings (2021 onward), prospective estimates of the upcoming release, and what-if analyses for individual hospitals.

**Live app:** https://cms-stars-app-2bb02f1a5ac2.herokuapp.com/

## Publication

Locey, K.J., Stein, B.D., Schipfer, R., Dotson, B., & Klemp, L. (2024). An open-source application for obtaining retrospective and prospective insights into overall hospital quality star ratings. *Healthcare Analytics*, 6, 100371. https://doi.org/10.1016/j.health.2024.100371

The paper describes the app as of the 2024 release. The code here has been updated continually since then for new CMS releases, methodology changes, and new features, so details may differ from the publication.

## What the app does

- **Choose a hospital and a comparison set.** Filter peer hospitals by hospital type, ownership type, US state or territory, and number of beds. A map shows the filtered hospitals.
- **Summary view.** Star rating, summary score, and peer group for the chosen hospital and release year.
- **Scores across domains.** Compare the hospital's domain scores (Mortality, Safety of Care, Readmission, Patient Experience, Timely and Effective Care) against the filtered hospitals or against the hospitals in its CMS peer group.
- **Examine individual measures.** Measure-level standardized scores, raw scores, and percentiles, including time series across release years.
- **Prospective estimates.** The most recent year in the data is an estimate of the upcoming release, made by applying the CMS methodology to data that have already been published.
- **What-if analysis.** Edit a hospital's measure values and recalculate its rating for the latest release year or the upcoming one.
- **Download data.** Export the compiled dataset as a CSV file.

### How the star ratings are calculated (brief)

Each measure is standardized as a z-score (the number of standard deviations from the national mean), with its sign set so that higher values are better. Measures are averaged within each of the five domains, and the domain averages are standardized again. A weighted summary score is then calculated: Timely and Effective Care is weighted 0.12 and each of the other four domains 0.22, with the weights redistributed when a hospital lacks a domain. Hospitals that meet the reporting thresholds are placed in peer groups by the number of domains they report (3, 4, or 5), and k-means clustering on the summary scores within each peer group assigns 1 to 5 stars.

## How to run the app locally

1. Clone or download this repository and open a terminal in its root folder.

2. Create and activate a virtual environment (Python 3.12 is used in deployment):

   macOS or Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the app:
   ```
   python app.py
   ```

   The terminal will show something like:
   ```
   Dash is running on http://0.0.0.0:8050/
   ```

5. Open that address (or http://localhost:8050/) in your web browser.

## Requirements

Listed in `requirements.txt` and installed in step 3 above:

* werkzeug==3.0.6
* dash==3.0.1
* gunicorn==23.0.0
* numpy==2.2.4
* pandas==2.2.3
* scipy==1.15.2
* Flask==3.0.3
* plotly==6.0.1
* DateTime==5.5
* dash_bootstrap_components==2.0.0
* lxml==5.3.1
* scikit-learn==1.6.1

## Files and directories

<details><summary>app.py</summary>

The entry point. It creates the Dash app and the `server` object that Heroku runs (see `Procfile`), builds the page from `app_layout.py`, registers the callbacks by importing `app_callbacks.py`, and starts the app when run with `python app.py`.
</details>

<details><summary>app_data.py</summary>

Loads the compiled dataset and the what-if input data from `dataframe_data/`, and derives the lists used by the app's controls (hospital types, ownership types, states, and the latest release years). The compiled dataset loaded at startup is set by the `month` and `year` variables in this file.
</details>

<details><summary>app_layout.py</summary>

Builds the page layout: the sidebar controls (hospital choice and filters), the main panels, and the help pop-ups. `create_layout(app)` returns the layout.
</details>

<details><summary>app_callbacks.py</summary>

Defines the callbacks, which are the functions that update the page when a user interacts with it, along with their helper functions. Importing the module registers the callbacks with Dash.
</details>

<details><summary>what_if_fxn.py</summary>

Contains `run_whatif(raw_data, pnum, yr)`, a Python translation of the CMS SAS code used for the what-if analyses. It selects the measures used in a given release year, standardizes measures and domains, applies the domain weights, assigns peer groups, and assigns stars by k-means clustering. Measure sets are defined per release year and need to be updated when CMS changes the methodology.
</details>

<details><summary>features.py</summary>

Defines `feature_dict`, which groups the dataset's columns (filter categories, date fields, standardized scores, domain weights, measure counts, and the measures and display labels for each domain) for use by the app.
</details>

<details><summary>dataframe_data/</summary>

Pickled pandas DataFrames used by the app:

- `hosp_stars_dat_from_<Month>_<Year>.pkl`: the compiled hospital-level dataset across release years, including the prospective estimates for the upcoming year. The suffix records when the dataset was compiled.
- `<year>_data_for_whatifs.pkl`: measure-level input data for the what-if analyses, one file for each year offered in the what-if tool.
</details>

<details><summary>assets/</summary>

Files used to style the interface. All except `RUSH_full_color.jpg` and `custom_styles.css` were obtained from another open-source Plotly Dash app (https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-clinical-analytics/assets).

- `Acumin-BdPro.otf`: an OpenType font file.
- `base.css`, `clinical-analytics.css`, `custom_styles.css`: cascading style sheets (CSS), which set the appearance of the page.
- `resizing.js`: a small JavaScript file used for resizing.
- `plotly_logo.png`, `RUSH_full_color.jpg`: logo images.
</details>

<details><summary>Procfile</summary>

Required for deployment on Heroku. It tells Heroku to serve the app with the gunicorn web server: `web: gunicorn app:server`
</details>

<details><summary>requirements.txt</summary>

The Python libraries needed to run the app, used both by Heroku and by `pip install -r requirements.txt` locally.
</details>

<details><summary>runtime.txt</summary>

Tells Heroku which Python version to use: `python-3.12`
</details>

## License

MIT License. See `LICENSE`.

## Developer

Kenneth J. Locey, PhD. Senior Clinical Data Scientist. Center for Quality, Safety & Value Analytics. Rush University Medical Center.
