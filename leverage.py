"""Improvement path: a balanced, prioritized path to a target star rating.

For a chosen hospital, release year and target star rating, find a set of measure
improvements that would reach the target, the way a quality leader could use it:
similar-sized moves spread across measures, favoring measures that help the summary
score most, and supported by evidence.

How it works
- Leverage: within a domain, every measure counts equally. One standard deviation
  (SD) of improvement on a measure raises the summary score by
      domain weight / (number of measures the hospital reports in the domain x domain SD).
- Size of a move: the larger of its percentile gain and its change in SDs x 34.1
  (1 SD = 34.1 percentile points at the middle of a bell curve), so moves into the
  tails, where a few percentile points hide large changes, count as large.
- Limits: each move is at least 5 percentile points and at most 33 in size, or up to
  50 where the hospital has held that percentile before. Targets are values hospitals
  actually reported that year, so they are on the reporting grid and possible.
- Balance with priority: the path minimizes the largest move, using as few measures as
  keep every move within about 20 points (at most 10 measures, widening to 15, 20, ...
  only if needed); measures that help the summary score most are picked first.
- Evidence (realistic room) comes from three references, each worked out in
  percentiles within each year and expressed in the chosen year's values: the
  hospital's best standing in any year so far; what similar hospitals (same type,
  ownership and size) typically achieve (median regression); and the standing the
  better-performing 10% of hospitals starting near the same level reached one year
  later. Paths use only supported moves when possible.
- Star cutoffs and the Safety of Care bottom-quarter line come from the what-if
  engine's own results for the unchanged data (cached per year), since paths are
  confirmed with the engine. If the target can't be reached within the limits, the
  closest reachable rating is shown.

Self-checks: the hospital's summary score is rebuilt from its measures; if it does
not match the stored score (for example because the domain measure lists below no
longer match what_if_fxn.py), no path is shown. Every value shown is checked against
the range observed among hospitals that year.
"""

import hashlib
import threading
from functools import lru_cache

import numpy as np
import pandas as pd

from app_data import main_df, what_if_df_2026, what_if_df_2027

WHATIF_DATA = {2026: what_if_df_2026, 2027: what_if_df_2027}

# Tuning constants
MIN_PCT_GAIN = 5.0   # each change moves the hospital at least 5 percentile points
SOFT_MAX_GAIN = 33.0 # no measure moves more than 33 points without evidence
HARD_MAX_GAIN = 50.0 # ... and never more than 50 points (evidence: the hospital has held that percentile before)
MODERATE_GAIN = 20.0 # use the fewest measures that keep every move within about 20 points
SD_POINTS = 34.1     # size of a move: the larger of its percentile gain and its size in SDs x 34.1
                     # (1 SD = 34.1 percentile points at the middle of a bell curve), so that moves in
                     # the tails, where few percentile points hide large changes, count as large
MAX_MEASURES = (10, 15, 20)   # widen the list only when the target can't be reached otherwise
NEAR = 0.10          # precedent: hospitals starting within 10 percentile points
PREC_Q = 0.90        # precedent: the better-performing 10% of those hospitals
MARGIN = 0.005       # aim this far past the star cutoff (summary-score units)
AIM = 'upper'        # cutoff = lowest summary score of the target star group ('mid' = middle of the gap)
SAFETY_MARGIN = 0.01 # aim this far above the Safety of Care bottom-quarter line

# Domain names (as in the what-if table) -> stored columns: (domain score, domain weight).
# The domain means and SDs are computed from the measures, as the engine does
# (the stored *_mean / *_sd columns are not filled for every year).
DOMAIN_COLS = {
    'Mortality': ('Std_Outcomes_Mortality_score', 'weight_Outcomes_Mortality'),
    'Safety of Care': ('Std_Outcomes_Safety_score', 'weight_Outcomes_Safety'),
    'Readmission': ('Std_Outcomes_Readmission_score', 'weight_Outcomes_Readmission'),
    'Patient Experience': ('Std_PatientExp_score', 'weight_PatientExperience'),
    'Timely & Effective Care': ('Std_Process_score', 'weight_Process'),
}


def engine_domains(yr):
    """Domain measure lists used by what_if_fxn.run_whatif for release year yr.
    These must match what_if_fxn.py; the self-check in find_path detects a mismatch."""
    return {
        'Mortality': ['MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF', 'MORT_30_PN', 'MORT_30_STK']
                     + ([] if yr == 2027 else ['PSI_4_SURG_COMP']) + ['Hybrid_HWM'],
        'Readmission': ['EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32', 'READM_30_CABG', 'READM_30_COPD',
                        'READM_30_HIP_KNEE', 'Hybrid_HWR', 'OP_35_ADM', 'OP_35_ED', 'OP_36'],
        'Safety of Care': ['COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 'HAI_6', 'PSI_90_SAFETY'],
        'Patient Experience': ['O-COMP-1', 'O-COMP-2', 'O-COMP-3', 'O-PATIENT-RATE', 'O-PATIENT-REC',
                               'H_COMP_1_LINEAR_SCORE', 'H_COMP_2_LINEAR_SCORE', 'H_COMP_3_LINEAR_SCORE',
                               'H_COMP_5_LINEAR_SCORE', 'H_COMP_6_LINEAR_SCORE', 'H_CLEAN_LINEAR_SCORE',
                               'H_RECMND_LINEAR_SCORE', 'H_HSP_RATING_LINEAR_SCORE']
                              + (['H_COMP_8_LINEAR_SCORE', 'H_COMP_9_LINEAR_SCORE', 'H_SYMP_LINEAR_SCORE'] if yr == 2027
                                 else ['H_COMP_7_LINEAR_SCORE', 'H_QUIET_LINEAR_SCORE']),
        'Timely & Effective Care': ['IMM_3', 'OP_10', 'OP_18B', 'SAFE_USE_OF_OPIOIDS', 'OP_22', 'OP_23', 'OP_29', 'SEP_1']
                                   + ([] if yr == 2027 else ['OP_13', 'OP_8']),
    }


def safety_penalty(yr, star):
    """True if the Safety of Care rule (bottom quarter) lowers this star in release year yr,
    as applied in what_if_fxn.run_whatif."""
    if yr == 2027:
        return star >= 2
    if yr == 2026:
        return star == 5
    return False


# ----------------------------------------------------------------------------
# Year-level quantities (the same for every hospital); cached per year
# ----------------------------------------------------------------------------
def _design(attrs):
    """Predictors: hospital type and ownership (indicator columns) and log(beds, capped at 2700)."""
    Xd = pd.get_dummies(attrs[['Hospital Type', 'Hospital Ownership']].fillna('Unknown'), drop_first=True).astype(float)
    beds = attrs['Beds'].clip(upper=2700)
    Xd['logbeds'] = np.log1p(beds.fillna(beds.median())).values
    return Xd


def _median_fit(X1, y, iters=60, eps=1e-6):
    """Median (least absolute deviation) regression by iteratively reweighted least squares.
    Solves a small p-by-p system at each step; lstsq tolerates redundant predictors."""
    beta = np.linalg.lstsq(X1.T @ X1, X1.T @ y, rcond=None)[0]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - X1 @ beta), eps)
        Xw = X1 * w[:, None]
        new = np.linalg.lstsq(X1.T @ Xw, Xw.T @ y, rcond=None)[0]
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break
    return beta


def _decimals(values):
    """Number of decimal places the measure is reported with (0 to 4)."""
    for d in range(5):
        if np.allclose(values * 10 ** d, np.round(values * 10 ** d), atol=1e-6):
            return d
    return 4


@lru_cache(maxsize=8)
def _year_basis(yr, signs_key):
    """Everything about release year yr that does not depend on the hospital.
    signs_key: tuple of (measure, +1 if higher is better else -1)."""
    W = WHATIF_DATA[yr]
    doms = engine_domains(yr)
    sign = {m: 1 for d in doms for m in doms[d]}
    sign.update(dict(signs_key))
    dom_of = {m: d for d in doms for m in doms[d]}
    # the measures the engine keeps: reported by at least 101 hospitals
    meas = [m for d in doms for m in doms[d] if m in W.columns and W[m].notna().sum() >= 101]
    U = W[['PROVIDER_ID'] + meas].dropna(how='all', subset=meas)
    mu, sd = U[meas].mean(), U[meas].std(ddof=1)
    Z = pd.DataFrame({m: sign[m] * (U[m].values - mu[m]) / sd[m] for m in meas}, index=U['PROVIDER_ID'].values)
    # domain mean and SD of the hospitals' average z-scores (as in what_if_fxn: zscore of row means, ddof=1)
    dom_mean, dom_sd = {}, {}
    for d in DOMAIN_COLS:
        avg = Z[[m for m in meas if dom_of[m] == d]].mean(axis=1)
        dom_mean[d], dom_sd[d] = float(avg.mean()), float(avg.std(ddof=1))

    yrdf = main_df[main_df['Release year'] == yr]
    lo = {m: float(yrdf[m].min()) for m in meas if m in yrdf.columns and yrdf[m].notna().any()}
    hi = {m: float(yrdf[m].max()) for m in meas if m in yrdf.columns and yrdf[m].notna().any()}
    decimals = {m: _decimals(U[m].dropna().values) for m in meas}
    sortedZ = {m: np.sort(Z[m].dropna().values) for m in meas}

    # median regressions (one per measure); predictions are made per hospital later
    attrs = (yrdf.drop_duplicates('Facility ID').set_index('Facility ID')
             [['Hospital Type', 'Hospital Ownership', 'Beds']].reindex(Z.index))
    Xall = _design(attrs)
    reg = {}
    for m in meas:
        ok = Z[m].notna().values
        Xm = Xall.loc[ok]
        keep = Xm.columns[Xm.std() > 0]
        X1 = np.column_stack([np.ones(ok.sum()), Xm[keep].values])
        reg[m] = (list(keep), _median_fit(X1, Z[m].values[ok]))

    # Historic best and precedent are worked out in percentiles within each year, so that a shift
    # in a measure's overall level across years (or a change in its definition) cannot distort them.
    # Percentile = share of hospitals with a worse value that year (better direction).
    cols = ['Facility ID', 'Release year'] + [m for m in meas if m in main_df.columns]
    md = main_df[cols].drop_duplicates(['Facility ID', 'Release year'])
    pct_by_year, prec, cur_vals = {}, {}, {}
    for m in meas:
        if m not in md.columns:
            continue
        P = md.pivot(index='Facility ID', columns='Release year', values=m)
        P = P[[y for y in sorted(P.columns) if y <= yr]]
        O = P if sign[m] > 0 else -P
        Q = (O.rank(method='min') - 1) / O.notna().sum()
        pct_by_year[m] = Q
        starts, nexts = [], []
        for y0, y1 in zip(Q.columns[:-1], Q.columns[1:]):
            ok = Q[y0].notna() & Q[y1].notna()
            if ok.sum() < 50:
                continue
            starts.append(Q.loc[ok, y0].values)
            nexts.append(Q.loc[ok, y1].values)
        if starts:
            prec[m] = (np.concatenate(starts), np.concatenate(nexts))
        if yr in O.columns:   # this year's values, better direction, sorted (to turn a percentile into a value)
            cur_vals[m] = np.sort(O[yr].dropna().values)

    return dict(meas=meas, sign=sign, dom_of=dom_of, mu=mu, sd=sd, Z=Z, lo=lo, hi=hi, dom_mean=dom_mean, dom_sd=dom_sd,
                decimals=decimals, sortedZ=sortedZ, Xall=Xall, reg=reg, prec=prec,
                pct_by_year=pct_by_year, cur_vals=cur_vals)


BASELINE_FILE = 'dataframe_data/engine_baselines.pkl'   # made by stars-data-builder's generate_engine_baselines.py


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _load_saved_baselines():
    """Engine baselines saved by generate_engine_baselines.py, for each year whose what-if pickle
    and engine (what_if_fxn.py) are the same files the baselines were made from (same md5).
    Years that don't match, or a missing file, are left out and computed at startup as before."""
    try:
        saved = pd.read_pickle(BASELINE_FILE)
    except FileNotFoundError:
        return {}
    fp = saved['fingerprints']
    if fp.get('what_if_fxn.py') != _md5('what_if_fxn.py'):
        return {}
    return {yr: df for yr, df in saved['years'].items()
            if yr in WHATIF_DATA
            and fp.get(f'{yr}_data_for_whatifs.pkl') == _md5(f'dataframe_data/{yr}_data_for_whatifs.pkl')}


_saved_baselines = _load_saved_baselines()
_baseline_cache = {}
_baseline_locks = {y: threading.Lock() for y in WHATIF_DATA}


def _engine_baseline(yr):
    """The what-if engine's results for release year yr on the unchanged data (star cutoffs and the
    Safety of Care bottom-quarter line are taken from these, since paths are confirmed with the engine).
    Taken from the saved file when it matches the data; otherwise computed once per year, and a request
    that arrives while the background warm-up is computing the same year waits for it instead of
    computing it a second time."""
    with _baseline_locks[yr]:
        if yr not in _baseline_cache:
            out = _saved_baselines.get(yr)
            if out is None:
                import what_if_fxn
                out = what_if_fxn.run_whatif(WHATIF_DATA[yr].copy(), None, yr)
            out = out[['PROVIDER_ID', 'cnt_grp', 'summary_score', 'star', 'Std_Outcomes_Safety_score']].copy()
            out['grp'] = out['cnt_grp'].astype(str).str[0].astype(int)   # '3) # of groups=5' -> 3
            _baseline_cache[yr] = out
        return _baseline_cache[yr]


def warm_up():
    """Prepare the engine baselines so path requests don't wait. Years in the saved file are loaded at
    once; any others are computed in a background thread (predicted year first)."""
    years = sorted(WHATIF_DATA, reverse=True)
    for y in years:
        if y in _saved_baselines:
            _engine_baseline(y)
    todo = [y for y in years if y not in _saved_baselines]
    if todo:
        threading.Thread(target=lambda: [_engine_baseline(y) for y in todo], daemon=True).start()


# ----------------------------------------------------------------------------
# The path for one hospital
# ----------------------------------------------------------------------------
def _message(text):
    return {'ok': False, 'message': text, 'rows': [], 'targets': {}}


def find_path(hospital, pnum, yr, target, table_rows):
    """Path of least resistance for one hospital.
    table_rows: rows of the app's what-if table for this hospital and year (they supply
    the measures users can edit, their names, and whether higher is better)."""
    signs_key = tuple(sorted((r['Measure-ID'], 1 if r['Higher is better'] == 'Yes' else -1) for r in table_rows))
    B = _year_basis(yr, signs_key)
    meas, sign, dom_of, mu, sd, Z = B['meas'], B['sign'], B['dom_of'], B['mu'], B['sd'], B['Z']
    lo, hi = B['lo'], B['hi']
    names = {r['Measure-ID']: r['Measure Name'] for r in table_rows}

    yrdf = main_df[main_df['Release year'] == yr]
    hrow = yrdf[yrdf['Facility ID'] == pnum]
    if hrow.empty or pnum not in Z.index:
        return _message(hospital + ' has no data for ' + str(yr) + '.')
    H = hrow.iloc[0]
    if pd.isna(H['star']):
        return _message(hospital + ' did not receive a star rating for ' + str(yr) + '.')
    current = int(H['star'])
    if target <= current:
        return _message('Choose a target above the current rating (' + str(current) + ' stars).')
    # Paths are confirmed with the what-if engine, so start from the engine's own rating of the unchanged
    # data. It can differ from the stored rating shown on the main page (for example, the stored 2027
    # predictions don't always reflect the Safety of Care rule that the engine applies).
    erow = _engine_baseline(yr)
    erow = erow[erow['PROVIDER_ID'] == pnum]
    engine_star = int(erow['star'].iloc[0]) if not erow.empty and pd.notna(erow['star'].iloc[0]) else current
    eng_safety = float(erow['Std_Outcomes_Safety_score'].iloc[0]) if not erow.empty else float(H['Std_Outcomes_Safety_score'])
    if target <= engine_star:
        return _message('On the unchanged data, the what-if engine already rates ' + hospital + ' at ' +
                        str(engine_star) + ' stars for ' + str(yr) + ' (the stored rating on the main page is ' +
                        str(current) + '). Choose a target above ' + str(engine_star) + ' stars.')
    zh = Z.loc[pnum]

    # self-check: rebuild the summary score from the measures
    dom_mean, dom_sd = B['dom_mean'], B['dom_sd']
    summ = 0.0
    for d, (sc, w) in DOMAIN_COLS.items():
        ms = [m for m in meas if dom_of[m] == d and pd.notna(zh[m])]
        if ms and pd.notna(H[sc]):
            summ += H[w] * (np.mean([zh[m] for m in ms]) - dom_mean[d]) / dom_sd[d]
    if not np.isclose(summ, H['summary_score'], atol=1e-6):
        return _message('Internal check failed: the measure lists used here no longer match the '
                        'what-if engine (what_if_fxn.py), so no path is shown.')

    # leverage per SD of each measure
    n_dom = {d: sum(1 for m in meas if dom_of[m] == d and pd.notna(zh[m])) for d in DOMAIN_COLS}
    def lev(m):
        sc, w = DOMAIN_COLS[dom_of[m]]
        return H[w] / (n_dom[dom_of[m]] * dom_sd[dom_of[m]]) if pd.notna(zh[m]) and pd.notna(H[sc]) else 0.0

    def clamp(m, v): return float(min(max(v, lo[m]), hi[m]))
    def zof(m, v): return float(sign[m] * (v - mu[m]) / sd[m])
    def raw(m, z): return float(mu[m] + sign[m] * z * sd[m])

    editable = set(names)
    items = [m for m in meas if m in editable and m in lo and pd.notna(zh[m]) and lev(m) > 0]
    if not items:
        return _message('No editable measures with data for ' + hospital + ' in ' + str(yr) + '.')

    # references, all kept within this year's observed range
    def value_at_pct(m, p):
        """This year's value at percentile p (an observed value, better direction)."""
        o = B['cur_vals'][m]
        i = int(min(max(np.floor(p * len(o)), 0), len(o) - 1))
        return float(sign[m] * o[i])

    hist_z, exp_z, prec_z = {}, {}, {}
    xrow = B['Xall'].loc[pnum]
    for m in items:
        Q = B['pct_by_year'].get(m)
        has_pct = Q is not None and pnum in Q.index and m in B['cur_vals']
        # historic best: the hospital's best percentile in any year up to this one, as this year's value
        if has_pct and Q.loc[pnum].notna().any():
            hist_z[m] = zof(m, clamp(m, value_at_pct(m, float(Q.loc[pnum].max()))))
        # similar hospitals: median regression within this year
        keep, beta = B['reg'][m]
        pred = float(beta[0] + xrow[keep].values @ beta[1:])
        exp_z[m] = zof(m, clamp(m, raw(m, pred)))
        # precedent: the percentile the better-performing 10% of similar starters reached a year later
        if has_pct and m in B['prec'] and yr in Q.columns and pd.notna(Q.loc[pnum, yr]):
            st, nx = B['prec'][m]
            near = np.abs(st - float(Q.loc[pnum, yr])) <= NEAR
            if near.sum() >= 30:
                prec_z[m] = zof(m, clamp(m, value_at_pct(m, float(np.quantile(nx[near], PREC_Q)))))

    z0 = np.array([zh[m] for m in items])
    L = np.array([max(0.0, max(hist_z.get(m, -np.inf), exp_z.get(m, -np.inf)) - zh[m]) for m in items])
    C = np.array([max(0.0, prec_z[m] - zh[m]) if m in prec_z else 0.0 for m in items])
    R = np.minimum(L, C)                       # realistic room (for the "Within realistic room" flag)
    a = np.array([lev(m) for m in items])      # summary-score gain per SD
    is_safety = np.array([dom_of[m] == 'Safety of Care' for m in items])
    b = np.where(is_safety, 1.0 / (max(n_dom['Safety of Care'], 1) * dom_sd['Safety of Care']), 0.0)
    sortedZ = [B['sortedZ'][m] for m in items]
    def pct(i, z): return 100.0 * np.searchsorted(sortedZ[i], z, 'left') / len(sortedZ[i])
    p0 = np.array([pct(i, z0[i]) for i in range(len(items))])

    # Candidate targets for each measure: values hospitals actually reported this year that are
    # better than the current one (so every target is on the reporting grid and possible).
    # Allowed moves: at least 5 percentile points; at most 33, or up to 50 if the hospital has
    # held that percentile before (its best within-year percentile in any year so far).
    cands_all, cands_sup = [], []
    for i, m in enumerate(items):
        zs = np.unique(sortedZ[i])
        zs = zs[zs > z0[i] + 1e-12]
        zs = zs[[lo[m] - 1e-9 <= raw(m, z) <= hi[m] + 1e-9 for z in zs]] if len(zs) else zs
        g = 100.0 * np.searchsorted(sortedZ[i], zs, 'left') / len(sortedZ[i]) - p0[i]   # percentile gain
        size = np.maximum(g, SD_POINTS * (zs - z0[i]))                                     # size of the move
        Q = B['pct_by_year'].get(m)
        best_pct = float(Q.loc[pnum].max()) * 100 if (Q is not None and pnum in Q.index and Q.loc[pnum].notna().any()) else -1.0
        limit = np.where(best_pct >= p0[i] + g, HARD_MAX_GAIN, SOFT_MAX_GAIN)
        ok = (g >= MIN_PCT_GAIN) & (size <= limit)
        supported = zs <= z0[i] + R[i] + 1e-9         # within the realistic room (evidence)
        cands_all.append((size[ok], zs[ok]))
        cands_sup.append((size[ok & supported], zs[ok & supported]))
    CUR = {}
    def use(cset):
        CUR['c'] = cset
        CUR['gains'] = np.unique(np.concatenate([c[0] for c in cset])) if cset else np.array([])

    def best_at(k):
        """For each measure, the largest allowed move of at most k points: (gain, dz) or None."""
        out = []
        for g, zs in CUR['c']:
            j = np.searchsorted(g, k, 'right') - 1   # g is increasing
            out.append(None if j < 0 else (float(g[j]), float(zs[j])))
        return out

    def select(k, N, G, Hs):
        """Pick at most N measures, each moving up to k points, meeting the summary goal G
        (and the Safety of Care goal Hs). Returns {item index: target z} or None."""
        moves = best_at(k)
        avail = [i for i in range(len(items)) if moves[i] is not None]
        chosen = []
        if Hs is not None and Hs > 0:   # first the fewest safety measures that lift Safety of Care enough
            saf = sorted([i for i in avail if is_safety[i]], key=lambda i: -b[i] * (moves[i][1] - z0[i]))
            tot = 0.0
            for i in saf:
                if tot >= Hs or len(chosen) >= N:
                    break
                chosen.append(i); tot += b[i] * (moves[i][1] - z0[i])
            if tot < Hs:
                return None
        rest = sorted([i for i in avail if i not in chosen], key=lambda i: -a[i] * (moves[i][1] - z0[i]))
        chosen += rest[:max(0, N - len(chosen))]
        if sum(a[i] * (moves[i][1] - z0[i]) for i in chosen) < G:
            return None
        return {i: moves[i][1] for i in chosen}

    def smallest_k(N, G, Hs):
        """The smallest largest-move (in percentile points) that reaches the goals with <= N measures."""
        gains = CUR['gains']
        lo_, hi_ = 0, len(gains) - 1
        if hi_ < 0 or select(gains[hi_], N, G, Hs) is None:
            return None
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if select(gains[mid], N, G, Hs) is not None:
                hi_ = mid
            else:
                lo_ = mid + 1
        return float(gains[lo_])

    def trim(plan, G, Hs):
        """Lower or drop changes that aren't needed, starting with the least helpful ones."""
        plan = dict(plan)
        def totals(p):
            return (sum(a[i] * (p[i] - z0[i]) for i in p), sum(b[i] * (p[i] - z0[i]) for i in p))
        for i in sorted(plan, key=lambda i: a[i] * (plan[i] - z0[i])):
            g, zs = CUR['c'][i]
            for z in [None] + [float(z) for z in zs if z < plan[i]]:   # try dropping, then smaller moves
                trial = dict(plan)
                if z is None:
                    trial.pop(i)
                else:
                    trial[i] = z
                tg, th = totals(trial)
                if tg >= G - 1e-12 and (Hs is None or Hs <= 0 or th >= Hs - 1e-12):
                    plan = trial
                    break
        return plan

    def plan_for(t):
        """Balanced, prioritized plan for target star t, or None if it can't be reached within the limits."""
        base = _engine_baseline(yr)
        q1 = base['Std_Outcomes_Safety_score'].quantile(0.25)
        lowered = (base['Std_Outcomes_Safety_score'] <= q1) & base['star'].apply(lambda s_: safety_penalty(yr, s_ + 1))
        peer = base[(base['grp'] == int(H['cnt_grp'])) & ~lowered]
        def cutoff(s_):
            below, above = peer.loc[peer['star'] == s_ - 1, 'summary_score'], peer.loc[peer['star'] >= s_, 'summary_score']
            if below.empty or above.empty:
                return None
            return above.min() if AIM == 'upper' else (below.max() + above.min()) / 2
        options = []
        if eng_safety <= q1 and safety_penalty(yr, t):
            if cutoff(t) is not None:
                options.append(('lift safety', max(0.0, cutoff(t) + MARGIN - H['summary_score']),
                                q1 + SAFETY_MARGIN - eng_safety))
            if t < 5 and cutoff(t + 1) is not None:
                options.append(('next cutoff', max(0.0, cutoff(t + 1) + MARGIN - H['summary_score']), None))
        elif cutoff(t) is not None:
            options.append(('cutoff', max(0.0, cutoff(t) + MARGIN - H['summary_score']), None))
        best = None
        for name, G, Hs in options:
            for cap_n in MAX_MEASURES + (len(items),):
                ks = [(N, smallest_k(N, G, Hs)) for N in range(1, min(cap_n, len(items)) + 1)]
                ks = [(N, k) for N, k in ks if k is not None]
                if ks:
                    moderate = [(N, k) for N, k in ks if k <= MODERATE_GAIN]
                    N, k = moderate[0] if moderate else min(ks, key=lambda x: (x[1], x[0]))
                    plan = trim(select(k, N, G, Hs), G, Hs)
                    key = (max(max(pct(i, z) - p0[i], SD_POINTS * (z - z0[i])) for i, z in plan.items()), len(plan))
                    if best is None or key < best[0]:
                        best = (key, name, plan)
                    break
        return best

    def plan_with_fallback(t):
        use(cands_sup)
        p = plan_for(t)
        if p is not None:
            return p, False
        use(cands_all)
        p = plan_for(t)
        return p, p is not None

    best, beyond_evidence = plan_with_fallback(target)
    reached = target
    if best is None:
        for t in range(target - 1, engine_star, -1):
            best, beyond_evidence = plan_with_fallback(t)
            if best is not None:
                reached = t
                break
    if best is None:
        return _message('Within these limits (at most 33 percentile points per measure, or 50 where the hospital '
                        'has done it before), no improvement path above ' + str(engine_star) + ' stars was found for ' +
                        hospital + ' in ' + str(yr) + '.')
    _, name, plan = best
    if not plan:
        return _message('No measure changes are needed to reach ' + str(reached) + ' stars for ' + hospital +
                        ' in ' + str(yr) + ' according to the what-if engine.')

    # rows (ranked by how much each change helps the summary score), with a range check on every value
    rows, targets, problems = [], {}, []
    for i in sorted(plan, key=lambda i: -a[i] * (plan[i] - z0[i])):
        m = items[i]
        dec = B['decimals'][m]
        zt = plan[i]
        vals = {'Current': raw(m, z0[i]), 'Target': raw(m, zt),
                'Historic best': raw(m, hist_z[m]) if m in hist_z else None,
                'Similar hospitals': raw(m, exp_z[m]),
                'Precedent': raw(m, prec_z[m]) if m in prec_z else None}
        tol = 1e-9 * max(1.0, abs(hi[m]), abs(lo[m]))
        for k_, v in vals.items():
            if v is not None and not (lo[m] - tol <= v <= hi[m] + tol):
                problems.append((m, k_, v))
        targets[m] = round(vals['Target'], dec)
        rows.append({
            'Domain': dom_of[m],
            'Measure-ID': m,
            'Measure Name': names[m],
            'Higher is better': 'Yes' if sign[m] > 0 else 'No',
            'Current value': round(vals['Current'], dec),
            'Target value': round(vals['Target'], dec),
            'Percentile now': float(round(p0[i], 1)),
            'Percentile at target': float(round(pct(i, zt), 1)),
            'Points gained': float(round(pct(i, zt) - p0[i], 1)),
            'Size of move': float(round(max(pct(i, zt) - p0[i], SD_POINTS * (zt - z0[i])), 1)),
            'Historic best': None if vals['Historic best'] is None else round(vals['Historic best'], dec),
            'Similar hospitals': round(vals['Similar hospitals'], dec),
            'Precedent': None if vals['Precedent'] is None else round(vals['Precedent'], dec),
            'Within realistic room': 'Yes' if zt - z0[i] <= R[i] + 1e-9 else 'No',
        })
    if problems:
        return _message('Internal check failed: a value fell outside the range observed among hospitals, '
                        'so no path is shown.')

    notes = []
    if engine_star != current:
        notes.append('Note: on the unchanged data, the what-if engine rates this hospital ' + str(engine_star) +
                     ' stars; the stored rating on the main page is ' + str(current) + '. This path starts from the '
                     "engine's rating, since it is confirmed with the engine.")
    if reached < target:
        notes.append('Within these limits, ' + str(target) + ' stars could not be reached; this is the path to ' +
                     str(reached) + ' stars, the closest reachable rating.')
    if beyond_evidence:
        notes.append('No path was found using only moves supported by the evidence, so some moves go beyond '
                     'the realistic room (marked No).')
    if name == 'lift safety':
        notes.append('This path also lifts Safety of Care out of the bottom quarter of hospitals; '
                     'otherwise the rating would be lowered by one star.')
    elif name == 'next cutoff':
        notes.append('Safety of Care stays in the bottom quarter of hospitals, which lowers the rating by one star, '
                     'so this path clears the cutoff for ' + str(reached + 1) + ' stars.')
    return {'ok': True, 'message': '', 'rows': rows, 'targets': targets, 'current': current,
            'target': target, 'reached': reached, 'max_gain': max(r['Size of move'] for r in rows),
            'note': ' '.join(notes)}
