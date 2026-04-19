#!/usr/bin/env python3
"""
Sinclair worker — fetches and computes GEX/VEX/CEX for a single ticker.
Called by sinclair_server.py as a subprocess.
Outputs JSON to stdout.
"""

import sys
import os
import json
import math
import warnings
import urllib.request
import urllib.parse
warnings.filterwarnings('ignore')


def _load_env(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE    = 'https://financialmodelingprep.com/stable'


def _fmp_get(path, params):
    params = dict(params or {})
    params['apikey'] = FMP_API_KEY
    url = f"{FMP_BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent':'sinclair/1.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _clean(obj):
    """Recursively replace NaN/Inf with None so JSON.parse() doesn't choke."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

from sinclair_classifier import classify_nodes, read_scenario

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

RISK_FREE_RATE  = 0.053
NUM_EXPIRATIONS = 5
STRIKE_BUFFER   = 0.22

# ─────────────────────────────────────────────
#  BLACK-SCHOLES GREEKS
# ─────────────────────────────────────────────

def d1d2(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_gamma(S, K, T, r, sigma):
    d1, _ = d1d2(S, K, T, r, sigma)
    if d1 is None: return 0.0
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vanna(S, K, T, r, sigma):
    d1, d2 = d1d2(S, K, T, r, sigma)
    if d1 is None: return 0.0
    return -d2 * norm.pdf(d1) / sigma

def bs_charm(S, K, T, r, sigma):
    d1, d2 = d1d2(S, K, T, r, sigma)
    if d1 is None or T <= 1e-6: return 0.0
    return -norm.pdf(d1) * (
        2 * r * T - d2 * sigma * np.sqrt(T)
    ) / (2 * T * sigma * np.sqrt(T))

# ─────────────────────────────────────────────
#  PER-EXPIRY COMPUTATION
# ─────────────────────────────────────────────

def compute_exposure_for_expiry(calls_df, puts_df, spot, expiry_str, r=RISK_FREE_RATE):
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
    T = max((expiry_dt - datetime.now()).days / 365.0, 1/365)
    lo, hi = spot * (1 - STRIKE_BUFFER), spot * (1 + STRIKE_BUFFER)

    records = {}

    def process_leg(df, is_call):
        for _, row in df.iterrows():
            K   = row.get('strike', np.nan)
            oi  = row.get('openInterest', 0) or 0
            iv  = row.get('impliedVolatility', np.nan)
            if pd.isna(K) or pd.isna(iv) or K < lo or K > hi or oi == 0:
                continue

            gamma = bs_gamma(spot, K, T, r, iv)
            vanna = bs_vanna(spot, K, T, r, iv)
            charm = bs_charm(spot, K, T, r, iv)

            sign = 1 if is_call else -1
            gex  = sign * gamma * oi * (spot**2) * 100
            vex  = sign * vanna * oi * spot * iv
            cex  = sign * charm * oi * spot

            if K not in records:
                records[K] = {'gex':0,'vex':0,'cex':0,'call_oi':0,'put_oi':0,'T':T}
            records[K]['gex'] += gex
            records[K]['vex'] += vex
            records[K]['cex'] += cex
            if is_call:
                records[K]['call_oi'] += oi
            else:
                records[K]['put_oi'] += oi

    process_leg(calls_df, True)
    process_leg(puts_df, False)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(records, orient='index')
    df.index.name = 'strike'
    df['net_oi'] = df['call_oi'] + df['put_oi']
    return df.sort_index()

# ─────────────────────────────────────────────
#  AGGREGATE
# ─────────────────────────────────────────────

def aggregate_exposures(ticker_obj, spot):
    all_exps = ticker_obj.options
    today    = datetime.now()
    valid    = [
        e for e in all_exps
        if (datetime.strptime(e, "%Y-%m-%d") - today).days >= 3
    ][:NUM_EXPIRATIONS]

    if not valid:
        return pd.DataFrame(), []

    agg = {}
    exp_list = []

    for exp in valid:
        try:
            chain = ticker_obj.option_chain(exp)
            calls = chain.calls[['strike','openInterest','impliedVolatility']].copy()
            puts  = chain.puts [['strike','openInterest','impliedVolatility']].copy()
        except Exception:
            continue

        df = compute_exposure_for_expiry(calls, puts, spot, exp)
        if df.empty:
            continue

        T      = df['T'].iloc[0]
        weight = 1.0 / np.sqrt(max(T, 1/365))
        dte    = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        exp_list.append({'expiry': exp, 'dte': dte, 'weight': round(weight, 3)})

        for strike, row in df.iterrows():
            if strike not in agg:
                agg[strike] = {'gex':0,'vex':0,'cex':0,'call_oi':0,'put_oi':0}
            agg[strike]['gex']     += row['gex']     * weight
            agg[strike]['vex']     += row['vex']     * weight
            agg[strike]['cex']     += row['cex']     * weight
            agg[strike]['call_oi'] += row['call_oi']
            agg[strike]['put_oi']  += row['put_oi']

    if not agg:
        return pd.DataFrame(), exp_list

    df_out = pd.DataFrame.from_dict(agg, orient='index').sort_index()
    df_out.index.name = 'strike'
    df_out['net_oi'] = df_out['call_oi'] + df_out['put_oi']
    return df_out, exp_list

# ─────────────────────────────────────────────
#  KEY LEVELS
# ─────────────────────────────────────────────

def extract_levels(agg_df, spot):
    if agg_df.empty:
        return {}

    above = agg_df[agg_df.index >= spot]
    below = agg_df[agg_df.index <  spot]

    call_wall = float(above['gex'].idxmax()) if not above.empty else None
    put_wall  = float(below['gex'].idxmin()) if not below.empty else None

    strikes  = agg_df.index.values
    net_gex  = agg_df['gex'].values
    flip     = None
    for i in range(len(strikes)-1):
        if net_gex[i] * net_gex[i+1] < 0:
            flip = float(strikes[i])
            break

    all_strikes = agg_df.index.values
    call_oi_map = dict(zip(agg_df.index, agg_df['call_oi']))
    put_oi_map  = dict(zip(agg_df.index, agg_df['put_oi']))
    min_pain, max_pain_strike = float('inf'), float(all_strikes[0])
    for ts in all_strikes:
        cp = sum(max(s-ts,0)*oi for s,oi in call_oi_map.items())
        pp = sum(max(ts-s,0)*oi for s,oi in put_oi_map.items())
        if cp+pp < min_pain:
            min_pain = cp+pp
            max_pain_strike = float(ts)

    net_gex_total = float(agg_df['gex'].sum())
    net_vex_total = float(agg_df['vex'].sum())
    net_cex_total = float(agg_df['cex'].sum())

    above_flip = spot > flip if flip else net_gex_total > 0
    regime     = 'DAMPENED' if above_flip else 'TRENDING'

    vex_bias = 'IV_TAILWIND'  if net_vex_total > 0 else 'IV_HEADWIND'
    cex_bias = 'TIME_TAILWIND' if net_cex_total > 0 else 'TIME_HEADWIND'

    signals    = [1 if net_gex_total>0 else -1,
                  1 if net_vex_total>0 else -1,
                  1 if net_cex_total>0 else -1]
    confluence = abs(sum(signals))

    return {
        'call_wall':     call_wall,
        'put_wall':      put_wall,
        'gamma_flip':    flip,
        'max_pain':      max_pain_strike,
        'net_gex_m':     round(net_gex_total / 1e6, 2),
        'net_vex_m':     round(net_vex_total / 1e6, 2),
        'net_cex_k':     round(net_cex_total / 1e3, 2),
        'regime':        regime,
        'vex_bias':      vex_bias,
        'cex_bias':      cex_bias,
        'confluence':    confluence,
    }

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: sinclair_worker.py <SYMBOL>'}))
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        ticker = yf.Ticker(symbol)

        # ── Real-time spot + daily history from FMP ──
        from datetime import timedelta
        today = datetime.now().date()
        quote = _fmp_get('quote', {'symbol': symbol})
        if not quote or not isinstance(quote, list) or not quote[0].get('price'):
            print(json.dumps({'symbol': symbol, 'error': 'FMP quote failed'}))
            sys.exit(0)
        q = quote[0]
        spot  = float(q['price'])
        chg_d = round(float(q.get('change') or 0), 2)
        chg   = round(float(q.get('changePercentage') or 0), 2)

        eod = _fmp_get('historical-price-eod/full', {
            'symbol': symbol,
            'from':   (today - timedelta(days=90)).isoformat(),
            'to':     today.isoformat(),
        }) or []
        eod = sorted(eod, key=lambda r: r.get('date',''))

        def _fmt_daily(rows):
            return [{
                'time':  r['date'],
                'open':  round(float(r['open']), 2),
                'high':  round(float(r['high']), 2),
                'low':   round(float(r['low']), 2),
                'close': round(float(r['close']), 2),
            } for r in rows if r.get('date') and r.get('open') is not None]

        price_history = _fmt_daily(eod)
        price_frames  = {'1D': price_history}

        # Intraday 1H (FMP Starter supports historical-chart)
        try:
            hc = _fmp_get('historical-chart/1hour', {
                'symbol': symbol,
                'from':   (today - timedelta(days=30)).isoformat(),
                'to':     today.isoformat(),
            }) or []
            def _fmt_intraday(rows):
                from datetime import datetime as _dt
                out = []
                for r in rows:
                    try:
                        t = int(_dt.strptime(r['date'], '%Y-%m-%d %H:%M:%S').timestamp())
                        out.append({
                            'time':  t,
                            'open':  round(float(r['open']), 2),
                            'high':  round(float(r['high']), 2),
                            'low':   round(float(r['low']), 2),
                            'close': round(float(r['close']), 2),
                        })
                    except Exception:
                        continue
                return sorted(out, key=lambda x: x['time'])
            h1 = _fmt_intraday(hc)
            if h1:
                price_frames['1H'] = h1
                # Downsample 1H → 4H
                buckets = {}
                for b in h1:
                    key = b['time'] - (b['time'] % (4*3600))
                    if key not in buckets:
                        buckets[key] = {'time':key,'open':b['open'],'high':b['high'],'low':b['low'],'close':b['close']}
                    else:
                        bk = buckets[key]
                        bk['high']  = max(bk['high'], b['high'])
                        bk['low']   = min(bk['low'],  b['low'])
                        bk['close'] = b['close']
                price_frames['4H'] = sorted(buckets.values(), key=lambda x: x['time'])
        except Exception as e:
            sys.stderr.write(f"1H fetch failed: {e}\n")

        agg_df, exp_list = aggregate_exposures(ticker, spot)
        if agg_df.empty:
            print(json.dumps({'symbol': symbol, 'error': 'No option chain data'}))
            sys.exit(0)

        levels   = extract_levels(agg_df, spot)
        nodes    = classify_nodes(agg_df, spot, levels)
        scenario = read_scenario(nodes, levels, spot)

        result = {
            'symbol':     symbol,
            'spot':       round(spot, 2),
            'chg_pct':    chg,
            'chg_dollar': chg_d,
            'levels':     levels,
            'nodes':      nodes,
            'scenario':   scenario,
            'expirations': exp_list,
            'price_history': price_history,
            'price_frames':  price_frames,
            'fetched_at': datetime.now().isoformat(),
            'error':      None,
        }

        print(json.dumps(_clean(result)))

    except Exception as e:
        print(json.dumps({'symbol': symbol, 'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
