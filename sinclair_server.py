"""
=============================================================
  SINCLAIR AI — GEX API SERVER
  Bridges the Python scanner → HTML dashboard
=============================================================

Run this once before/during market hours:
    python sinclair_server.py

Then open sinclair_heatseeker_live.html in your browser.
The dashboard will auto-fetch and auto-refresh every 5 minutes.

Endpoints:
    GET /api/tickers          → list of available tickers
    GET /api/gex/<SYMBOL>     → full node-level GEX + VEX data for one ticker
    GET /api/gex/all          → summary levels for all core tickers
    GET /api/health           → server status + last scan time

All endpoints return JSON with CORS headers so the HTML
file can fetch from file:// or any local origin.
=============================================================
"""

import subprocess, json, sys, os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

CORE_TICKERS = [
    "SPY", "SPX", "UVXY", "XOM", "UPS", "RIVN", "NVDA",
    "AMD", "AAPL", "HOOD", "AVGO", "SOFI", "AFRM"
]

CACHE_TTL = 300     # Seconds before re-fetching (5 min)
PORT      = int(os.environ.get('PORT', 5050))  # Render sets $PORT

# Path to the worker script (runs yfinance in its own process)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sinclair_worker.py')
PYTHON = sys.executable

# ─────────────────────────────────────────────
#  CACHE
# ─────────────────────────────────────────────

_cache = {}
_cache_lock = threading.Lock()

def cache_get(symbol):
    with _cache_lock:
        entry = _cache.get(symbol)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['data']
    return None

def cache_set(symbol, data):
    with _cache_lock:
        _cache[symbol] = {'data': data, 'ts': time.time()}

# ─────────────────────────────────────────────
#  FETCH VIA SUBPROCESS
# ─────────────────────────────────────────────

def fetch_ticker_data(symbol):
    """
    Fetch ticker data by spawning a subprocess.
    yfinance + curl_cffi run in their own process,
    avoiding any conflict with Flask's socket handling.
    """
    cached = cache_get(symbol)
    if cached:
        return cached

    try:
        result = subprocess.run(
            [PYTHON, WORKER, symbol],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            err = result.stderr.strip().split('\n')[-1] if result.stderr else 'Unknown error'
            return {'symbol': symbol, 'error': err}

        data = json.loads(result.stdout)
        if not data.get('error'):
            cache_set(symbol, data)
        return data

    except subprocess.TimeoutExpired:
        return {'symbol': symbol, 'error': 'Fetch timed out (60s)'}
    except json.JSONDecodeError:
        return {'symbol': symbol, 'error': 'Invalid response from worker'}
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def root():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'sinclair_heatseeker_live.html')

@app.route('/api/health')
def health():
    with _cache_lock:
        cached_syms = list(_cache.keys())
        oldest = min(
            (time.time() - v['ts'] for v in _cache.values()), default=0
        )
    return jsonify({
        'status':      'ok',
        'cached':      cached_syms,
        'oldest_data': f'{int(oldest)}s ago',
        'cache_ttl':   CACHE_TTL,
        'tickers':     CORE_TICKERS,
        'server_time': datetime.now().isoformat(),
    })

@app.route('/api/tickers')
def tickers():
    return jsonify({'tickers': CORE_TICKERS})

@app.route('/api/gex/<symbol>')
def gex_single(symbol):
    sym = symbol.upper()
    data = fetch_ticker_data(sym)
    return jsonify(data)

@app.route('/api/gex/all')
def gex_all():
    results = []
    for sym in CORE_TICKERS:
        d = cache_get(sym)
        if d:
            results.append({
                'symbol':  d['symbol'],
                'spot':    d['spot'],
                'chg_pct': d['chg_pct'],
                'levels':  d['levels'],
                'fetched_at': d['fetched_at'],
            })
        else:
            results.append({'symbol': sym, 'spot': None, 'levels': None, 'cached': False})
    return jsonify({'tickers': results, 'count': len(results)})

@app.route('/api/refresh/<symbol>')
def refresh(symbol):
    sym = symbol.upper()
    with _cache_lock:
        _cache.pop(sym, None)
    data = fetch_ticker_data(sym)
    return jsonify({'refreshed': sym, 'ok': data.get('error') is None})

# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 58)
    print("  SINCLAIR AI — GEX API SERVER")
    print(f"  http://localhost:{PORT}")
    print("=" * 58)
    print(f"  Tickers : {', '.join(CORE_TICKERS)}")
    print(f"  Cache   : {CACHE_TTL}s TTL")
    print(f"  Greeks  : GEX + VEX (Vanna) + CEX (Charm)")
    print("=" * 58)

    app.run(host='0.0.0.0', port=PORT, debug=False)
