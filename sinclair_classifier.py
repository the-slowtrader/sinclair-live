"""
=============================================================
  SINCLAIR AI — NODE CLASSIFICATION ENGINE
  Mimics Heatseeker's UX layer exactly as documented
=============================================================

Classification hierarchy (from Skylit docs, in priority order):

  KING         Highest absolute GEX on the entire map.
               Can be yellow (positive) or purple (negative).
               Color is irrelevant — magnitude is everything.
               EOW/EOD settlement gravity target.

  GATEKEEPER   2nd/3rd highest absolute GEX nodes that sit
               between current price and the King Node.
               Block access to King, trigger reversals.
               "Bouncers at the door of the nightclub."

  CLUSTER      3+ consecutive strikes with the same GEX sign
               and magnitude > CLUSTER_THRESHOLD × max_abs.
               Positive cluster = stabilizing spine (pin zone).
               Negative cluster = acceleration channel.

  DOUBLE_STACK Strike where GEX magnitude ranks in top-N
               across multiple aggregated expirations.
               Stronger bounce expected vs single-expiry node.
               (Approximated from weighted-OI concentration.)

  VELOCITY     Node whose magnitude is disproportionately
               large relative to its immediate neighbors
               (local gradient spike). Fast, wicky moves
               expected when price enters.

  STANDARD     Everything else — still a magnet, just lower
               in the hierarchy. Sized by absolute magnitude.

Color encoding (follows Skylit exactly):
  Positive GEX → yellow/Pika    → absorbing, smooth, low-vol
  Negative GEX → purple/Barney  → amplifying, wicky, high-vol

Node strength (1–5 stars):
  Derived from absolute GEX percentile rank across the map.
  5★ = top 5%  |  4★ = top 20%  |  3★ = top 50%  |  etc.
  Shown as a number in the bar label (like Skylit's node values).

=============================================================
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional


def _safe_int(v, default: int = 0) -> int:
    """Convert to int safely — treats None / NaN / inf as default."""
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────
#  THRESHOLDS  (tune after live observation)
# ─────────────────────────────────────────────

# Gatekeepers: top-N nodes by |GEX| that block path to King
GATEKEEPER_COUNT       = 2      # How many gatekeepers to label

# Cluster: min consecutive same-sign nodes to form a cluster
CLUSTER_MIN_RUN        = 3
# Min magnitude relative to map max to count toward a cluster
CLUSTER_THRESH_PCT     = 0.18   # 18% of map max

# Velocity: node is velocity if its |GEX| is >= this multiple
# of the average of its 2 immediate neighbors
VELOCITY_RATIO         = 3.0

# Double stack: top-N OI concentration by strike (cross-expiry)
DOUBLE_STACK_TOP_N     = 3

# Proximity: fraction of spot price within which a node is "near"
NEAR_SPOT_PCT          = 0.03   # 3% from spot

# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

NODE_TYPES = (
    'KING',
    'GATEKEEPER',
    'CLUSTER',
    'DOUBLE_STACK',
    'VELOCITY',
    'STANDARD',
)

@dataclass
class ClassifiedNode:
    strike:      float
    gex:         float
    vex:         float
    cex:         float
    call_oi:     int
    put_oi:      int
    net_oi:      int

    # Classification output
    node_type:   str   = 'STANDARD'   # one of NODE_TYPES
    strength:    int   = 1            # 1–5 stars
    is_positive: bool  = True         # True=yellow/Pika, False=purple/Barney
    near_spot:   bool  = False        # within NEAR_SPOT_PCT of current price
    cluster_id:  Optional[int] = None # which cluster this belongs to (if any)

    # Carry-forward flags for dashboard
    is_flip:        bool = False
    is_call_wall:   bool = False
    is_put_wall:    bool = False

    def to_dict(self) -> dict:
        return {
            'strike':      round(self.strike, 4),
            'gex':         round(self.gex, 2),
            'vex':         round(self.vex, 2),
            'cex':         round(self.cex, 2),
            'call_oi':     self.call_oi,
            'put_oi':      self.put_oi,
            'net_oi':      self.net_oi,
            'node_type':   self.node_type,
            'strength':    self.strength,
            'is_positive': self.is_positive,
            'near_spot':   self.near_spot,
            'cluster_id':  self.cluster_id,
            'is_flip':     self.is_flip,
            'is_call_wall': self.is_call_wall,
            'is_put_wall':  self.is_put_wall,
            # Legacy flags for backward compat with existing dashboard
            'is_king':     self.node_type == 'KING',
            'is_gatekeeper': self.node_type == 'GATEKEEPER',
            'is_cluster':  self.node_type == 'CLUSTER',
            'is_velocity': self.node_type == 'VELOCITY',
            'is_double_stack': self.node_type == 'DOUBLE_STACK',
        }

# ─────────────────────────────────────────────
#  MAIN CLASSIFIER
# ─────────────────────────────────────────────

def classify_nodes(agg_df, spot: float, levels: dict) -> list[dict]:
    """
    Takes the aggregated GEX DataFrame (index=strike, columns incl gex/vex/cex/call_oi/put_oi)
    and the levels dict from extract_levels(), returns a list of node dicts
    sorted high → low strike, each with full classification metadata.

    This is the complete Heatseeker UX classification layer.
    """
    if agg_df.empty:
        return []

    # ── Sort strikes low → high for sequential processing ──
    df = agg_df.sort_index(ascending=True).copy()
    strikes  = df.index.values.astype(float)
    gex_vals = df['gex'].values
    abs_vals = np.abs(gex_vals)
    max_abs  = abs_vals.max() if len(abs_vals) else 1.0
    N        = len(strikes)

    # ── STRENGTH percentile ranks (1–5) ──
    def strength_rank(abs_v: float) -> int:
        pct = abs_v / max_abs
        if pct >= 0.80: return 5
        if pct >= 0.55: return 4
        if pct >= 0.35: return 3
        if pct >= 0.15: return 2
        return 1

    # ── Step 1: Build baseline ClassifiedNode list ──
    nodes: list[ClassifiedNode] = []
    for i, (strike, row) in enumerate(df.iterrows()):
        cn = ClassifiedNode(
            strike    = float(strike),
            gex       = float(row['gex']),
            vex       = float(row.get('vex', 0)),
            cex       = float(row.get('cex', 0)),
            call_oi   = _safe_int(row.get('call_oi', 0)),
            put_oi    = _safe_int(row.get('put_oi', 0)),
            net_oi    = _safe_int(row.get('net_oi', 0)),
            strength  = strength_rank(abs_vals[i]),
            is_positive = float(row['gex']) >= 0,
            near_spot = abs(float(strike) - spot) / spot < NEAR_SPOT_PCT,
        )
        # Carry over level flags
        gf = levels.get('gamma_flip')
        cw = levels.get('call_wall')
        pw = levels.get('put_wall')
        if gf and abs(cn.strike - gf) / max(spot, 1) < 0.002:
            cn.is_flip = True
        if cw and abs(cn.strike - cw) / max(spot, 1) < 0.002:
            cn.is_call_wall = True
        if pw and abs(cn.strike - pw) / max(spot, 1) < 0.002:
            cn.is_put_wall = True
        nodes.append(cn)

    node_map = {n.strike: n for n in nodes}

    # ── Step 2: KING — highest absolute GEX on the map ──
    king_idx = int(np.argmax(abs_vals))
    king_strike = float(strikes[king_idx])
    node_map[king_strike].node_type = 'KING'
    node_map[king_strike].strength  = 5

    # ── Step 3: VELOCITY — local gradient spikes ──
    # A node is velocity if |gex| >= VELOCITY_RATIO × avg(|neighbors|)
    # Skip first/last (no two-sided neighbors)
    classified_strikes = {king_strike}
    for i in range(1, N-1):
        neighbor_avg = (abs_vals[i-1] + abs_vals[i+1]) / 2.0
        if neighbor_avg > 0 and abs_vals[i] / neighbor_avg >= VELOCITY_RATIO:
            s = float(strikes[i])
            if s not in classified_strikes:
                node_map[s].node_type = 'VELOCITY'
                classified_strikes.add(s)

    # ── Step 4: CLUSTER — runs of same-sign nodes above threshold ──
    cluster_thresh = max_abs * CLUSTER_THRESH_PCT
    cluster_id     = 0
    i = 0
    while i < N:
        # Find a run of same-sign nodes above threshold
        sign = np.sign(gex_vals[i])
        if sign == 0:
            i += 1
            continue
        run_start = i
        while i < N and np.sign(gex_vals[i]) == sign and abs_vals[i] >= cluster_thresh:
            i += 1
        run_end = i  # exclusive
        run_len = run_end - run_start
        if run_len >= CLUSTER_MIN_RUN:
            for j in range(run_start, run_end):
                s = float(strikes[j])
                if s not in classified_strikes:
                    node_map[s].node_type = 'CLUSTER'
                    node_map[s].cluster_id = cluster_id
            cluster_id += 1
        # If inner loop didn't advance, move past this node
        if i == run_start:
            i += 1

    # ── Step 5: DOUBLE STACK — high OI concentration across expirations ──
    # Approximate: strikes with highest net_oi in top DOUBLE_STACK_TOP_N
    # (True double stack requires per-expiry node data; we use net_oi as proxy)
    oi_vals   = np.array([n.net_oi for n in nodes], dtype=float)
    if oi_vals.max() > 0:
        ds_thresh_idx = np.argsort(oi_vals)[-DOUBLE_STACK_TOP_N:]
        for idx in ds_thresh_idx:
            s = float(strikes[idx])
            if s not in classified_strikes and oi_vals[idx] > 0:
                node_map[s].node_type = 'DOUBLE_STACK'
                classified_strikes.add(s)

    # ── Step 6: GATEKEEPER — high-abs nodes between spot and King ──
    # Gatekeepers sit BETWEEN spot and King, blocking the path
    between_lo = min(spot, king_strike)
    between_hi = max(spot, king_strike)
    candidates = [
        (abs_vals[i], float(strikes[i]))
        for i in range(N)
        if between_lo < strikes[i] < between_hi
           and float(strikes[i]) not in classified_strikes
    ]
    candidates.sort(reverse=True)  # highest abs first
    for _, s in candidates[:GATEKEEPER_COUNT]:
        node_map[s].node_type = 'GATEKEEPER'
        classified_strikes.add(s)

    # ── Return sorted high → low strike (matches dashboard expectation) ──
    result = sorted(nodes, key=lambda n: n.strike, reverse=True)
    return [n.to_dict() for n in result]


# ─────────────────────────────────────────────
#  SCENARIO READER
#  Translates the classified map into plain-
#  English setup context for Sinclair AI coach
# ─────────────────────────────────────────────

def read_scenario(nodes: list[dict], levels: dict, spot: float) -> dict:
    """
    Given a classified node list, produce a human-readable
    scenario summary matching Skylit's 3 archetypes:

      TRENDING_DAY     — price moving toward King with no strong opposing node
      RANGE_BOUND      — price between yellow clusters (pinning setup)
      VOLATILITY_EVENT — price approaching velocity or negative King Node
    """
    king   = next((n for n in nodes if n['node_type'] == 'KING'), None)
    gates  = [n for n in nodes if n['node_type'] == 'GATEKEEPER']
    velo   = [n for n in nodes if n['node_type'] == 'VELOCITY']
    clusters_pos = [n for n in nodes if n['node_type'] == 'CLUSTER' and n['is_positive']]
    clusters_neg = [n for n in nodes if n['node_type'] == 'CLUSTER' and not n['is_positive']]

    scenario = 'STANDARD'
    notes    = []

    if not king:
        return {'scenario': 'UNKNOWN', 'notes': []}

    dist_to_king = (king['strike'] - spot) / spot * 100

    # VOLATILITY_EVENT: velocity node near spot, or negative King near spot
    velo_near = [n for n in velo if abs(n['strike'] - spot) / spot < 0.04]
    if velo_near or (not king['is_positive'] and abs(dist_to_king) < 3):
        scenario = 'VOLATILITY_EVENT'
        notes.append(f"Velocity / negative King node within 4% of spot — expect fast, wicky moves")
        if velo_near:
            notes.append(f"Velocity node at ${velo_near[0]['strike']:.2f} — sharp reversal or acceleration on touch")

    # RANGE_BOUND: spot between two positive clusters
    elif clusters_pos and len(clusters_pos) >= 2:
        cl_above = [n for n in clusters_pos if n['strike'] > spot]
        cl_below = [n for n in clusters_pos if n['strike'] < spot]
        if cl_above and cl_below:
            scenario = 'RANGE_BOUND'
            notes.append(f"Price trapped between positive cluster bands — fade edges, avoid midpoint")
            notes.append(f"Upper cluster edge: ${cl_above[-1]['strike']:.2f} | Lower cluster edge: ${cl_below[0]['strike']:.2f}")

    # TRENDING_DAY: clean path to King with no strong gatekeeper blocking
    elif gates:
        gate_near = [g for g in gates if abs(g['strike'] - spot) / spot < 0.05]
        if not gate_near:
            scenario = 'TRENDING_DAY'
            notes.append(f"King Node at ${king['strike']:.2f} — clear path, momentum setups favored")
            notes.append(f"Gatekeepers at {[round(g['strike'],2) for g in gates]} — watch for rejection there")
        else:
            scenario = 'GATEKEEPER_TEST'
            notes.append(f"Price testing Gatekeeper at ${gate_near[0]['strike']:.2f} — rejection = powerful reversal")
            notes.append(f"Failure through Gatekeeper opens path to King at ${king['strike']:.2f}")
    else:
        scenario = 'TRENDING_DAY'
        notes.append(f"King Node at ${king['strike']:.2f} ({dist_to_king:+.1f}% from spot) — primary destination")

    # Always: note negative clusters if present
    if clusters_neg:
        notes.append(f"Negative cluster (acceleration zone) at strikes: {[round(n['strike'],2) for n in clusters_neg[:3]]}")

    # Regime append
    regime = levels.get('regime', '?')
    notes.append(f"Regime: {regime} — {'reversal/mean-reversion favored at nodes' if regime=='DAMPENED' else 'momentum / follow-through favored'}")

    return {
        'scenario':  scenario,
        'notes':     notes,
        'king_strike': king['strike'] if king else None,
        'king_positive': king['is_positive'] if king else None,
        'gatekeeper_strikes': [g['strike'] for g in gates],
        'velocity_strikes':   [n['strike'] for n in velo],
    }
