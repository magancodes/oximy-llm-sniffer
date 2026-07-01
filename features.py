"""
features.py — turn a pcap into one feature row per TCP flow.

============================================================================
THE WHOLE THESIS, IN ONE PARAGRAPH
============================================================================
We detect LLM/AI usage purely from the *shape* of the traffic: how packets
are timed and sized. We NEVER look at *who* the traffic is talking to.
That means the model is forbidden from using any of:

    - hostname / SNI          (hidden once Encrypted Client Hello ships)
    - destination IP address
    - TCP/UDP port numbers
    - TLS certificate / JA3 / anything that names the peer

The insight we classify on: an LLM *streaming* response is one long-lived TCP
connection that dribbles many small downstream packets over several seconds
(one chunk per token-ish). A web page load is one big burst of large packets.
A normal REST call is a short request/response. That "trickle of small packets
for a long time" shape is the fingerprint — and it survives ECH because there
is no identifier left to encrypt; the shape is on the wire in the clear.

============================================================================
THE "NO IDENTIFIERS AS FEATURES" RULE  (this is the point of the project)
============================================================================
IP addresses and ports appear in this file for EXACTLY ONE reason: to split a
bidirectional flow into its two halves — client->server (upstream) and
server->client (downstream). We need to know which direction a packet went to
measure the downstream trickle. That is a *bookkeeping* use, not a *feature*
use. The raw IP/port values never enter the feature vector.

`FEATURE_COLS` below is the model's entire input surface. Read it and confirm
for yourself: there is not one identifier in it. Only counts, durations,
sizes, ratios, and timing statistics. If a value could tell you *who* the
server is, it does not belong in that list.
============================================================================
"""

from __future__ import annotations

import logging
from statistics import mean, pstdev

import pandas as pd

# We parse pcaps in PURE PYTHON with scapy instead of shelling out to tshark.
# That removes the only system-binary dependency, so the detector runs the same
# way locally and in a serverless environment (e.g. Vercel) with nothing to
# install. scapy prints a one-line "no libpcap" notice on import that we do not
# care about here (we only READ pcap files, never sniff), so quiet it.
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import IP, TCP, PcapReader

# ---------------------------------------------------------------------------
# FEATURE_COLS: the model's ENTIRE input. Zero identifiers. This list is the
# contract the rest of the project is built to honor. Scan it — every entry is
# a count, a duration, a size statistic, a ratio, or an inter-arrival timing
# statistic. Nothing here names a host, IP, or port.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "duration",            # seconds from first to last packet of the flow
    "n_packets",           # total packets in the flow (both directions)
    "n_down",              # downstream (server->client) packet count
    "n_up",                # upstream   (client->server) packet count
    "down_up_byte_ratio",  # downstream bytes / upstream bytes
    "down_size_mean",      # mean downstream packet size (bytes)
    "down_size_std",       # std  downstream packet size (bytes)
    "frac_small_down",     # fraction of downstream packets smaller than 300B
    "down_pps",            # downstream packets per second
    "down_iat_mean",       # mean downstream inter-arrival gap (seconds)
    "down_iat_std",        # std  downstream inter-arrival gap
    "down_iat_max",        # max  downstream inter-arrival gap
    "down_iat_cv",         # coefficient of variation of downstream gaps (std/mean)
]

# NON-feature columns we also carry in the CSV. These are identifiers, kept
# ONLY so a human operator can be told which flow got flagged. They are
# deliberately EXCLUDED from FEATURE_COLS so the model can never learn on them.
# (Their presence in the CSV, right next to the features, is the demonstration:
#  the identifiers are available and we choose not to feed them to the model.)
META_COLS = ["client", "server"]

# A downstream packet smaller than this counts as "small". Streamed token
# chunks are tiny; page/asset data packets ride near the ~1500B MTU.
SMALL_PACKET_BYTES = 300

# Flows with fewer than this many packets are dropped — too little signal to
# say anything about their shape (a stray SYN/RST tells us nothing).
MIN_PACKETS = 8


# ---------------------------------------------------------------------------
# pcap parsing (pure Python via scapy; no tshark / system binary required)
# ---------------------------------------------------------------------------
def _read_packets(pcap_path: str) -> list[tuple]:
    """Parse a pcap into raw per-packet tuples, in pure Python.

    Each tuple: (t_epoch, src_ip, dst_ip, src_port, dst_port, frame_len).
    Only IPv4 TCP packets are kept (matching the old `ip and tcp` filter);
    IPv6 is left as future work -- see README limitations.

    We read ip/port here ONLY to know which way each packet flowed. Which
    fields become features is decided later, in _features_for_flow, and the raw
    ip/port values are never among them.
    """
    rows: list[tuple] = []
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            # Skip anything that is not IPv4 TCP -- same effect as tshark's
            # `ip and tcp` display filter.
            if not (pkt.haslayer(IP) and pkt.haslayer(TCP)):
                continue
            ip, tcp = pkt[IP], pkt[TCP]
            # frame_len = original on-wire length (wirelen from the pcap record
            # header); fall back to the captured byte count if it is absent.
            frame_len = int(getattr(pkt, "wirelen", 0) or len(pkt))
            rows.append((
                float(pkt.time),                  # timing -> becomes features
                ip.src, ip.dst,                   # direction split ONLY
                str(tcp.sport), str(tcp.dport),   # direction split ONLY
                frame_len,                         # size -> becomes features
            ))
    return rows


# ---------------------------------------------------------------------------
# flow grouping + feature construction
# ---------------------------------------------------------------------------
def _flow_key(sip, sport, dip, dport):
    """Direction-agnostic key identifying the connection (both halves).

    We sort the two endpoints so that a packet and its reply hash to the same
    flow regardless of which way it was going. This uses IP+port — but ONLY to
    group packets into a flow, never as a feature.
    """
    a = (sip, sport)
    b = (dip, dport)
    return tuple(sorted((a, b)))


def _safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _features_for_flow(packets: list[tuple]) -> dict | None:
    """Build one feature row from a flow's packets (already time-sorted).

    packets: list of (t, sip, dip, sport, dport, flen) for one flow.
    Returns a dict of FEATURE_COLS + META_COLS, or None if the flow is dropped.
    """
    if len(packets) < MIN_PACKETS:
        return None

    packets = sorted(packets, key=lambda p: p[0])

    # ---- DIRECTION SPLIT (the only place IP/port are allowed to be read) ----
    # The initiator of the flow is the client. In a real capture the client
    # sends the SYN first, so the earliest packet's source is the client and
    # its destination is the server. Downstream = server -> client.
    first = packets[0]
    client_ep = (first[1], first[3])   # (src_ip, src_port) of first packet
    server_ep = (first[2], first[4])   # (dst_ip, dst_port) of first packet

    down_times, down_sizes = [], []
    up_bytes = 0
    down_bytes = 0
    n_up = 0

    for t, sip, dip, sport, dport, flen in packets:
        if (sip, sport) == server_ep:      # server -> client  == downstream
            down_times.append(t)
            down_sizes.append(flen)
            down_bytes += flen
        else:                               # client -> server == upstream
            up_bytes += flen
            n_up += 1
    # --- from here on, sip/dip/sport/dport are NEVER touched again. Only the
    #     collected sizes/times/counts feed the features. ---

    n_down = len(down_sizes)
    duration = packets[-1][0] - packets[0][0]

    # Downstream inter-arrival gaps: the "trickle" signal lives here. Token
    # streaming produces many modest gaps; a burst download produces gaps near
    # zero. A high, variable gap over a long duration is the LLM tell.
    iats = [down_times[i] - down_times[i - 1] for i in range(1, len(down_times))]
    iat_mean = mean(iats) if iats else 0.0
    iat_std = _safe_std(iats)
    iat_max = max(iats) if iats else 0.0
    iat_cv = (iat_std / iat_mean) if iat_mean > 0 else 0.0

    down_size_mean = mean(down_sizes) if down_sizes else 0.0
    frac_small = (
        sum(1 for s in down_sizes if s < SMALL_PACKET_BYTES) / n_down
        if n_down else 0.0
    )
    # Guard duration==0 (all timestamps identical) to avoid div-by-zero.
    down_pps = (n_down / duration) if duration > 0 else 0.0
    ratio = down_bytes / up_bytes if up_bytes > 0 else float(down_bytes)

    return {
        # --- FEATURES (shape only) ---
        "duration": duration,
        "n_packets": len(packets),
        "n_down": n_down,
        "n_up": n_up,
        "down_up_byte_ratio": ratio,
        "down_size_mean": down_size_mean,
        "down_size_std": _safe_std(down_sizes),
        "frac_small_down": frac_small,
        "down_pps": down_pps,
        "down_iat_mean": iat_mean,
        "down_iat_std": iat_std,
        "down_iat_max": iat_max,
        "down_iat_cv": iat_cv,
        # --- META (identifiers, for human reporting ONLY — never a feature) ---
        "client": f"{client_ep[0]}:{client_ep[1]}",
        "server": f"{server_ep[0]}:{server_ep[1]}",
    }


def extract_features(pcap_path: str) -> pd.DataFrame:
    """Parse a pcap into a DataFrame: one row per TCP flow.

    Columns = FEATURE_COLS + META_COLS. Flows under MIN_PACKETS are dropped.
    """
    rows = _read_packets(pcap_path)

    flows: dict[tuple, list[tuple]] = {}
    for t, sip, dip, sport, dport, flen in rows:
        key = _flow_key(sip, sport, dip, dport)
        flows.setdefault(key, []).append((t, sip, dip, sport, dport, flen))

    feature_rows = []
    for packets in flows.values():
        feat = _features_for_flow(packets)
        if feat is not None:
            feature_rows.append(feat)

    # Build with an explicit column order so an empty capture still yields a
    # well-formed (0-row) frame with the right columns.
    return pd.DataFrame(feature_rows, columns=FEATURE_COLS + META_COLS)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python features.py <pcap>", file=sys.stderr)
        sys.exit(2)
    df = extract_features(sys.argv[1])
    print(f"{len(df)} flow(s) extracted from {sys.argv[1]}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False))
