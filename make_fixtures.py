"""
make_fixtures.py — generate synthetic pcaps so the pipeline can be proven
end-to-end WITHOUT capturing live traffic first.

It writes three pcaps into ./fixtures :
  - ai_sample.pcap      : an "AI-only session" — many LLM streaming flows
  - normal_sample.pcap  : a "normal-only session" — page loads, REST calls,
                          bulk downloads
  - mixed_capture.pcap  : a fresh mixed capture to run detect.py on

These are HANDCRAFTED to embody the thesis so the demo is legible:
  * an LLM streaming flow = one long-lived connection dribbling many small
    downstream packets over several seconds, with variable (bursty) gaps
  * a page load / bulk download = a short burst of large downstream packets
  * a REST call = a short request/response with few packets

Real captures are messier; these are clean teaching examples. IP/port values
here are arbitrary — the detector never uses them as features (it only uses
them to tell upstream from downstream).

NOTE: scapy is a *dev/test* dependency used only to synthesize these fixtures.
The actual detector (features.py / detect.py) uses tshark, not scapy.
"""

import logging
import os
import random

# Silence scapy's "no libpcap provider" / MAC-resolution warnings — we only
# WRITE pcaps here, we never sniff, so none of that machinery matters.
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import Ether, IP, TCP, Raw, wrpcap

# Fixed, arbitrary link-layer addresses. Using explicit MACs stops scapy from
# trying to ARP-resolve a real one (which spews warnings on a host with no
# capture driver). The detector never looks at MACs anyway.
_SRC_MAC = "02:00:00:00:00:01"
_DST_MAC = "02:00:00:00:00:02"

random.seed(1337)  # deterministic fixtures

BASE_TIME = 1_700_000_000.0   # arbitrary epoch base for packet timestamps
ETH_IP_TCP_OVERHEAD = 54      # Ether(14) + IP(20) + TCP(20, no options)
MTU_FRAME = 1500              # a "full" data packet on the wire


def _pkt(src_ip, dst_ip, sport, dport, frame_len, t, flags="PA"):
    """Build one packet whose on-wire length == frame_len, stamped at time t.

    frame.len that tshark reports == len(bytes(pkt)) == 54 + payload bytes.
    """
    payload_len = max(0, frame_len - ETH_IP_TCP_OVERHEAD)
    p = (
        Ether(src=_SRC_MAC, dst=_DST_MAC)
        / IP(src=src_ip, dst=dst_ip)
        / TCP(sport=sport, dport=dport, flags=flags)
        / Raw(load=b"\x00" * payload_len)
    )
    p.time = t
    return p


def _endpoints(i):
    """Distinct client/server endpoints per flow so flow keys don't collide."""
    client_ip = f"10.0.0.{10 + (i % 200)}"
    server_ip = f"203.0.113.{1 + (i % 250)}"       # TEST-NET-3 (docs range)
    client_port = 40000 + (i * 7) % 20000
    server_port = 443
    return client_ip, server_ip, client_port, server_port


def _handshake(cip, sip, cport, sport, t0):
    """Return (packets, t_after) for a 3-way handshake starting at t0.

    The client's SYN is the earliest packet, so the feature extractor's
    "first packet source == client" direction rule holds.
    """
    pkts = [
        _pkt(cip, sip, cport, sport, 54, t0 + 0.000, flags="S"),   # SYN   (client)
        _pkt(sip, cip, sport, cport, 54, t0 + 0.010, flags="SA"),  # SYNACK(server)
        _pkt(cip, sip, cport, sport, 54, t0 + 0.011, flags="A"),   # ACK   (client)
    ]
    return pkts, t0 + 0.012


def make_ai_flow(i, start):
    """One LLM streaming flow: long-lived, many small downstream packets,
    variable inter-arrival gaps (token-by-token generation)."""
    cip, sip, cport, sport = _endpoints(i)
    pkts, t = _handshake(cip, sip, cport, sport, start)

    # Client sends a small prompt/request (a couple of upstream packets).
    for _ in range(random.randint(1, 3)):
        pkts.append(_pkt(cip, sip, cport, sport, random.randint(120, 500), t))
        t += 0.002

    # Small "time to first token", then the stream.
    t += random.uniform(0.2, 0.8)

    n_down = random.randint(45, 220)                 # many downstream packets
    duration_target = random.uniform(3.0, 14.0)      # dribbled over seconds
    gap_mean = duration_target / n_down              # avg gap between chunks

    for k in range(n_down):
        # Small token-chunk packet: payload ~ a token or few -> frame < 300B.
        frame = ETH_IP_TCP_OVERHEAD + random.randint(8, 160)
        pkts.append(_pkt(sip, cip, sport, cport, frame, t))
        # Bursty, variable gaps (Poisson-ish) => high coefficient of variation,
        # the timing signature of token streaming.
        t += random.expovariate(1.0 / gap_mean)
        # Client acks every ~12 chunks (small upstream packet).
        if k % 12 == 11:
            pkts.append(_pkt(cip, sip, cport, sport, 54, t, flags="A"))
    return pkts


def make_normal_flow(i, start):
    """One 'normal' flow, randomly one of three shapes that are NOT the LLM
    trickle: a page-load burst, a short REST call, or a bulk download."""
    cip, sip, cport, sport = _endpoints(i)
    pkts, t = _handshake(cip, sip, cport, sport, start)
    kind = random.choice(["page", "rest", "bulk"])

    # Client request.
    for _ in range(random.randint(1, 3)):
        pkts.append(_pkt(cip, sip, cport, sport, random.randint(120, 600), t))
        t += 0.001

    if kind == "rest":
        # Short request/response: a handful of packets, sub-second.
        n_down = random.randint(6, 14)
        for k in range(n_down):
            frame = ETH_IP_TCP_OVERHEAD + random.randint(200, 1200)
            pkts.append(_pkt(sip, cip, sport, cport, frame, t))
            t += random.uniform(0.001, 0.02)
    else:
        # page/bulk: a burst of large (near-MTU) downstream packets, fast.
        n_down = random.randint(30, 140)
        for k in range(n_down):
            frame = random.choice([MTU_FRAME, MTU_FRAME, random.randint(900, 1500)])
            pkts.append(_pkt(sip, cip, sport, cport, frame, t))
            t += random.uniform(0.0004, 0.004)     # tiny, near-constant gaps
            if k % 10 == 9:
                pkts.append(_pkt(cip, sip, cport, sport, 54, t, flags="A"))
    return pkts


def build(pcap_path, flow_fn, n_flows, start_stagger=0.05):
    all_pkts = []
    for i in range(n_flows):
        start = BASE_TIME + i * start_stagger
        all_pkts.extend(flow_fn(i, start))
    all_pkts.sort(key=lambda p: p.time)     # write in capture (time) order
    wrpcap(pcap_path, all_pkts)
    print(f"wrote {pcap_path}: {n_flows} flows, {len(all_pkts)} packets")


def main():
    os.makedirs("fixtures", exist_ok=True)

    # Training captures: each class recorded "separately" (see README).
    build("fixtures/ai_sample.pcap", make_ai_flow, n_flows=45)
    build("fixtures/normal_sample.pcap", make_normal_flow, n_flows=45)

    # A fresh mixed capture to run detect.py on. Offset the flow indices so the
    # endpoints differ from the training set.
    mixed = []
    for i in range(6):
        mixed.extend(make_ai_flow(500 + i, BASE_TIME + i * 0.05))
    for i in range(6):
        mixed.extend(make_normal_flow(600 + i, BASE_TIME + 10 + i * 0.05))
    mixed.sort(key=lambda p: p.time)
    wrpcap("fixtures/mixed_capture.pcap", mixed)
    print(f"wrote fixtures/mixed_capture.pcap: 12 flows "
          f"(6 AI + 6 normal), {len(mixed)} packets")


if __name__ == "__main__":
    main()
