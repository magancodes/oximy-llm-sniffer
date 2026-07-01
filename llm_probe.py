"""
llm_probe.py — fire a REAL streaming LLM request and reconstruct its on-wire
shape, driver-free.

Why not sniff real packets? Live packet capture needs a capture driver (Npcap
on Windows) and admin rights — not "runs on any machine". Instead we make a
genuine streaming API call and measure the REAL thing that matters for this
detector: the arrival TIME and BYTE SIZE of each streamed token chunk. We then
lay those real timings/sizes down as a TCP flow in a pcap and run the exact
same tshark + model pipeline on it.

So the *shape* (timing + sizes) is real, measured from a real LLM. Only the
packet framing around it is reconstructed. We are careful never to claim we
sniffed raw packets — see the honest note surfaced in the portal UI.

Providers supported (pick in the UI):
  - "openai"    : any OpenAI-compatible /chat/completions endpoint
                  (OpenAI, Groq, Together, OpenRouter, LM Studio, vLLM, and
                   Vercel AI Gateway via provider/model strings)
  - "ollama"    : local Ollama (OpenAI-compatible at :11434/v1, no key)
  - "anthropic" : native Anthropic /v1/messages streaming
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

# Reuse the exact packet builder the fixtures use, so reconstructed flows are
# framed identically to synthetic ones.
from make_fixtures import _pkt, _handshake, ETH_IP_TCP_OVERHEAD, MTU_FRAME
from scapy.all import wrpcap

# Rough per-packet TLS record overhead (5-byte record header + AEAD nonce/tag).
# A streamed token chunk rides inside one such record.
TLS_OVERHEAD = 29

# Placeholder endpoints for the reconstructed flow. ARBITRARY on purpose — the
# detector never uses IP/port as a feature, so these values are irrelevant to
# the verdict. (TEST-NET ranges, reserved for documentation.)
_CLIENT_IP, _CLIENT_PORT = "10.0.0.2", 51000
_SERVER_IP, _SERVER_PORT = "198.51.100.10", 443


# ---------------------------------------------------------------------------
# Real streaming clients — each yields events as tokens actually arrive.
# Event shape: {"text": <delta str>, "t": <arrival epoch>, "bytes": <raw len>}
# ---------------------------------------------------------------------------
def _openai_url(base_url):
    """Build the chat-completions URL, tolerating common Base URL mistakes.

    Accepts:  https://api.openai.com            -> .../v1/chat/completions
              https://api.openai.com/v1         -> .../v1/chat/completions
              https://gw.example.com/openai/v1  -> .../chat/completions
              ...already ending in /chat/completions -> used as-is
    """
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    # A bare host (no path) conventionally needs the /v1 prefix.
    if urlparse(base).path in ("", "/"):
        base += "/v1"
    return base + "/chat/completions"


def _anthropic_url(base_url):
    """Build the messages URL, tolerating a base that already includes /v1."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")     # avoid /v1/v1/messages
    return base + "/v1/messages"


# Friendly labels for the headers we send, so a bad value names itself.
_HEADER_LABELS = {
    "Authorization": "API key",
    "x-api-key": "API key",
    "anthropic-version": "anthropic-version header",
    "Content-Type": "Content-Type header",
}


def _validate_headers(headers):
    """HTTP headers must be Latin-1 (ASCII) encodable. A pasted API key with a
    curly "smart quote" or other non-ASCII char otherwise blows up deep inside
    urllib with an opaque UnicodeEncodeError. Catch it here with a clear message.
    """
    for name, value in headers.items():
        try:
            value.encode("latin-1")
        except UnicodeEncodeError as e:
            bad = value[e.start:e.start + 1]
            label = _HEADER_LABELS.get(name, f"{name} header")
            raise RuntimeError(
                f"The {label} contains a non-ASCII character ({bad!r}) — most "
                f"often a curly quote from copy-pasting out of a doc, PDF, or "
                f"chat that auto-formats quotes. Re-type it (or paste as plain "
                f"text) using only plain ASCII characters."
            ) from None


def _open_stream(url, headers, body, timeout):
    _validate_headers(headers)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)
    except urllib.error.HTTPError as e:
        # Surface WHICH url failed and WHAT the server said — a bare "404" is
        # useless. The response body usually names the exact problem.
        body_txt = ""
        try:
            body_txt = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        hint = ""
        if e.code == 404:
            hint = " (404 usually means the Base URL path is wrong for this provider)"
        raise RuntimeError(
            f"HTTP {e.code} calling {url}{hint}. Server said: {body_txt or e.reason}"
        ) from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach {url}: {e.reason}. "
            f"Check the Base URL is correct and the server is running."
        ) from None


def _clean_key(api_key):
    """Trim whitespace and any wrapping quotes users paste around a key."""
    return (api_key or "").strip().strip('"').strip("'").strip("“”")


def _stream_openai(base_url, api_key, model, prompt, max_tokens, timeout):
    url = _openai_url(base_url)
    api_key = _clean_key(api_key)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    resp = _open_stream(url, headers, body, timeout)
    for raw in resp:                       # bytes, line-by-line as they arrive
        t = time.time()
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
            delta = obj["choices"][0]["delta"].get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            yield {"text": delta, "t": t, "bytes": len(raw)}


def _stream_anthropic(base_url, api_key, model, prompt, max_tokens, timeout):
    url = _anthropic_url(base_url)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _clean_key(api_key),
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = _open_stream(url, headers, body, timeout)
    for raw in resp:
        t = time.time()
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "content_block_delta":
            delta = obj.get("delta", {}).get("text", "")
            if delta:
                yield {"text": delta, "t": t, "bytes": len(raw)}
        elif obj.get("type") == "message_stop":
            break


def stream_llm(provider, base_url, api_key, model, prompt,
               max_tokens=220, timeout=90):
    """Dispatch to the right provider and yield token-arrival events."""
    provider = (provider or "openai").lower()
    if provider == "anthropic":
        yield from _stream_anthropic(base_url, api_key, model, prompt, max_tokens, timeout)
    else:  # "openai" and "ollama" are both OpenAI-compatible
        yield from _stream_openai(base_url, api_key, model, prompt, max_tokens, timeout)


# ---------------------------------------------------------------------------
# Reconstruct a TCP flow pcap from the measured stream events.
# ---------------------------------------------------------------------------
def synthesize_pcap_from_events(events, out_path, request_bytes=400):
    """Lay real (time, size) token events down as one server->client flow.

    events: list of {"t": epoch, "bytes": raw_chunk_len}. Each becomes one (or,
    if huge, several MTU-sized) downstream packet at its REAL arrival time.
    Returns a small summary dict.
    """
    if not events:
        raise ValueError("no stream events captured (empty LLM response?)")

    first_t = events[0]["t"]
    # Place the handshake + request slightly before the first token so the
    # flow initiator (client SYN) is the earliest packet, as in a real capture.
    t0 = first_t - 0.30
    pkts, t = _handshake(_CLIENT_IP, _SERVER_IP, _CLIENT_PORT, _SERVER_PORT, t0)
    # The client's prompt goes up as a small request.
    pkts.append(_pkt(_CLIENT_IP, _SERVER_IP, _CLIENT_PORT, _SERVER_PORT,
                     ETH_IP_TCP_OVERHEAD + request_bytes, t))

    total_down_bytes = 0
    n_down_pkts = 0
    for i, ev in enumerate(events):
        # A token chunk on the wire = TLS record overhead + the chunk bytes.
        remaining = TLS_OVERHEAD + ev["bytes"]
        et = ev["t"]
        # Split anything above the MTU across back-to-back packets (rare for
        # token streaming; common only for a big first/flush chunk).
        while remaining > 0:
            body = min(remaining, MTU_FRAME - ETH_IP_TCP_OVERHEAD)
            frame = ETH_IP_TCP_OVERHEAD + body
            pkts.append(_pkt(_SERVER_IP, _CLIENT_IP, _SERVER_PORT, _CLIENT_PORT,
                             frame, et))
            total_down_bytes += frame
            n_down_pkts += 1
            remaining -= body
            et += 0.0001
        # Client acks every ~12 downstream packets.
        if i % 12 == 11:
            pkts.append(_pkt(_CLIENT_IP, _SERVER_IP, _CLIENT_PORT, _SERVER_PORT,
                             ETH_IP_TCP_OVERHEAD, ev["t"], flags="A"))

    pkts.sort(key=lambda p: p.time)
    wrpcap(out_path, pkts)

    last_t = events[-1]["t"]
    return {
        "n_token_events": len(events),
        "n_down_pkts": n_down_pkts,
        "total_down_bytes": total_down_bytes,
        "stream_seconds": last_t - first_t,
        "ttft_seconds": None,  # filled in by caller (needs request-send time)
    }


# ---------------------------------------------------------------------------
# Synthetic demo pcaps for the "simulated" portal mode (no API key needed).
# ---------------------------------------------------------------------------
def synthesize_demo_pcap(kind, out_path, n_flows=4, seed_offset=0):
    """Build a small demo pcap of AI / normal / mixed flows for the portal."""
    from make_fixtures import make_ai_flow, make_normal_flow, BASE_TIME

    pkts = []
    if kind == "ai":
        for i in range(n_flows):
            pkts += make_ai_flow(700 + seed_offset + i, BASE_TIME + i * 0.05)
    elif kind == "normal":
        for i in range(n_flows):
            pkts += make_normal_flow(800 + seed_offset + i, BASE_TIME + i * 0.05)
    else:  # mixed
        half = max(1, n_flows // 2)
        for i in range(half):
            pkts += make_ai_flow(900 + seed_offset + i, BASE_TIME + i * 0.05)
        for i in range(half):
            pkts += make_normal_flow(950 + seed_offset + i, BASE_TIME + 5 + i * 0.05)

    pkts.sort(key=lambda p: p.time)
    wrpcap(out_path, pkts)
