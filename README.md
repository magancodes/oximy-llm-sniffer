# llm-shape-detector

Detect LLM/AI usage on a network **purely from the shape of the traffic**, the
packet **timing** and **sizes**, and nothing else. No hostname, no TLS SNI, no
IP address, no port number is ever used as a feature.

## Why shape, not names

Today most tools spot AI traffic by reading the **SNI** (the server name sent in
the clear during the TLS handshake) or by matching known IPs. **Encrypted
Client Hello (ECH)** encrypts the SNI, so those tools go blind.

This tool takes the opposite bet: it never looks at *who* you're talking to, only
at the *pattern* of packets. That pattern is on the wire in the clear and there
is **no identifier left to encrypt**, so it keeps working after ECH ships.

### The signal we classify on

| Traffic | Shape |
| --- | --- |
| **LLM streaming response** | one long-lived TCP connection **dribbling many small downstream packets over several seconds** (roughly one chunk per token), with variable, bursty inter-arrival gaps |
| Web page load | one big **burst** of large (near-MTU) packets, over in under a second or two |
| Normal REST/API call | a short **request/response**, few packets |

That "trickle of small packets for a long time" is the fingerprint.

## The one rule that defines this project

**No identifier is ever a feature.** IP addresses and ports are read for exactly
one purpose: to split each flow into its two directions (client to server vs
server to client) so we can measure the *downstream* trickle. Their raw values
never enter the model. See the long comment at the top of
[features.py](features.py) and the `FEATURE_COLS` list; every entry is a count,
duration, size statistic, ratio, or timing statistic. Zero identifiers.

The extracted CSVs deliberately keep the `client`/`server` columns **right next
to** the features, and the model is handed `FEATURE_COLS` only. The identifiers
are available and we choose not to feed them in. That choice is the whole thesis.

## Files

| File | Role |
| --- | --- |
| [features.py](features.py) | pcap to one feature row per TCP flow; defines `FEATURE_COLS` (no identifiers) |
| [extract_features.py](extract_features.py) | `<pcap> <out.csv>` CLI wrapper around features.py |
| [train.py](train.py) | `<ai.csv> <normal.csv>` to a 5-fold out-of-sample report + `model.joblib` |
| [detect.py](detect.py) | `<pcap>` to flagged LLM flows with confidence, on a fresh capture |
| [detector.py](detector.py) | shared scoring (`score_pcap`) + auto-bootstrap of a model on a fresh machine |
| [make_fixtures.py](make_fixtures.py) | synthesize demo pcaps so you can run the whole thing with no live capture |
| [app.py](app.py) + [templates/index.html](templates/index.html) | **web portal**: fire real LLM tests, simulate, or analyze pcaps in a browser |
| [llm_probe.py](llm_probe.py) | fire a real streaming LLM call and reconstruct its on-wire shape (driver-free) |

## Web portal (runs on any machine)

A local browser UI to drive the detector three ways. Start it:

```bash
pip install -r requirements.txt      # includes flask
python app.py                        # serves http://127.0.0.1:5000
```

On first launch with no `model.joblib`, the portal bootstraps one from the
bundled fixtures automatically, so it works out of the box. Three modes:

1. **Real LLM test.** Fire a **genuine streaming** chat completion and watch
   it get flagged from its shape alone. Works with any OpenAI-compatible
   endpoint (OpenAI, Groq, OpenRouter, Together, LM Studio, vLLM, **Vercel AI
   Gateway**), native **Anthropic**, or local **Ollama** (no key). Paste a base
   URL, model, and key in the form, or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
   in the environment.

   > **Honest note on "real":** live packet sniffing needs a capture driver
   > (Npcap) and admin rights, which is not "any machine". So instead of sniffing
   > raw packets, the portal makes a **real** streaming API call and measures the
   > **real** thing this detector cares about: each token chunk's **arrival time
   > and byte size**. It lays those real timings and sizes down as a TCP flow and
   > runs the identical pcap-parse + model pipeline. The *shape* is real, measured
   > from a real LLM; only the packet framing around it is reconstructed. We
   > never claim to have sniffed raw packets.

2. **Simulated.** Build synthetic AI / normal / mixed flows and run the real
   pcap-parse + model pipeline on them. No API key needed. (Add `?demo=1` to the URL
   to auto-run this, a handy shareable demo link.)

3. **Analyze a pcap.** Upload your own `.pcap`/`.pcapng`, or pick a bundled
   fixture, and see which flows are flagged.

Every mode shows, per flow: the LLM/normal verdict, a confidence bar, and the
shape stats that drove it, with the peer IP/port shown only as *"reference
only, not used to decide."* The same `FEATURE_COLS`-only rule holds everywhere.

> **Fonts.** The UI uses the PP Mondwest pixel display font (Pangram Pangram) for
> the title and badges. It is bundled at `static/PPMondwest-Regular.otf`; if that
> file is missing, the UI falls back to a monospace font.

## Features (the model's entire input surface)

All shape, zero identifiers:

- `duration`: first-to-last packet time of the flow (seconds)
- `n_packets`, `n_down`, `n_up`: packet counts (total / downstream / upstream)
- `down_up_byte_ratio`: downstream bytes over upstream bytes
- `down_size_mean`, `down_size_std`: downstream packet size stats (bytes)
- `frac_small_down`: fraction of downstream packets smaller than 300 B
- `down_pps`: downstream packets per second
- `down_iat_mean`, `down_iat_std`, `down_iat_max`, `down_iat_cv`: downstream
  inter-arrival timing (mean / std / max / coefficient of variation)

Flows with fewer than ~8 packets are dropped (too little signal).

---

## Setup

Just Python 3.9+ and the deps:

```bash
pip install -r requirements.txt
```

That is the whole install. pcaps are parsed in **pure Python** (scapy), so there
is no system binary such as tshark/Wireshark to install; it runs the same way
locally, on a fresh machine, and serverless.

### Deploy on Vercel

The repo ships a `vercel.json` that serves `app.py` (a Flask WSGI app) as a
single Python serverless function and routes every request to it, plus a
committed `model.joblib`, so it deploys as-is:

```bash
vercel login   # once
vercel --prod  # deploy the current directory
```

No tshark, no training step at deploy time. The bundled fixtures are not shipped
to the deploy (regenerable), so the "Analyze a pcap" fixture list is empty there;
"Simulated", "Real LLM test", and pcap upload all work.

---

## Capturing two clean, labeled pcaps

Labels here come from **capturing each class on its own** (see Limitations). The
goal is one pcap that contains *only* AI traffic and one that contains *only*
normal traffic.

**Capture AI-only.** Close everything else that talks to the network (other tabs,
sync clients, chat apps). Then start the capture and do nothing but hold a
streaming chat session with an LLM for a minute or two:

```bash
# swap eth0 for your interface (see `tcpdump -D`)
sudo tcpdump -i eth0 -w ai_session.pcap 'tcp'
# ...now open ONE LLM chat and stream several responses, then Ctrl-C
```

**Capture normal-only.** Fresh capture, no LLM this time. Browse a few pages,
hit some normal APIs, download a file:

```bash
sudo tcpdump -i eth0 -w normal_session.pcap 'tcp'
# ...browse / use non-AI apps for a minute or two, then Ctrl-C
```

Keep them clean: the cleaner the separation, the better the labels. On Windows
you can capture with `dumpcap -i <n> -w ai_session.pcap` instead.

---

## The 4-step run flow

```bash
# 1. AI capture      -> features CSV (label comes from the file, not the flow)
python extract_features.py ai_session.pcap ai.csv

# 2. Normal capture  -> features CSV
python extract_features.py normal_session.pcap normal.csv

# 3. Train + evaluate (5-fold, out-of-sample) + save model.joblib
python train.py ai.csv normal.csv

# 4. Point it at a FRESH capture and see what it flags
python detect.py some_new_capture.pcap
```

### Try it right now with synthetic fixtures (no capture needed)

```bash
python make_fixtures.py                                     # writes fixtures/*.pcap
python extract_features.py fixtures/ai_sample.pcap     ai.csv
python extract_features.py fixtures/normal_sample.pcap normal.csv
python train.py ai.csv normal.csv
python detect.py fixtures/mixed_capture.pcap
```

On the synthetic data the classes are cleanly separable, so `train.py` reports a
perfect 5-fold confusion matrix and `detect.py` flags all 6 AI flows in the mixed
capture (and none of the 6 normal ones). Top features are `down_iat_max` and
`down_pps`, i.e. the trickle timing. **Real traffic will not be this clean**; the
fixtures are teaching examples, not a benchmark.

---

## Limitations (honest)

- **Streaming is the easy case.** A token-by-token streamed response has a loud,
  distinctive shape. A **non-streaming** LLM API call (one request, one buffered
  JSON response) looks like any other short REST call and this shape-only method
  will miss it. Catching those would need extra signals bolted on, namely **JA4/JA4S
  TLS fingerprinting** plus **IP/ASN reputation**, which reintroduces
  identifiers and is out of scope here by design.
- **Labels come from capturing classes separately.** We label every flow in
  `ai.csv` as AI and every flow in `normal.csv` as normal. If your "AI" capture
  has background chatter (telemetry, an open email tab), those flows get
  mislabeled. Capture clean, or filter afterward. This is weaker than true
  per-flow ground truth.
- **Synthetic fixtures are idealized.** They exist to prove the pipeline runs
  end-to-end, not to prove real-world accuracy. Expect messier features and
  lower scores on live captures, and retrain on your own traffic.
- **Metadata-only, by design.** It reads packet timing and sizes and never
  decrypts, never reassembles payloads, never resolves names. That is a feature
  (privacy-preserving, ECH-proof), but it also caps how much it can ever know:
  it can say "this flow *looks like* token streaming," not "this is model X."
- **Evadable.** An adversary who pads packets to uniform size and paces them
  regularly can blur the shape. Shape-based detection and shape-based evasion are
  a cat-and-mouse game; this is the opening move, not the last word.
- **IPv4 TCP only** in this prototype (the parser keeps IPv4 TCP packets only). QUIC /
  HTTP-3 (UDP) and IPv6 are future work, and QUIC in particular is where a lot
  of real LLM traffic is heading.
