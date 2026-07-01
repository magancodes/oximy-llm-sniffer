"""
extract_features.py <pcap> <out.csv>

Thin CLI wrapper: parse a pcap into per-flow features and write them to CSV.
The CSV contains FEATURE_COLS (the model's shape-only inputs) plus META_COLS
(client/server identifiers kept for human reference only — see features.py for
why they are NOT features).
"""

import sys

from features import FEATURE_COLS, META_COLS, extract_features


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python extract_features.py <pcap> <out.csv>", file=sys.stderr)
        return 2

    pcap_path, out_csv = sys.argv[1], sys.argv[2]
    df = extract_features(pcap_path)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {len(df)} flow row(s) -> {out_csv}")
    print(f"  feature columns ({len(FEATURE_COLS)}, all shape, zero identifiers): "
          f"{FEATURE_COLS}")
    print(f"  meta columns (identifiers, NOT used by the model): {META_COLS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
