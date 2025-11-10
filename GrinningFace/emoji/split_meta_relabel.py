#!/usr/bin/env python3
"""Split a meta_relabel_dedup.json file into train/val by the natural load order.

Rule (ignore any existing 'no' field):
    enumerate(data): index i even -> train, odd -> val

Outputs: <input_stem>_train.json and <input_stem>_val.json

Usage:
    python split_meta_relabel.py --input simulation/emoji/meta_relabel_dedup.json [--overwrite]
"""
import json
import argparse
from pathlib import Path


def split_file(input_path: Path, overwrite: bool = False):
    assert input_path.exists(), f"Input file not found: {input_path}"
    data = json.loads(input_path.read_text(encoding='utf-8'))
    train = [item for i, item in enumerate(data) if i % 2 == 0][:100]
    val   = [item for i, item in enumerate(data) if i % 2 == 1][:100]

    out_train = input_path.with_name(f"{input_path.stem}_train{input_path.suffix}")
    out_val = input_path.with_name(f"{input_path.stem}_val{input_path.suffix}")

    if not overwrite:
        for p in (out_train, out_val):
            if p.exists():
                raise FileExistsError(f"Output exists: {p} (use --overwrite)")

    out_train.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding='utf-8')
    out_val.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Total: {len(data)}  Train(even idx): {len(train)}  Val(odd idx): {len(val)}")
    print(f"Train -> {out_train}\nVal   -> {out_val}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to meta_relabel_dedup.json')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = ap.parse_args()

    split_file(Path(args.input), overwrite=args.overwrite)

if __name__ == '__main__':
    main()
