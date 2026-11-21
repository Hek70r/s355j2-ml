"""Draw the thesis figures from the published dataset and result tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from plots import methods, results
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/dataset.csv'))
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('output/figures'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.data.with_name('generator.json').read_text())
    with args.data.open(newline='', encoding='utf-8') as source:
        methods.draw(list(csv.DictReader(source)), metadata, load_config(args.data.with_name('config.json')), args.out)
    dataset = pd.read_csv(args.data, float_precision='round_trip').set_index('curve_id')
    results.draw(dataset, metadata, args.results, args.out)


if __name__ == '__main__':
    main()
