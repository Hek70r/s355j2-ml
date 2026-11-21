"""Recreate the F1–F4 cooling stages, optionally labelling them with Trans_e on Windows."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

from src.config import load_config
from src.cooling_generator import generate
from src.curves import canonical_stages
from src.simulator import run_once, trans_dat_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('data/config.json'))
    parser.add_argument('--out', type=Path, default=Path('output/curves.csv'))
    parser.add_argument('--program', type=Path, help='Directory containing Trans_e.exe, coeff.dat and its DLLs')
    parser.add_argument('--example-input', action='store_true', help='Print trans.dat for curve 5898 (Fig. 5)')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.example_input:
        with args.config.with_name('dataset.csv').open(newline='') as source:
            example = next(row for row in csv.DictReader(source) if row['curve_id'] == '5898')
        print(trans_dat_text(json.loads(example['stages_json']), config), end='')
        return
    if args.program and not (args.program / 'Trans_e.exe').is_file():
        parser.error('Missing Trans_e.exe in the specified program directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['family', 'stages_json']
    if args.program:
        fields += ['status', 'error_reason', 'F', 'P', 'B', 'M', 'RA']
    with args.out.open('x', newline='', encoding='utf-8') as target, tempfile.TemporaryDirectory() as work:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for family, stages in generate(config.power, config.seed):
            stages = canonical_stages(stages)
            row = dict(family=family, stages_json=json.dumps(stages))
            if args.program:
                result = run_once(stages, args.program, config, workdir=work)
                for field in ('status', 'error_reason', 'F', 'P', 'B'):
                    row[field] = result[field]
                row['RA'] = result['RA_raw']
                if result['status'] == 'ok':
                    row['M'] = 1 - sum(row[phase] for phase in 'FPB')
            writer.writerow(row)


if __name__ == '__main__':
    main()
