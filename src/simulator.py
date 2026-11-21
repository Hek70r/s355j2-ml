"""Trans_e input and current-process parsing. Never reuse simulator outputs."""
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from src.config import CONFIG, T_START
from src.curves import validate_stages


def _fmt(value):
    # Round-trip binary doubles rather than the former six-significant-digit truncation.
    return format(float(value), '.17g')


def trans_dat_text(stages, cfg=CONFIG, minimum_steps=1_000_000):
    validate_stages(stages)
    if sum(float(t) for t, _ in stages) > cfg.integration_step_s*cfg.max_steps:
        raise ValueError('curve exceeds Trans_e integration budget')
    # Keep the historical minimum: smaller budgets caused observed Windows hangs.
    # Long curves can still request more than one million steps.
    steps = math.ceil(sum(duration for duration, _ in stages) / cfg.integration_step_s) + 2
    steps = max(steps, minimum_steps)
    if steps > cfg.max_steps:
        raise ValueError('curve exceeds Trans_e integration budget')
    lines = ['0 12 - one case only; i - calculated i cooling rates', '.false.',
             '3      model', f'{steps} number of time steps',
             f'{_fmt(T_START)} initial temperature', f'{_fmt(cfg.integration_step_s)} time step',
             '.True. printing the results on the screen', '.False. writing to the files',
             '.False. (without heating) .True. (with heating)', '1 number of cycles',
             '---------------------------------------------------------------',
             f'{len(stages)} number of stages time cr']
    lines.extend(f'{_fmt(t)} {_fmt(r)}' for t, r in stages)
    lines.append('---------------------------------------------------------------')
    return '\r\n'.join(lines)+'\r\n'


NUMBER = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)'
FIELDS = {'F': 'volume fraction of ferrite', 'P': 'volume fraction of pearlite',
          'B': 'volume fraction of bainite', 'M': 'volume fraction of martensite',
          'RA': 'volume fraction of retained austenite', 'cementite': 'volume fraction of cementite',
          'tempered_M': 'including tempered martensite', 'transformed_fraction': 'transformed fraction'}


def parse_stdout(text):
    result = {}
    # One final block only: duplicate blocks are ambiguous and must not create mixed labels.
    for key, label in FIELDS.items():
        matches = re.findall(r'^\s*'+label+r'\s+'+NUMBER+r'\s*$', text, re.MULTILINE | re.IGNORECASE)
        if len(matches) > 1:
            raise ValueError('ambiguous duplicate result blocks')
        value = float(matches[0].replace('D', 'E').replace('d', 'e')) if matches else None
        result[key] = 0.0 if value == 0 else value
    return result


def qualify_result(result, cfg=CONFIG):
    """Qualify before correcting rounding; significant RA is never folded into M."""
    out = dict(result)
    out.update(status='invalid_phases', error_reason='', rounding_correction=0.)
    out.update({p: '' for p in ('F', 'P', 'B')})
    for key in FIELDS:
        out[key+'_raw'] = result.get(key, '')
    required = (*'FPBM', 'RA', 'cementite', 'transformed_fraction')
    if any(result.get(k) is None or not math.isfinite(result[k]) or not 0 <= result[k] <= 1 for k in required):
        out['error_reason'] = 'missing_or_invalid_fraction'
    elif result['RA'] >= cfg.reporting_quantum/2:
        out['error_reason'] = 'retained_austenite'
    elif result['cementite'] >= cfg.reporting_quantum/2:
        out['error_reason'] = 'unmodelled_cementite'
    elif abs(result['transformed_fraction']-1) > cfg.reporting_quantum/2 + 1e-12:
        out['error_reason'] = 'incomplete_transformation'
    else:
        total = sum(result[p] for p in 'FPBM')
        if abs(total-1) > cfg.sum_tolerance + 1e-12:
            out['error_reason'] = 'incomplete_phase_sum'
        else:
            normalized = [result[p]/total for p in 'FPBM']
            out.update(status='ok', rounding_correction=max(abs(a-result[p]) for a,p in zip(normalized, 'FPBM')),
                       **dict(zip('FPB', normalized[:3])))
    return out


def read_volume_fractions(path):
    try:
        lines = Path(path).read_text(errors='replace').splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        parts = line.split()
        if len(parts) == 4:
            try:
                return [float(v.replace('D', 'E').replace('d', 'e')) for v in parts]
            except ValueError:
                pass
    return None


def run_once(stages, program_dir, cfg=CONFIG, minimum_steps=1_000_000, workdir=None):
    """Reuse campaign runtime files; always clear outputs before a new process."""
    input_text = trans_dat_text(stages, cfg, minimum_steps)
    started = time.perf_counter()
    out = dict(status='execution_error', error_reason='', stdout='', stderr='',
               input_text=input_text, exit_code='', runtime_s=0.,
               **{k+'_raw': '' for k in FIELDS}, **{p: '' for p in 'FPB'})
    program_dir = Path(program_dir).resolve()
    temporary = None
    work = None
    try:
        if workdir is None:
            temporary = tempfile.TemporaryDirectory(prefix='trans_e_', ignore_cleanup_errors=True)
            work = Path(temporary.name)
        else:
            work = Path(workdir).resolve()
            if work == program_dir:
                raise ValueError('working directory must differ from the original program directory')
            work.mkdir(parents=True, exist_ok=True)
        runtime_files = [program_dir/'Trans_e.exe', program_dir/'coeff.dat', *program_dir.glob('*.dll')]
        for source in runtime_files:
            destination = work/source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                # Copy runtime once, without inherited read-only attributes.
                shutil.copyfile(source, destination)
        if (work/'coeff.dat').read_bytes() != (program_dir/'coeff.dat').read_bytes():
            raise ValueError('working coefficients differ from program/coeff.dat')
        # This is our dedicated directory, containing only runtime and outputs.
        # Remove prior outputs so no old result or growing append file is reused.
        for path in work.iterdir():
            if path.is_file() and path.name != 'coeff.dat' and path.suffix.lower() not in ('.exe', '.dll'):
                path.unlink()
        (work/'trans.dat').write_bytes(input_text.encode('ascii'))
        (work/'VolumeFractions.dat').touch()
        (work/'a.dat').touch()
        proc = subprocess.run([str(work/'Trans_e.exe')], cwd=work,
                              capture_output=True, text=True, errors='replace', timeout=cfg.timeout_s)
        out.update(stdout=proc.stdout or '', stderr=proc.stderr or '', exit_code=proc.returncode)
        if proc.returncode:
            out['error_reason'] = 'nonzero_exit'
        else:
            parsed = parse_stdout(out['stdout'])
            qualified = qualify_result(parsed, cfg)
            out.update(qualified)
            vf = read_volume_fractions(work/'VolumeFractions.dat')
            if out['status'] == 'ok' and (vf is None or any(not math.isfinite(v) for v in vf)):
                out.update(status='execution_error', error_reason='missing_fresh_volume_row')
            elif out['status'] == 'ok' and max(abs(v-parsed[p]) for v,p in zip(vf, 'FPBM')) > cfg.sum_tolerance:
                out.update(status='execution_error', error_reason='volume_row_mismatch')
    except subprocess.TimeoutExpired as exc:
        out.update(error_reason='timeout', stdout=(exc.stdout or b'').decode(errors='replace')
                   if isinstance(exc.stdout, bytes) else (exc.stdout or ''), stderr='timeout')
    except (OSError, ValueError) as exc:
        reason = 'disk_full' if isinstance(exc, OSError) and (exc.errno == 28 or getattr(exc, 'winerror', None) == 112) else str(exc)
        out.update(status='execution_error', error_reason=reason)
    finally:
        if temporary is not None:
            # Single-call tests may use Temp. Campaigns keep one reusable directory.
            cleanup_error = ''
            try:
                temporary.cleanup()
            except OSError as exc:
                cleanup_error = str(exc)
            if cleanup_error or work.exists():
                warning = f'[wrapper cleanup warning] Temporary directory: {work}'
                if cleanup_error:
                    warning += f'; {cleanup_error}'
                out['stderr'] = (out['stderr']+'\n'+warning).lstrip('\n')
    out['runtime_s'] = time.perf_counter()-started
    if out['status'] != 'ok':
        out.update({p: '' for p in 'FPB'})
    return out
