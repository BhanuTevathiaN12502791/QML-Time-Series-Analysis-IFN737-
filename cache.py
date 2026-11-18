"""Bind resumable jobs to frozen inputs and verify their complete output files."""
import hashlib,json,platform
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy,pandas,sklearn
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(site,phase):
    protocol=json.loads((ROOT/'protocol.json').read_text())
    expected=next(r['sha256'] for r in protocol['sites'] if r['site']==site)
    actual=sha(ROOT/'data'/f'{site}.csv')
    if actual!=expected:raise ValueError(f'{site}: source data no longer match the frozen snapshot')
    if sha(ROOT/'protocol.json')!=(ROOT/'protocol.sha256').read_text().strip():raise ValueError('Frozen protocol digest mismatch')
    return dict(site=site,phase=phase,source_sha256=actual,protocol_sha256=sha(ROOT/'protocol.json'),
        fitting_code={name:sha(ROOT/name) for name in ['core.py','experiment.py','cache.py']},
        selection_sha256=sha(ROOT/'selected_parameters.json') if phase!='development' else None,
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__))

def paths(site,phase):
    folder=ROOT/'results'/phase
    if phase=='development':return [folder/f'{site}.csv']
    return [folder/f'{site}_{kind}.csv' for kind in ['metrics','splits']]+[folder/f'{site}_fold{f}.npz' for f in range(3)]

def stamp(site,phase,origin='recorded_at_job_completion',expected=None):
    current=fingerprint(site,phase)
    if expected is not None and current!=expected:raise ValueError('Inputs changed while the job was running')
    outputs=paths(site,phase)
    if not all(p.is_file() for p in outputs):raise ValueError(f'{site}: incomplete result files')
    metadata=dict(fingerprint=current,output_sha256={p.name:sha(p) for p in outputs},binding_origin=origin,
        recorded_at=datetime.now(ZoneInfo('Australia/Brisbane')).isoformat())
    target=ROOT/'results'/phase/f'{site}_cache.json';temporary=target.with_suffix('.part')
    temporary.write_text(json.dumps(metadata,indent=2));temporary.replace(target)

def reusable(site,phase):
    outputs=paths(site,phase)
    if not any(p.exists() for p in outputs):return False
    metadata=ROOT/'results'/phase/f'{site}_cache.json'
    if not metadata.exists():raise ValueError(f'{site}: unbound legacy/partial cache. Back up and remove that site’s cached outputs before rerunning.')
    record=json.loads(metadata.read_text())
    if record['fingerprint']!=fingerprint(site,phase):raise ValueError(f'{site}: cached inputs/code/environment changed. Back up results and run a fresh experiment.')
    if not all(p.exists() and record['output_sha256'].get(p.name)==sha(p) for p in outputs):
        raise ValueError(f'{site}: cached outputs are missing or changed. Back up results and recompute the site.')
    return True
