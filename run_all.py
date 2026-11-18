"""Run the frozen experiment, analyses and report; completed site jobs are reused."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def missing_source_files():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    return [ROOT/'data'/f'{row["site"]}.csv' for row in protocol['sites'] if not (ROOT/'data'/f'{row["site"]}.csv').is_file()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true',help='Download and hash-verify the pinned source data if files are missing')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if missing_source_files():
        if not args.fetch:parser.error('Source data are missing. Run python fetch_data.py first, or rerun with --fetch.')
        subprocess.run([sys.executable,str(ROOT/'fetch_data.py')],check=True)
        if missing_source_files():parser.error('Source acquisition did not provide all required CSV files')
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    for phase in ['tune','evaluate','normal']:
        subprocess.run([sys.executable,str(ROOT/'experiment.py'),phase,'--workers',str(args.workers)],check=True)
    subprocess.run([sys.executable,str(ROOT/'train_site.py'),'--model','sitdd_adapted','--out',str(ROOT/'models/0185-E_sitdd')],check=True)
    for script in ['analyse.py','benchmark.py','train_site.py','make_report.py','verify_results.py']:
        subprocess.run([sys.executable,str(ROOT/script)],check=True)


if __name__=='__main__':main()
