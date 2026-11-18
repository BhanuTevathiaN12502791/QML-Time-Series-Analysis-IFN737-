"""Acquire and verify the exact frozen AnoLT CSV snapshot."""
import argparse,hashlib,json,subprocess,tempfile,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default=str(ROOT/'data'));args=parser.parse_args()
    p=json.loads((ROOT/'protocol.json').read_text());dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    expected={r['site']+'.csv':r for r in p['sites']}
    with tempfile.TemporaryDirectory() as td:
        archive=Path(td)/'source.zip';url=f'https://codeload.github.com/imaantaheri/AnoLT/zip/{p["snapshot"]}'
        subprocess.run(['curl','--fail','-L','--retry','3','-sS',url,'-o',str(archive)],check=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                name=Path(member).name
                if '/data/' not in member or name not in expected:continue
                raw=z.read(member);assert hashlib.sha256(raw).hexdigest()==expected[name]['sha256'],name
                (dest/name).write_bytes(raw)
    assert all((dest/name).exists() for name in expected)
    print('Acquired and SHA-256 verified all',len(expected),'CSV files')
if __name__=='__main__':main()
