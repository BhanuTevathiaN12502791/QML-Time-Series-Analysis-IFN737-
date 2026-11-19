"""Download verified 2025 SCATS archives and extract the study's two detectors."""
import argparse,csv,hashlib,json,subprocess,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent

def extract_archive(path,lanes):
    wanted={tuple(map(int,lane.split(':'))) for lane in lanes};prefix=tuple(f'{s},'.encode() for s,_ in wanted);points=[];daily=[]
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith('.csv'):continue
            with z.open(name) as stream:
                header=next(csv.reader([stream.readline().decode('utf-8-sig')]))
                for line in stream:
                    if not line.startswith(prefix):continue
                    r=dict(zip(header,next(csv.reader([line.decode()]))));key=int(r['NB_SCATS_SITE']),int(r['NB_DETECTOR'])
                    if key not in wanted:continue
                    lane=f'{key[0]}:{key[1]}';day=pd.Timestamp(r['QT_INTERVAL_COUNT']).tz_localize('Etc/GMT-10');records=int(r['CT_RECORDS']);alarm=int(r['CT_ALARM_24HOUR']);clean=records==96 and alarm==0
                    daily.append(dict(lane_id=lane,date=str(day.date()),records=records,alarm_count=alarm,daily_total=r['QT_VOLUME_24HOUR'],usable=clean))
                    for i in range(96):
                        value=pd.to_numeric(r[f'V{i:02d}'],errors='coerce')
                        points.append(dict(lane_id=lane,timestamp=day+pd.Timedelta(minutes=15*(i+1)),volume=float(value) if clean and pd.notna(value) and value>=0 else None))
    return daily,points

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',default=str(ROOT/'work/event_cache'));p.add_argument('--out',default=str(ROOT/'data_events'));p.add_argument('--keep-archives',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'event_data_manifest.json').read_text());cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    daily=[];points=[]
    for resource in manifest['sources']:
        path=cache/resource['url'].rsplit('/',1)[-1];downloaded=False
        if not path.exists():
            temporary=path.with_suffix('.part');subprocess.run(['curl','--fail','-L','--retry','3','-sS',resource['url'],'-o',str(temporary)],check=True);temporary.replace(path);downloaded=True
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=resource['sha256']:raise ValueError(f'Upstream archive changed: {path.name}; investigate rather than mixing data versions')
        d,s=extract_archive(path,manifest['lanes']);daily.extend(d);points.extend(s);print(path.name,len(d),'detector-days',flush=True)
        if downloaded and not a.keep_archives:path.unlink()
    frame=pd.DataFrame(points).sort_values(['lane_id','timestamp'])
    if frame.duplicated(['lane_id','timestamp']).any():raise ValueError('Duplicate source lane/timestamps')
    if set(frame.lane_id)!=set(manifest['lanes']):raise ValueError('Requested detector absent')
    frame.to_csv(out/'lane_series.csv',index=False);pd.DataFrame(daily).sort_values(['lane_id','date']).to_csv(out/'daily_quality.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2));print('Saved',len(frame),'intervals',flush=True)
if __name__=='__main__':main()
