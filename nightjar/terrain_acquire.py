"""Acquire aligned real heights for the reviewed geographic incident family.

Uses IGN/CNIG WCS in Spain and Copernicus GLO-30 DSM outside Spain. Rasterio
is optional until DSM acquisition is requested; no interpolated/fabricated gaps.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

from .environment_family import _assert_family_grids, _dem_url
from .geographic_ppo import load_geographic_split
from .geodesy import utm_to_wgs84
from .pilot_terrain import _parse_ascii


def acquire_terrain(scenarios_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('terrain output must be new or empty')
    scenarios=[]
    for split in ('development','validation','reserved'):
        scenarios.extend(load_geographic_split(split,scenarios_dir))
    groups={a:[s for s in scenarios if s['activation_id']==a] for a in sorted({s['activation_id'] for s in scenarios})}
    output_dir.mkdir(parents=True,exist_ok=True)
    sources={}
    for activation, group in groups.items():
        grid=_assert_family_grids(group,activation)
        rows,cols,size,x0,y0,crs=grid
        path=output_dir/(activation.lower()+'.asc')
        if activation in ('EMSR578','EMSR590'):
            url=_dem_url(activation,grid)
            subprocess.run(['curl','--fail','--silent','--show-error','--max-time','50',url,'-o',str(path)],check=True)
            header,values=_parse_ascii(path)
            if (header['nrows'],header['ncols'])!=(rows,cols) or any(v is None for row in values for v in row):
                raise ValueError('IGN WCS did not return a complete aligned elevation grid')
            source={'provider':'IGN/CNIG','dataset':'Elevacion25830_25','native_resolution_m':25,
                    'vertical_reference':'orthometric','acquisition_epoch':None,'url':url,
                    'source_footprint':'unknown' if 'nodata_value' not in header else 'nodata_encoded',
                    'resampling':'WCS service default; not specified in request','surface':'terrain',
                    'license_url':'https://www.ign.es/web/ign/portal/politica-datos'}
        else:
            import rasterio
            from rasterio.vrt import WarpedVRT
            from rasterio.enums import Resampling
            from rasterio.transform import Affine
            lon,lat=utm_to_wgs84(x0+cols*size/2,y0+rows*size/2,zone=group[0]['crs']['utm_zone'])
            tx,ty=math.floor(lon),math.floor(lat)
            tile=f'Copernicus_DSM_COG_10_{"N" if ty>=0 else "S"}{abs(ty):02d}_00_{"E" if tx>=0 else "W"}{abs(tx):03d}_00_DEM'
            url=f'https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif'
            with rasterio.Env(GDAL_HTTP_TIMEOUT='50'):
                with rasterio.open(url) as ds:
                    with WarpedVRT(ds,crs=crs,transform=Affine(size,0,x0,0,-size,y0+rows*size),
                                   width=cols,height=rows,resampling=Resampling.nearest,
                                   nodata=-9999,src_nodata=ds.nodata) as vrt:
                        values=vrt.read(1,masked=True)
                    if values.mask.any() or not __import__('numpy').isfinite(values).all():
                        raise ValueError('Copernicus DSM crop has uncovered or invalid cells')
                    lines=[f'ncols {cols}',f'nrows {rows}',f'xllcorner {x0}',f'yllcorner {y0}',f'cellsize {size}']
                    lines.extend(' '.join(f'{v:.6f}' for v in row) for row in values)
                    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
            source={'provider':'Copernicus DEM','dataset':'GLO-30 Public DSM','native_resolution_m':30,
                    'vertical_reference':'EGM2008 orthometric','acquisition_epoch':'2011–2015 main acquisition; edited dataset',
                    'url':url,'tile':tile,'source_footprint':'complete crop checked by raster mask',
                    'resampling':'nearest-neighbor','surface':'DSM includes vegetation and buildings',
                    'license_url':'https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM',
                    'attribution':'produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved'}
        source.update(raw_sha256=sha256(path.read_bytes()).hexdigest(),crs=crs,
                      crs_assertion='recorded from acquisition request/target reprojection',
                      retrieved_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'))
        sources[activation]=source
        manifest={**sources,'schema_version':'nightjar-terrain-sources-0.1','seed':None,
                  'units':{'coordinates':'m','elevation':'m'},
                  'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')}
        (output_dir/'sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        print(f'{activation}: {source["dataset"]} {rows}×{cols}',flush=True)
    return sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();acquire_terrain(a.scenarios_dir,a.output_dir)

if __name__=='__main__':main()
