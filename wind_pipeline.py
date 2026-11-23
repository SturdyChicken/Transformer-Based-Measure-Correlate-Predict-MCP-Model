"""Wind loaders, leakage-safe training preparation, and shared model inference.

AZMet format: https://azmet.arizona.edu/azmet/raw2003.htm
Missing values / intervals: https://azmet.arizona.edu/azmet/data.html
Clock: https://azmet.arizona.edu/azmet/doy.htm
ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests

SITES = ('az06', 'az05', 'az22', 'az26', 'az19', 'az15')
START, END = '2005-01-01', '2019-12-31'
TZ = 'America/Phoenix'
ARCHIVES = {s: f'https://azmet.arizona.edu/azmet/{s[2:]}.htm' for s in SITES}
WIND_COLUMNS = ['site_id', 'time_utc', 'height_m', 'speed_mps', 'direction_deg',
                'vector_speed_mps', 'speed_valid', 'direction_valid']
ERA_VARIABLES = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                 '100m_u_component_of_wind', '100m_v_component_of_wind']
ERA_COLUMNS = ['u10_mps', 'v10_mps', 'u100_mps', 'v100_mps']


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf-8')


def download_original(url, path, session=None):
    """Immutable cache, with checksum and origin; never overwrite a downloaded file."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.provenance.json')
    if path.exists():
        if not sidecar.exists():
            raise ValueError(f'Untracked cache file: {path}')
        info = json.loads(sidecar.read_text(encoding='utf-8'))
        if info['url'] != url or info['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Cache provenance mismatch: {path}')
        return path
    response = (session or requests).get(url, timeout=90)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(response.content)
    save_json(sidecar, {'url': url, 'resolved_url': response.url,
                       'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                       'sha256': hashlib.sha256(response.content).hexdigest(),
                       'bytes': len(response.content)})
    return path


class ArchiveParser(HTMLParser):
    """Read table rows and their link labels, including legacy HTML without </td>."""
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.anchor = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            if self.row:
                self.rows.append(self.row)
            self.row = {'text': '', 'links': []}
        if tag == 'a' and self.row is not None:
            self.anchor = {'href': dict(attrs).get('href', ''), 'text': ''}

    def handle_data(self, data):
        if self.row is not None:
            self.row['text'] += ' ' + data
        if self.anchor is not None:
            self.anchor['text'] += data

    def handle_endtag(self, tag):
        if tag == 'a' and self.anchor is not None:
            if self.row is not None:
                self.row['links'].append(self.anchor)
            self.anchor = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def discover_hourly_links(html, archive_url, years=range(2005, 2020)):
    """Select explicit 'Hourly' anchors from year rows; do not construct data URLs."""
    parser = ArchiveParser()
    parser.feed(html)
    records = []
    for row in parser.rows:
        year_match = re.match(r'\s*(20\d{2})\b', row['text'])
        if not year_match or int(year_match[1]) not in years:
            continue
        links = {a['text'].strip().lower(): urljoin(archive_url, a['href'])
                 for a in row['links']}
        if 'hourly' in links:
            records.append({'year': int(year_match[1]), 'url': links['hourly'],
                            'notes_url': links.get('notes')})
    # Queen Creek's archive repeats an identical 2011 row. Collapse identical
    # page links only; conflicting links for a year still fail below.
    result = pd.DataFrame(records, columns=['year', 'url', 'notes_url']).drop_duplicates()
    if result.empty or result.year.duplicated().any():
        raise ValueError(f'No unambiguous annual hourly links at {archive_url}')
    return result.sort_values('year').reset_index(drop=True)


def acquire_azmet(root, sites=SITES, years=range(2005, 2020)):
    """Download originals plus archive/metadata/notes evidence; log missing links and 404s."""
    root = Path(root)
    downloads, notes, issues = [], [], []
    with requests.Session() as session:
        for site in sites:
            archive = download_original(ARCHIVES[site], root / 'sources' / f'{site}_archive.html', session)
            metadata_url = f'https://azmet.arizona.edu/about/station-metadata/{site}'
            try:
                download_original(metadata_url, root / 'sources' / f'{site}_metadata.html', session)
            except requests.HTTPError as exc:
                issues.append({'site_id': site, 'kind': 'metadata_http', 'detail': str(exc)})
            links = discover_hourly_links(archive.read_text(encoding='utf-8', errors='replace'), ARCHIVES[site], years)
            for year in sorted(set(years) - set(links.year)):
                issues.append({'site_id': site, 'year': year, 'kind': 'hourly_link_absent'})
            for row in links.to_dict('records'):
                name = Path(urlparse(row['url']).path).name
                path = root / 'azmet' / site / name
                try:
                    download_original(row['url'], path, session)
                    downloads.append({'site_id': site, **row, 'path': str(path)})
                except requests.HTTPError as exc:
                    issues.append({'site_id': site, 'year': row['year'], 'kind': 'hourly_http', 'detail': str(exc)})
                if row['notes_url']:
                    note_path = root / 'sources' / site / Path(urlparse(row['notes_url']).path).name
                    try:
                        download_original(row['notes_url'], note_path, session)
                        text = note_path.read_text(encoding='utf-8', errors='replace')
                        notes.append({'site_id': site, 'year': row['year'], 'source': row['notes_url'],
                                      'text': text, 'mentions_estimates': bool(re.search(r'estimat', text, re.I)),
                                      'mentions_change': bool(re.search(r'sensor|anemometer|relocat|replace|instrument', text, re.I))})
                    except requests.HTTPError as exc:
                        issues.append({'site_id': site, 'year': row['year'], 'kind': 'notes_http', 'detail': str(exc)})
    manifest = pd.DataFrame(downloads)
    manifest.to_csv(root / 'azmet_manifest.csv', index=False)
    save_json(root / 'source_issues.json', issues)
    save_json(root / 'station_notes.json', notes)
    return manifest, pd.DataFrame(notes), pd.DataFrame(issues)


def html_text(html):
    return unescape(re.sub(r'<[^>]+>', ' ', html))


def load_metadata(root, sites=SITES):
    """Use archive locations; assume current 3 m height throughout the study."""
    rows = []
    for site in sites:
        text = html_text((Path(root) / 'sources' / f'{site}_archive.html').read_text(encoding='utf-8', errors='replace'))
        coords = re.search(r'Coordinates\s*:\s*([-\d.]+)\s*,\s*([-\d.]+)', text, re.I)
        elevation = re.search(r'Elevation\s*:\s*(\d+)\s*meters', text, re.I)
        meta_path = Path(root) / 'sources' / f'{site}_metadata.html'
        current = html_text(meta_path.read_text(encoding='utf-8', errors='replace')) if meta_path.exists() else ''
        wind_section = current.split('DIRECTION SENSOR:')[-1] if 'DIRECTION SENSOR:' in current else ''
        height = re.search(r'SPEED SENSOR HEIGHT\s*:\s*[\d.]+\s*ft\s*\(\s*([\d.]+)\s*m\s*\)', wind_section)
        history = ('Queen Creek moved before study; legacy page says 2002-present, modern page starts 2003-01-01. '
                   if site == 'az22' else 'No previous location listed on current metadata page; this is not a complete event log. ')
        history += 'Dated 2005-2019 instrument/height history unverified; inspect archived annual notes.'
        rows.append(dict(site_id=site, latitude_deg=float(coords[1]) if coords else np.nan,
                         longitude_deg=float(coords[2]) if coords else np.nan,
                         ground_elevation_m=float(elevation[1]) if elevation else np.nan,
                         measurement_height_m=3.0, source=ARCHIVES[site], data_kind='AZMet hourly summary',
                         timestamp_convention='time_utc = end of preceding hour; original hours 1..24, MST UTC-07',
                         station_history=history, era5_grid_latitude=np.nan, era5_grid_longitude=np.nan,
                         current_documented_height_m=float(height[1]) if height else np.nan,
                         height_source=f'https://azmet.arizona.edu/about/station-metadata/{site}',
                         height_status='assumed 3 m throughout 2005-2019 based on current height',
                         location_status='archive-published location; historical precision/changes not fully verified',
                         instrument_history_status='unknown'))
    return pd.DataFrame(rows)


def expected_hours(start=START, end=END):
    """All local observation-day intervals, including last day's hour 24."""
    first = pd.Timestamp(start, tz=TZ) + pd.Timedelta(hours=1)
    last = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
    return pd.date_range(first, last, freq='h').tz_convert('UTC')


def read_azmet(path, site_id, height_m=np.nan, calm_threshold=0.0):
    """Read all 18 fields for schema validation; retain only time and wind fields.

    Raw rows and unparsed timestamps remain in audit, never in the model features.
    Calm threshold 0 means exact zero; nonzero sensitivity thresholds are a study choice.
    """
    content = path.getvalue() if hasattr(path, 'getvalue') else Path(path).read_text(encoding='utf-8')
    lines = [line for line in content.splitlines() if line.strip()]
    widths = pd.Series([len(fields) for fields in csv.reader(lines)])
    # Maricopa 2008 has 16 missing-data rows with an extra 999 field.
    # Read every row; flag schema anomalies and invalidate their measurements.
    raw = pd.read_csv(io.StringIO('\n'.join(lines)), header=None, names=range(max(18, widths.max())),
                      dtype=str, keep_default_na=False)
    d = raw.iloc[:, [0, 1, 2, 10, 11, 12]].copy()
    d.columns = ['year', 'day_of_year', 'hour', 'speed_mps', 'vector_speed_mps', 'direction_deg']
    original = d.copy().add_prefix('raw_')
    d = d.apply(pd.to_numeric, errors='coerce')
    schema_anomaly = widths.ne(18)
    missing = d[['speed_mps', 'vector_speed_mps', 'direction_deg']].isin([999, 9999])
    d[['speed_mps', 'vector_speed_mps', 'direction_deg']] = d[['speed_mps', 'vector_speed_mps', 'direction_deg']].mask(missing)
    d.loc[schema_anomaly, ['speed_mps', 'vector_speed_mps', 'direction_deg']] = np.nan
    year_start = pd.to_datetime(d.year.astype('Int64').astype(str), format='%Y', errors='coerce')
    days_in_year = d.year.map(lambda y: 366 if pd.notna(y) and calendar.isleap(int(y)) else 365)
    valid_time = (d.year.between(2003, 2100) & d.day_of_year.between(1, days_in_year)
                  & d.hour.between(1, 24) & d[['year', 'day_of_year', 'hour']].mod(1).eq(0).all(axis=1))
    local_end = (year_start + pd.to_timedelta(d.day_of_year - 1, unit='D')
                 + pd.to_timedelta(d.hour, unit='h')).where(valid_time)
    d['time_utc'] = local_end.dt.tz_localize(TZ).dt.tz_convert('UTC')
    d['interval_start_utc'] = d.time_utc - pd.Timedelta(hours=1)
    d['observation_date_local'] = year_start + pd.to_timedelta(d.day_of_year - 1, unit='D')
    d['site_id'], d['height_m'] = site_id, height_m
    d['speed_valid'] = np.isfinite(d.speed_mps) & d.speed_mps.ge(0)
    d['vector_speed_valid'] = np.isfinite(d.vector_speed_mps) & d.vector_speed_mps.ge(0)
    d['direction_range_valid'] = np.isfinite(d.direction_deg) & d.direction_deg.between(0, 360)
    d['calm_or_undefined_direction'] = d.speed_mps.le(calm_threshold) | d.vector_speed_mps.le(calm_threshold)
    d['direction_valid'] = (d.direction_range_valid & d.speed_valid & d.vector_speed_valid
                            & ~d.calm_or_undefined_direction)
    d['vector_exceeds_scalar'] = d.vector_speed_mps.gt(d.speed_mps + 0.11)  # rounding tolerance, flag only
    d['documented_missing'] = missing.any(axis=1)
    d['invalid_timestamp'] = ~valid_time
    d['schema_anomaly'] = schema_anomaly
    d['estimated_observation'] = pd.Series(pd.NA, index=d.index, dtype='boolean')
    d['instrument_history_unknown'] = True
    d['source_file'], d['source_row'] = str(path), np.arange(1, len(d) + 1)
    audit = pd.concat([d, original], axis=1)
    # Never coerce out-of-range values into valid values; retain raw values in audit.
    audit.loc[~audit.speed_valid, 'speed_mps'] = np.nan
    audit.loc[~audit.vector_speed_valid, 'vector_speed_mps'] = np.nan
    audit.loc[~audit.direction_valid, 'direction_deg'] = np.nan
    audit.loc[audit.direction_deg.eq(360), 'direction_deg'] = 0.0
    return audit


def documented_station_events(notes):
    """Curated interpretations of downloaded evidence; never infer event dates with NLP.

    A source is included only when its expected identifying phrases are present.
    Year-level maintenance evidence does not establish an exact replacement date.
    """
    events = []
    if notes is not None and not notes.empty:
        for note in notes.itertuples():
            text = note.text.lower()
            if note.site_id == 'az15' and note.year == 2009 and 'october 4, 2009' in text and 'phoenix greenway' in text:
                events.append(dict(site_id='az15', start_local='2009-10-04', end_local='2009-10-05',
                    kind='substituted_wind', date_precision='day', source=note.source,
                    description='Mast fell; both days wind replaced with Phoenix Greenway values; exclude as target-site observations.'))
            if note.site_id == 'az06' and note.year == 2008 and 'april 16, 2008' in text and '4:00pm' in text:
                events.append(dict(site_id='az06', start_local='2008-04-16', end_local='2008-04-16',
                    kind='hourly_outage', date_precision='hour', first_hour=1, last_hour=16, source=note.source,
                    description='Battery failure; hourly hours 1-16 lost; only DAILY data estimated, not hourly.'))
            if note.site_id == 'az22' and note.year == 2006 and 'replaced all' in text:
                events.append(dict(site_id='az22', start_local='2006-01-01', end_local='2006-12-31',
                    kind='equipment_change_date_unknown', date_precision='year', source=note.source,
                    description='Power faults and sensor replacement documented; exact dates unresolved, so exclude the entire year.'))
    return pd.DataFrame(events)


def clean_azmet(manifest, metadata, start=START, end=END, notes=None):
    heights = metadata.set_index('site_id').measurement_height_m
    audit = pd.concat([read_azmet(r.path, r.site_id, heights.loc[r.site_id])
                       for r in manifest.itertuples()], ignore_index=True)
    audit['duplicate'] = audit.time_utc.notna() & audit.duplicated(['site_id', 'time_utc'], keep=False)
    audit['annual_notes_estimate_review'] = False
    audit['station_instrument_change_review'] = False
    audit['documented_station_event'] = False
    audit['substituted_from_other_site'] = False
    if notes is not None and not notes.empty:
        for n in notes.itertuples():
            mask = audit.site_id.eq(n.site_id) & audit.year.eq(n.year)
            audit.loc[mask, 'annual_notes_estimate_review'] = n.mentions_estimates
            audit.loc[mask, 'station_instrument_change_review'] = n.mentions_change
    events = documented_station_events(notes)
    for event in events.to_dict('records'):
        affected = audit.site_id.eq(event['site_id']) & audit.observation_date_local.between(
            pd.Timestamp(event['start_local']), pd.Timestamp(event['end_local']))
        if event['kind'] == 'hourly_outage':
            affected &= audit.hour.between(event['first_hour'], event['last_hour'])
        audit.loc[affected, 'documented_station_event'] = True
        if event['kind'] == 'substituted_wind':
            audit.loc[affected, 'substituted_from_other_site'] = True
            audit.loc[affected, 'estimated_observation'] = True
        # Unknown replacement date: quarantine the entire documented year.
        if event['kind'] in ('substituted_wind', 'hourly_outage', 'equipment_change_date_unknown'):
            audit.loc[affected, ['speed_valid', 'direction_valid', 'vector_speed_valid']] = False
            audit.loc[affected, ['speed_mps', 'direction_deg', 'vector_speed_mps']] = np.nan
    # All duplicate keys are quarantined, including identical duplicates; no arbitrary keep-first.
    accepted = audit.loc[~audit.invalid_timestamp & ~audit.duplicate
                         & audit.observation_date_local.between(pd.Timestamp(start), pd.Timestamp(end))]
    wind = accepted[WIND_COLUMNS].sort_values(['site_id', 'time_utc']).reset_index(drop=True)
    gaps = []
    for site in metadata.site_id:
        absent = expected_hours(start, end).difference(wind.loc[wind.site_id.eq(site), 'time_utc'])
        gaps.append(pd.DataFrame({'site_id': site, 'time_utc': absent, 'gap': True}))
    return wind, audit, pd.concat(gaps, ignore_index=True)


def coverage_report(wind, sites=SITES, start=START, end=END):
    """Denominator includes every expected hour, including absent station-years."""
    reports = []
    for site in sites:
        grid = pd.DataFrame({'time_utc': expected_hours(start, end)})
        grid = grid.merge(wind.loc[wind.site_id.eq(site), ['time_utc', 'speed_valid', 'direction_valid']],
                          on='time_utc', how='left', validate='one_to_one')
        local_day = (grid.time_utc.dt.tz_convert(TZ) - pd.Timedelta(hours=1))
        grid['year'], grid['month'] = local_day.dt.year, local_day.dt.month
        grid['present'] = grid.speed_valid.notna()
        grid[['speed_valid', 'direction_valid']] = grid[['speed_valid', 'direction_valid']].astype('boolean').fillna(False).astype(bool)
        monthly = grid.groupby(['year', 'month']).agg(expected=('time_utc', 'size'),
                     present=('present', 'sum'), speed_valid=('speed_valid', 'sum'),
                     direction_valid=('direction_valid', 'sum')).reset_index()
        monthly['site_id'] = site
        reports.append(monthly)
    monthly = pd.concat(reports, ignore_index=True)
    yearly = monthly.groupby(['site_id', 'year'], as_index=False)[['expected', 'present', 'speed_valid', 'direction_valid']].sum()
    station = yearly.groupby('site_id', as_index=False)[['expected', 'present', 'speed_valid', 'direction_valid']].sum()
    for frame in (monthly, yearly, station):
        frame['speed_coverage'] = frame.speed_valid / frame.expected
        frame['direction_coverage'] = frame.direction_valid / frame.expected
        frame['missing_hours'] = frame.expected - frame.present
    return station, yearly, monthly


def era5_request(latitude, longitude, start=START, end=END):
    # Include UTC next-day hours needed by the last AZMet local observation day.
    utc_end = expected_hours(start, end)[-1].date().isoformat()
    return {'variable': ERA_VARIABLES.copy(), 'location': {'latitude': float(latitude), 'longitude': float(longitude)},
            'date': [f'{start}/{utc_end}'], 'data_format': 'netcdf'}


def acquire_era5(metadata, root, start=START, end=END, client=None):
    """cdsapi reads ~/.cdsapirc or CDSAPI_URL/CDSAPI_KEY. No credentials saved."""
    import cdsapi
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    client = client or cdsapi.Client()
    paths = {}
    for row in metadata.itertuples():
        if not np.isfinite([row.latitude_deg, row.longitude_deg]).all():
            raise ValueError(f'Unknown coordinates for {row.site_id}')
        request = era5_request(row.latitude_deg, row.longitude_deg, start, end)
        # Some CDS responses are ZIP; keep response bytes unchanged, independent of suffix.
        path = root / f'{row.site_id}_{start}_{end}.download'
        sidecar = path.with_suffix('.request.json')
        if path.exists():
            info = json.loads(sidecar.read_text(encoding='utf-8'))
            if info['request'] != request or info['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'ERA5 cache mismatch: {path}')
        else:
            temp = path.with_suffix('.partial')
            client.retrieve('reanalysis-era5-single-levels-timeseries', request).download(str(temp))
            temp.rename(path)
            save_json(sidecar, {'dataset': 'reanalysis-era5-single-levels-timeseries', 'request': request,
                               'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        paths[row.site_id] = path
    return paths


def load_era5(path, site_id):
    """Inspect CF names/units, require one grid cell, preserve original grid longitude."""
    import xarray as xr
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.nc')]
            if not names:
                raise ValueError(f'No NetCDF member: {path}')
            datasets = []
            for name in names:
                # Derived extraction beside immutable response; hash avoids member path traversal.
                member = path.parent / 'extracted' / (hashlib.sha256(archive.read(name)).hexdigest() + '.nc')
                member.parent.mkdir(exist_ok=True)
                if not member.exists():
                    with member.open('xb') as output:
                        output.write(archive.read(name))
                with xr.open_dataset(member, engine='netcdf4') as ds:
                    datasets.append(ds.load())
            dataset = xr.merge(datasets, compat='no_conflicts')
    else:
        with xr.open_dataset(path, engine='netcdf4') as ds:
            dataset = ds.load()
    inspection = {'site_id': site_id, 'file': str(path), 'dimensions': dict(dataset.sizes),
                  'variables': {v: dict(dataset[v].attrs) for v in dataset.data_vars},
                  'coordinates': {v: dataset[v].values.tolist() for v in dataset.coords if dataset[v].size <= 10}}
    time_name = next((n for n in ['valid_time', 'time'] if n in dataset.coords), None)
    if time_name is None:
        raise ValueError('Missing ERA5 time coordinate')
    lat_name = next((n for n in ['latitude', 'lat'] if n in dataset.coords), None)
    lon_name = next((n for n in ['longitude', 'lon'] if n in dataset.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError('ERA5 grid coordinates absent; cannot substitute requested coordinates')
    if dataset[lat_name].size != 1 or dataset[lon_name].size != 1:
        raise ValueError('Expected one ERA5 grid point per station request')
    mapping = {}
    for canonical, long_name, short_names in zip(ERA_COLUMNS, ERA_VARIABLES,
                  [('u10', '10u'), ('v10', '10v'), ('u100', '100u'), ('v100', '100v')]):
        candidates = [n for n in (canonical, long_name, *short_names) if n in dataset.data_vars]
        if len(candidates) != 1:
            raise ValueError(f'Cannot uniquely identify {canonical}; available: {list(dataset.data_vars)}')
        name = candidates[0]
        units = str(dataset[name].attrs.get('units', '')).replace(' ', '').replace('**', '^')
        if units not in ('ms^-1', 'ms-1', 'm/s', 'm.s-1'):
            raise ValueError(f'Unverified units for {name}: {units}')
        mapping[name] = canonical
    selected = dataset[list(mapping)].rename(mapping)
    for dim in list(selected.dims):
        if dim != time_name:
            if selected.sizes[dim] != 1:
                raise ValueError(f'Unresolved dimension {dim}; do not flatten ensemble/expver')
            selected = selected.isel({dim: 0}, drop=True)
    table = selected.to_dataframe().reset_index().rename(columns={time_name: 'time_utc'})
    table['time_utc'] = pd.to_datetime(table.time_utc, utc=True)
    table['site_id'] = site_id
    if table.time_utc.duplicated().any() or not table.time_utc.eq(table.time_utc.dt.floor('h')).all():
        raise ValueError('Duplicate or non-hourly ERA5 timestamps')
    table['era5_grid_latitude'] = float(dataset[lat_name].values.item())
    table['era5_grid_longitude'] = float(dataset[lon_name].values.item())
    table[ERA_COLUMNS] = table[ERA_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return table[['site_id', 'time_utc', *ERA_COLUMNS, 'era5_grid_latitude', 'era5_grid_longitude']], inspection


def fit_scaler(training_inputs):
    """NaN-preserving scaler. Unknown metadata stays NaN and has an explicit mask."""
    columns = training_inputs.select_dtypes('number').columns.tolist()
    stats = pd.DataFrame({'mean': training_inputs[columns].mean(), 'std': training_inputs[columns].std(ddof=0),
                          'n_observed': training_inputs[columns].count()})
    stats['std'] = stats['std'].mask(stats['std'].eq(0), 1.0)
    return stats


def apply_scaler(inputs, stats):
    features = inputs[stats.index].copy()
    observed = features.notna() & np.isfinite(features)
    return (features - stats['mean']) / stats['std'], observed


def cached_azmet(root, download=True):
    """Load and verify cached originals, or acquire an initial archive."""
    root = Path(root)
    manifest_path = root / 'azmet_manifest.csv'
    if not manifest_path.exists():
        if not download:
            raise FileNotFoundError('AZMet cache absent; enable download.')
        return acquire_azmet(root)
    manifest = pd.read_csv(manifest_path)
    for record in manifest.itertuples():
        if not Path(record.path).exists():
            raise FileNotFoundError(f'Missing original: {record.path}')
        download_original(record.url, record.path)
    notes = pd.DataFrame(json.loads((root / 'station_notes.json').read_text(encoding='utf-8')))
    issues = pd.DataFrame(json.loads((root / 'source_issues.json').read_text(encoding='utf-8')))
    return manifest, notes, issues


def quality_summary(audit):
    """Counts of data-quality flags; no resource or model analysis."""
    flags = ['invalid_timestamp', 'duplicate', 'documented_missing', 'schema_anomaly',
             'substituted_from_other_site', 'vector_exceeds_scalar',
             'annual_notes_estimate_review', 'station_instrument_change_review']
    result = audit.groupby('site_id')[flags].sum()
    result['raw_rows'] = audit.groupby('site_id').size()
    for flag in ['speed_valid', 'direction_valid']:
        result[flag] = audit.groupby('site_id')[flag].sum()
    result['estimated_status_unknown'] = audit.groupby('site_id').estimated_observation.apply(lambda s: s.isna().sum())
    return result


def load_era5_collection(paths, metadata):
    """Load station responses and attach returned grid points to separate metadata."""
    frames, inspections = [], []
    metadata = metadata.copy()
    for site, path in paths.items():
        table, inspection = load_era5(path, site)
        frames.append(table)
        inspections.append(inspection)
    grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({
        'site_id': pd.Series(dtype=str), 'time_utc': pd.Series(dtype='datetime64[ns, UTC]'),
        **{c: pd.Series(dtype=float) for c in [*ERA_COLUMNS, 'era5_grid_latitude', 'era5_grid_longitude']}})
    if grid.duplicated(['site_id', 'time_utc']).any():
        raise ValueError('Duplicate ERA5 station/timestamp keys')
    for site in grid.site_id.unique():
        point = grid.loc[grid.site_id.eq(site), ['era5_grid_latitude', 'era5_grid_longitude']].drop_duplicates()
        if len(point) != 1:
            raise ValueError(f'Changing grid at {site}; need time-dependent metadata.')
        metadata.loc[metadata.site_id.eq(site), point.columns] = point.iloc[0].to_numpy()
    metadata['era5_source'] = 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries'
    metadata['era5_data_kind'] = 'instantaneous reanalysis wind components'
    metadata['era5_timestamp_convention'] = 'UTC validity time at AZMet interval endpoint'
    return grid[['site_id', 'time_utc', *ERA_COLUMNS]].copy(), grid, metadata, inspections


def era5_coverage_report(era5, sites=SITES, start=START, end=END):
    """Monthly coverage requiring all four finite wind components."""
    reports = []
    for site in sites:
        check = pd.DataFrame({'time_utc': expected_hours(start, end)}).merge(
            era5.loc[era5.site_id.eq(site)], on='time_utc', how='left', validate='one_to_one')
        local_day = check.time_utc.dt.tz_convert(TZ) - pd.Timedelta(hours=1)
        check['year'], check['month'] = local_day.dt.year, local_day.dt.month
        check['components_valid'] = np.isfinite(check[ERA_COLUMNS]).all(axis=1)
        report = check.groupby(['year', 'month']).agg(expected=('time_utc', 'size'),
                    era5_valid=('components_valid', 'sum')).reset_index()
        report['site_id'] = site
        report['era5_coverage'] = report.era5_valid / report.expected
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def save_tables(folder, tables, csv=False):
    """Persist named data frames as Parquet, with optional readable CSV copies."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(folder / f'{name}.parquet', index=False)
        if csv:
            table.to_csv(folder / f'{name}.csv', index=False)


def prepare_training_data(
        wind_path='analyses_output_files/wind_pipeline/observations/wind.parquet',
        era5_path='analyses_output_files/wind_pipeline/observations/era5.parquet',
        metadata_path='analyses_output_files/wind_pipeline/metadata/metadata.parquet', *,
        development_sites=('az06', 'az05', 'az22', 'az26'),
        validation_sites=('az19',), test_sites=('az15',), fit_end_local='2015-12-31',
        n_references=2, campaign_months=(3,), start=START, end=END,
        metadata_columns=('latitude_deg', 'longitude_deg', 'ground_elevation_m')):
    """Load cleaned tables ONCE; cache development-only, cutoff-limited hourly arrays.

    No downloads, files written, combinations, or episode manifests. Pooled
    development statistics are shared across reference slots and target variables.
    Wind speed is standardized; direction sin/cos and cyclic time stay unchanged.
    Entirely unobserved metadata columns are omitted in every slot.
    """
    sites, validation, test = list(development_sites), list(validation_sites), list(test_sites)
    assigned = sites + validation + test
    if len(assigned) != len(set(assigned)):
        raise ValueError('Development/validation/test sites must be unique and disjoint.')
    if not isinstance(n_references, (int, np.integer)) or not 1 <= n_references < len(sites):
        raise ValueError('N_REFERENCES must be at least 1 and smaller than the development-site count.')
    months = list(campaign_months)
    if not months or len(months) != len(set(months)) or any(m not in (3, 6, 12) for m in months):
        raise ValueError('Campaign lengths must be unique values from 3, 6, 12 months.')
    metadata_columns = list(metadata_columns)
    allowed = {'latitude_deg', 'longitude_deg', 'ground_elevation_m'}
    if len(metadata_columns) != len(set(metadata_columns)) or not set(metadata_columns).issubset(allowed):
        raise ValueError('Select unique coordinate/elevation metadata columns.')
    first = pd.Timestamp(start).normalize().tz_localize(TZ)
    cutoff = pd.Timestamp(fit_end_local).normalize().tz_localize(TZ) + pd.DateOffset(days=1)
    last = min(pd.Timestamp(end).normalize().tz_localize(TZ) + pd.DateOffset(days=1), cutoff)
    if last <= first:
        raise ValueError('The training period must contain hours before the fitting cutoff.')
    times = expected_hours(str(first.date()), str((last - pd.DateOffset(days=1)).date()))
    filters = [('site_id', 'in', sites), ('time_utc', '>', first.tz_convert('UTC')),
               ('time_utc', '<=', last.tz_convert('UTC'))]
    wind = pd.read_parquet(wind_path, columns=['site_id', 'time_utc', 'speed_mps', 'direction_deg',
                                              'speed_valid', 'direction_valid'], filters=filters)
    era = pd.read_parquet(era5_path, columns=['site_id', 'time_utc', *ERA_COLUMNS], filters=filters)
    metadata_path = Path(metadata_path)
    meta = (pd.read_json(metadata_path) if metadata_path.suffix.lower() == '.json'
            else pd.read_parquet(metadata_path))
    meta = meta.loc[meta.site_id.isin(sites), ['site_id', *metadata_columns]]
    if meta.site_id.duplicated().any() or set(meta.site_id) != set(sites):
        raise ValueError('Require one metadata row for each development site.')
    for name, table in [('wind', wind), ('ERA5', era)]:
        if table.duplicated(['site_id', 'time_utc']).any():
            raise ValueError(f'Duplicate cleaned {name} station/timestamp keys')
    # Pool physical variables once, never by reference slot or held-out site.
    wind.loc[~wind.speed_valid.astype('boolean').fillna(False) | wind.speed_mps.lt(0), 'speed_mps'] = np.nan
    wind['speed_mps'] = wind.speed_mps.replace([np.inf, -np.inf], np.nan)
    wind_stats = fit_scaler(wind[['speed_mps']])
    era_stats = fit_scaler(era[ERA_COLUMNS].replace([np.inf, -np.inf], np.nan))
    if wind_stats.n_observed.eq(0).any() or era_stats.n_observed.eq(0).any():
        raise ValueError('Require observed development training speeds and all four ERA5 components.')
    meta = meta.set_index('site_id').loc[sites].replace([np.inf, -np.inf], np.nan)
    meta_stats = fit_scaler(meta)
    excluded = meta_stats.index[meta_stats.n_observed.eq(0)].tolist()
    meta_stats = meta_stats.loc[meta_stats.n_observed.gt(0)]
    selected_metadata = meta_stats.index.tolist()

    def normalize(frame, stats):
        scaled, mask = apply_scaler(frame, stats)
        values = scaled.where(mask, 0.).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite normalized values; inspect the development training statistics.')
        return values, mask.to_numpy(dtype=bool)

    cache = {}
    for site in sites:
        w = wind.loc[wind.site_id.eq(site)].set_index('time_utc').reindex(times)
        e = era.loc[era.site_id.eq(site)].set_index('time_utc').reindex(times)
        speed, speed_mask = normalize(w[['speed_mps']], wind_stats)
        direction_mask = (w.direction_valid.astype('boolean').fillna(False) & w.speed_mps.gt(0)
                          & np.isfinite(w.direction_deg) & speed_mask[:, 0]).to_numpy(dtype=bool)
        angle = np.deg2rad(w.direction_deg.where(direction_mask))
        direction = np.column_stack([np.sin(angle), np.cos(angle)])
        direction = np.where(direction_mask[:, None], direction, 0.).astype(np.float32)
        wind_values = np.column_stack([speed, direction])
        wind_mask = np.column_stack([speed_mask, direction_mask, direction_mask])
        era_values, era_mask = normalize(e[ERA_COLUMNS], era_stats)
        meta_values, meta_mask = normalize(meta.loc[[site]], meta_stats)
        cache[site] = {'wind': wind_values, 'wind_mask': wind_mask,
                       'era5': era_values, 'era5_mask': era_mask,
                       'metadata': meta_values[0], 'metadata_mask': meta_mask[0],
                       'essential': speed_mask[:, 0] & era_mask.all(axis=1)}
    # Local interval midpoint for cyclic predictors.
    local = times.tz_convert(TZ) - pd.Timedelta(minutes=30)
    hour = (local.hour.to_numpy() + local.minute.to_numpy() / 60) / 24
    doy = ((local.dayofyear.to_numpy() - 1) + hour) / np.where(local.is_leap_year, 366, 365)
    cyclic = np.column_stack([np.sin(2 * np.pi * hour), np.cos(2 * np.pi * hour),
                              np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)]).astype(np.float32)
    eligible_dates = {}
    dates = pd.date_range(first, last - pd.DateOffset(days=1), freq='D')
    for length in months:
        eligible_dates[length] = dates[[d + pd.DateOffset(months=length) <= last for d in dates]]
        if not len(eligible_dates[length]):
            raise ValueError(f'No complete {length}-month campaign fits inside the training period.')
    wind_names = ['speed_mps', 'direction_sin', 'direction_cos']
    reference_variables = wind_names + ERA_COLUMNS + selected_metadata
    feature_names = [f'reference_{slot}_{c}' for slot in range(1, n_references + 1) for c in reference_variables]
    feature_names += ['target_' + c for c in ERA_COLUMNS + selected_metadata]
    feature_names += ['local_hour_sin', 'local_hour_cos', 'local_doy_sin', 'local_doy_cos']
    return {'sites': cache, 'development_sites': tuple(sites), 'validation_sites': tuple(validation),
            'test_sites': tuple(test), 'n_references': n_references, 'campaign_months': tuple(months),
            'eligible_campaign_dates': eligible_dates, 'times_utc': times.tz_localize(None).to_numpy(),
            'cyclic': cyclic, 'feature_names': feature_names,
            'target_names': ['target_speed_mps', 'target_direction_sin', 'target_direction_cos'],
            'metadata_columns': selected_metadata, 'excluded_unobserved_metadata': excluded,
            'normalization': {'wind_speed': wind_stats, 'era5': era_stats, 'metadata': meta_stats,
                              'direction_sin_cos': 'unchanged; missing values zero with masks'},
            'start_utc': first.tz_convert('UTC'), 'end_utc': last.tz_convert('UTC'),
            'fit_end_utc': cutoff.tz_convert('UTC'),
            'timestamp_convention': 'NumPy datetime64[ns] is UTC; campaign start < hourly endpoint <= end'}


def sample_training_episode(data, rng, n_context=256, n_query=256, *,
                            min_context_hours=24, min_context_fraction=0.5,
                            min_query_hours=1, max_attempts=100):
    """Sample a rotating target and simultaneous reference slots from prepared data.

    Campaign validity is checked on ALL eligible campaign hours before subsampling.
    Returned rows are sorted by UTC timestamp, without replacement or padding.
    All choices use the caller's Generator; no validation/test observations enter.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a supplied numpy.random.Generator.')
    counts = [n_context, n_query, min_context_hours, min_query_hours, max_attempts]
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in counts) or not 0 <= min_context_fraction <= 1:
        raise ValueError('Sample counts, minimum hours, and max_attempts must be positive integers; fraction must be in [0, 1].')
    sites = data['development_sites']
    times = data['times_utc']
    last_reason = 'no candidate tried'
    for _ in range(max_attempts):
        target = str(rng.choice(sites))
        references = tuple(str(s) for s in rng.choice([s for s in sites if s != target],
                                                     size=data['n_references'], replace=False))
        length = int(rng.choice(data['campaign_months']))
        dates = data['eligible_campaign_dates'][length]
        begin = dates[int(rng.integers(len(dates)))]
        finish = begin + pd.DateOffset(months=length)
        begin_utc, finish_utc = begin.tz_convert('UTC'), finish.tz_convert('UTC')
        inside = (times > begin_utc.tz_localize(None).to_datetime64()) & (times <= finish_utc.tz_localize(None).to_datetime64())
        target_data = data['sites'][target]
        usable = target_data['wind_mask'][:, 0] & target_data['era5_mask'].all(axis=1)
        for site in references:
            usable = usable & data['sites'][site]['essential']
        context_ids, query_ids = np.flatnonzero(inside & usable), np.flatnonzero(~inside & usable)
        expected = int((finish_utc - begin_utc) / pd.Timedelta(hours=1))
        if len(context_ids) < min_context_hours or len(context_ids) / expected < min_context_fraction:
            last_reason = f'{target}: only {len(context_ids)}/{expected} usable campaign hours'
            continue
        if len(query_ids) < min_query_hours:
            last_reason = f'{target}: only {len(query_ids)} usable outside-campaign hours'
            continue
        context_ids = np.sort(rng.choice(context_ids, size=min(n_context, len(context_ids)), replace=False))
        query_ids = np.sort(rng.choice(query_ids, size=min(n_query, len(query_ids)), replace=False))

        def assemble(ids):
            values, masks = [], []
            for site in references:
                reference = data['sites'][site]
                values.extend([reference['wind'][ids], reference['era5'][ids],
                               np.broadcast_to(reference['metadata'], (len(ids), len(reference['metadata'])))])
                masks.extend([reference['wind_mask'][ids], reference['era5_mask'][ids],
                              np.broadcast_to(reference['metadata_mask'], (len(ids), len(reference['metadata_mask'])))])
            values.extend([target_data['era5'][ids],
                           np.broadcast_to(target_data['metadata'], (len(ids), len(target_data['metadata']))),
                           data['cyclic'][ids]])
            masks.extend([target_data['era5_mask'][ids],
                          np.broadcast_to(target_data['metadata_mask'], (len(ids), len(target_data['metadata_mask']))),
                          np.ones((len(ids), 4), dtype=bool)])
            return (np.concatenate(values, axis=1), target_data['wind'][ids].copy(),
                    np.concatenate(masks, axis=1), target_data['wind_mask'][ids].copy())

        cx, cy, cx_mask, cy_mask = assemble(context_ids)
        qx, qy, qx_mask, qy_mask = assemble(query_ids)
        return {'context_x': cx, 'context_y': cy, 'query_x': qx, 'query_y': qy,
                'context_x_mask': cx_mask, 'context_y_mask': cy_mask,
                'query_x_mask': qx_mask, 'query_y_mask': qy_mask,
                'target_site_id': target, 'reference_site_ids': references,
                'campaign_months': length, 'campaign_start_utc': begin_utc, 'campaign_end_utc': finish_utc,
                'context_times_utc': times[context_ids].copy(), 'query_times_utc': times[query_ids].copy(),
                'usable_context_hours': int((inside & usable).sum()),
                'feature_names': data['feature_names'], 'target_names': data['target_names']}
    raise ValueError(f'Could not sample an adequate training episode after {max_attempts} attempts: {last_reason}. '
                     'Check development coverage, campaign lengths, and context/query thresholds.')




# Shared inference definitions. Loading data in notebook 1 does not require PyTorch.
def _build_context_model_class():
    import torch
    from torch import nn

    class WindContextModel(nn.Module):
        def __init__(self, n_features, n_targets, d_model, n_heads, d_ff,
                     l_enc, l_dec, dropout, activation, layer_norm_eps):
            super().__init__()
            if n_heads < 1 or d_model % n_heads:
                raise ValueError("D_MODEL must be divisible by N_HEADS.")
            if n_targets != 3 or activation != 'gelu' or min(l_enc, l_dec) < 1:
                raise ValueError("Require three wind outputs, GELU, and positive layer counts.")
            self.head_width = d_model // n_heads
            # 1. Projection input widths: 2F+6 and 2F.
            self.context_projection = nn.Linear(2*n_features + 2*n_targets, d_model)
            self.query_projection = nn.Linear(2*n_features, d_model)

            def layer():
                return nn.ModuleDict({
                    'attention': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                    'attention_dropout': nn.Dropout(dropout),
                    'norm1': nn.LayerNorm(d_model, eps=layer_norm_eps),
                    'ff': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)),
                    'ff_dropout': nn.Dropout(dropout),
                    'norm2': nn.LayerNorm(d_model, eps=layer_norm_eps)})
            self.encoder = nn.ModuleList([layer() for _ in range(l_enc)])
            self.decoder = nn.ModuleList([layer() for _ in range(l_dec)])
            self.output_head = nn.Linear(d_model, n_targets)

        def encode_context(self, context_x, context_x_mask, context_y, context_y_mask):
            # 1. Observed masks are part of the embedding input.
            h = self.context_projection(torch.cat((context_x, context_x_mask.to(context_x.dtype),
                                                  context_y, context_y_mask.to(context_y.dtype)), dim=-1))
            # 2. Separately parameterized context self-attention layers.
            for layer in self.encoder:
                attended, _ = layer['attention'](h, h, h, need_weights=False)
                z = layer['norm1'](h + layer['attention_dropout'](attended))
                h = layer['norm2'](z + layer['ff_dropout'](layer['ff'](z)))
            return h

        def predict_queries(self, encoded_context, query_x, query_x_mask):
            h = self.query_projection(torch.cat((query_x, query_x_mask.to(query_x.dtype)), dim=-1))
            # 3. Cross-attention to the final context encoding, with no query self-attention.
            for layer in self.decoder:
                attended, _ = layer['attention'](h, encoded_context, encoded_context, need_weights=False)
                z = layer['norm1'](h + layer['attention_dropout'](attended))
                h = layer['norm2'](z + layer['ff_dropout'](layer['ff'](z)))
            # 4. Unconstrained [1,Q,len(target_names)] predictions.
            return self.output_head(h)

        def forward(self, context_x, context_x_mask, context_y, context_y_mask, query_x, query_x_mask):
            encoded = self.encode_context(context_x, context_x_mask, context_y, context_y_mask)
            return self.predict_queries(encoded, query_x, query_x_mask)

    WindContextModel.__qualname__ = 'WindContextModel'
    return WindContextModel


def __getattr__(name):
    # Load the attention class only when a training/inference notebook requests it.
    if name == 'WindContextModel':
        if name not in globals():
            globals()[name] = _build_context_model_class()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def plain_normalization(normalization):
    # Preserve statistic dtypes as plain strings to reproduce float32 preprocessing.
    return {k: {'values': v.to_dict(orient='index'), 'dtypes': v.dtypes.astype(str).to_dict()}
            if isinstance(v, pd.DataFrame) else v for k, v in normalization.items()}


def restore_normalization(normalization):
    return {k: pd.DataFrame.from_dict(v['values'], orient='index').reindex(columns=list(v['dtypes'])).astype(v['dtypes'])
            if k in ('wind_speed', 'era5', 'metadata') and not isinstance(v, pd.DataFrame) else v
            for k, v in normalization.items()}


def load_aligned_sites(wind_path, era5_path, metadata_path, site_ids, start_local, end_local,
                       normalization, metadata_columns, local_timezone, wind_windows=None):
    """Read selected sites once and apply saved normalization without fitting.

    Optional wind_windows maps every site to (start_utc, end_utc), with
    start < endpoint <= end. Target campaign-only reads prevent label leakage.
    """
    site_ids = tuple(site_ids)
    first = pd.Timestamp(start_local).normalize().tz_localize(local_timezone)
    last = pd.Timestamp(end_local).normalize().tz_localize(local_timezone) + pd.DateOffset(days=1)
    if last <= first or len(set(site_ids)) != len(site_ids):
        raise ValueError("Require distinct sites and a positive date range.")
    times = pd.date_range(first + pd.Timedelta(hours=1), last, freq='h').tz_convert('UTC')
    base_filters = [('site_id', 'in', list(site_ids)), ('time_utc', '>', first.tz_convert('UTC')),
                    ('time_utc', '<=', last.tz_convert('UTC'))]
    if wind_windows is None:
        wind_filters = base_filters
    else:
        if set(wind_windows) != set(site_ids):
            raise ValueError("Supply wind read bounds for every selected site.")
        wind_filters = [[('site_id', '==', s),
                         ('time_utc', '>', max(pd.Timestamp(wind_windows[s][0]), first.tz_convert('UTC'))),
                         ('time_utc', '<=', min(pd.Timestamp(wind_windows[s][1]), last.tz_convert('UTC')))]
                        for s in site_ids]
    wind = pd.read_parquet(wind_path, columns=['site_id', 'time_utc', 'speed_mps', 'direction_deg',
                                              'speed_valid', 'direction_valid'], filters=wind_filters)
    era = pd.read_parquet(era5_path, columns=['site_id', 'time_utc', *ERA_COLUMNS], filters=base_filters)
    # Height is read only for output provenance; normalize/assemble retained predictors only.
    columns = list(dict.fromkeys(['site_id', *metadata_columns, 'measurement_height_m']))
    meta = pd.read_parquet(metadata_path, columns=columns, filters=[('site_id', 'in', list(site_ids))])
    if meta.site_id.duplicated().any() or set(meta.site_id) != set(site_ids):
        raise ValueError("Require one metadata row per selected site.")
    for name, table in [('wind', wind), ('ERA5', era)]:
        if table.duplicated(['site_id', 'time_utc']).any():
            raise ValueError(f"Duplicate cleaned {name} site/timestamp keys.")
    meta = meta.set_index('site_id')
    stats = restore_normalization(normalization)
    if list(stats['metadata'].index) != list(metadata_columns):
        raise ValueError("Retained metadata must match saved column order.")
    wind.loc[~wind.speed_valid.astype('boolean').fillna(False) | wind.speed_mps.lt(0), 'speed_mps'] = np.nan
    wind['speed_mps'] = wind.speed_mps.replace([np.inf, -np.inf], np.nan)

    def normalize(frame, scaler):
        scaled, mask = apply_scaler(frame, scaler)
        values = scaled.where(mask, 0.).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite normalized values; inspect saved training statistics.")
        return values, mask.to_numpy(dtype=bool)

    cache = {}
    for site in site_ids:
        w = wind.loc[wind.site_id.eq(site)].set_index('time_utc').reindex(times)
        e = era.loc[era.site_id.eq(site)].set_index('time_utc').reindex(times)
        speed, speed_mask = normalize(w[['speed_mps']], stats['wind_speed'])
        dm = (w.direction_valid.astype('boolean').fillna(False) & w.speed_mps.gt(0)
              & np.isfinite(w.direction_deg) & speed_mask[:, 0]).to_numpy(dtype=bool)
        angle = np.deg2rad(w.direction_deg.where(dm))
        direction = np.where(dm[:, None], np.column_stack([np.sin(angle), np.cos(angle)]), 0.).astype(np.float32)
        era_values, era_mask = normalize(e[ERA_COLUMNS], stats['era5'])
        metadata_values, metadata_mask = normalize(meta.loc[[site], list(metadata_columns)], stats['metadata'])
        cache[site] = dict(wind=np.column_stack([speed, direction]),
                           wind_mask=np.column_stack([speed_mask, dm, dm]),
                           era5=era_values, era5_mask=era_mask,
                           metadata=metadata_values[0], metadata_mask=metadata_mask[0],
                           essential=speed_mask[:, 0] & era_mask.all(axis=1))
    local = times.tz_convert(local_timezone) - pd.Timedelta(minutes=30)
    hour = (local.hour.to_numpy() + local.minute.to_numpy()/60) / 24
    doy = ((local.dayofyear.to_numpy()-1) + hour) / np.where(local.is_leap_year, 366, 365)
    cyclic = np.column_stack([np.sin(2*np.pi*hour), np.cos(2*np.pi*hour),
                              np.sin(2*np.pi*doy), np.cos(2*np.pi*doy)]).astype(np.float32)
    return dict(sites=cache, times_utc=times.tz_localize(None).to_numpy(), cyclic=cyclic,
                measurement_heights_m={s: float(meta.loc[s, 'measurement_height_m']) for s in site_ids})


def assemble_predictors(data, target, references, ids, feature_names, metadata_columns):
    """Reference sequence defines slot order. Target measured wind never enters X."""
    values, masks, names = [], [], []
    metadata_columns = list(metadata_columns)
    for slot, site in enumerate(references, start=1):
        ref = data['sites'][site]
        values.extend([ref['wind'][ids], ref['era5'][ids], np.broadcast_to(ref['metadata'], (len(ids), len(metadata_columns)))])
        masks.extend([ref['wind_mask'][ids], ref['era5_mask'][ids], np.broadcast_to(ref['metadata_mask'], (len(ids), len(metadata_columns)))])
        names.extend([f'reference_{slot}_{c}' for c in ['speed_mps', 'direction_sin', 'direction_cos', *ERA_COLUMNS, *metadata_columns]])
    tar = data['sites'][target]
    values.extend([tar['era5'][ids], np.broadcast_to(tar['metadata'], (len(ids), len(metadata_columns))), data['cyclic'][ids]])
    masks.extend([tar['era5_mask'][ids], np.broadcast_to(tar['metadata_mask'], (len(ids), len(metadata_columns))), np.ones((len(ids), 4), dtype=bool)])
    names.extend(['target_' + c for c in [*ERA_COLUMNS, *metadata_columns]])
    names.extend(['local_hour_sin', 'local_hour_cos', 'local_doy_sin', 'local_doy_cos'])
    if names != list(feature_names):
        raise ValueError("Feature order differs from the trained model.")
    return np.concatenate(values, axis=1), np.concatenate(masks, axis=1)


def campaign_indices(data, target, references, begin, finish, min_context_hours, min_context_fraction):
    times = pd.to_datetime(data['times_utc'], utc=True)
    if begin < times[0]-pd.Timedelta(hours=1) or finish > times[-1] or finish <= begin:
        raise ValueError(f"Campaign {begin} to {finish} lies outside the available period.")
    inside = np.asarray((times > begin) & (times <= finish))
    tar = data['sites'][target]
    usable = tar['wind_mask'][:, 0] & tar['era5_mask'].all(axis=1)
    for ref in references:
        usable = usable & data['sites'][ref]['essential']
    ci = np.flatnonzero(inside & usable)
    expected = int((finish-begin)/pd.Timedelta(hours=1))
    if len(ci) < min_context_hours or len(ci)/expected < min_context_fraction:
        raise ValueError(f"{target}: campaign {begin} to {finish} has only {len(ci)}/{expected} usable hours.")
    return ci, np.flatnonzero(~inside & usable)


def load_checkpoint(path, device):
    import torch
    from wind_pipeline import WindContextModel
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    loaded_model = WindContextModel(**checkpoint['architecture']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    return loaded_model, checkpoint, restore_normalization(checkpoint['normalization'])


def assemble_reconstruction_inputs(wind_path, era5_path, metadata_path, saved, normalization,
                                   target, references, campaign_start_local, campaign_months,
                                   start_local, end_local, context_mode="full", rng=None):
    settings = saved['preprocessing']
    references = tuple(references)
    if context_mode not in ('trained', 'full'):
        raise ValueError("context_mode must be 'trained' or 'full'.")
    if context_mode == 'trained' and not isinstance(rng, np.random.Generator):
        raise TypeError("Supply a separate NumPy Generator for trained-count context selection.")
    if len(references) != settings['n_references'] or len(set(references)) != len(references):
        raise ValueError("Require the configured number of distinct references in explicit slot order.")
    if target in references or not set(references).issubset(saved['site_roles']['development_sites']):
        raise ValueError("References must be development sites distinct from the target.")
    timezone = settings['local_timezone']
    begin = pd.Timestamp(campaign_start_local).normalize().tz_localize(timezone)
    finish = begin + pd.DateOffset(months=campaign_months)
    begin, finish = begin.tz_convert('UTC'), finish.tz_convert('UTC')
    period_start = pd.Timestamp(start_local).normalize().tz_localize(timezone).tz_convert('UTC')
    period_end = (pd.Timestamp(end_local).normalize().tz_localize(timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    windows = {s: (period_start, period_end) for s in references}
    windows[target] = (begin, finish)
    data = load_aligned_sites(wind_path, era5_path, metadata_path, [target, *references],
                              start_local, end_local, normalization, saved['metadata_columns'], timezone, windows)
    ci, _ = campaign_indices(data, target, references, begin, finish,
                              settings['min_context_hours'], settings['min_context_fraction'])
    usable_context_hours = len(ci)
    if context_mode == 'trained':
        trained_count = settings['n_context']
        if not isinstance(trained_count, (int, np.integer)) or trained_count < 1:
            raise ValueError("The checkpoint must contain a positive training n_context.")
        ci = np.sort(rng.choice(ci, size=min(trained_count, len(ci)), replace=False))
    qi = np.arange(len(data['times_utc']))  # Every requested hour, including missing predictors.
    cx, cm = assemble_predictors(data, target, references, ci, saved['feature_names'], saved['metadata_columns'])
    qx, qm = assemble_predictors(data, target, references, qi, saved['feature_names'], saved['metadata_columns'])
    tar = data['sites'][target]
    details = dict(target_site_id=target, reference_site_ids=list(references),
                   measurement_height_m=data['measurement_heights_m'][target],
                   campaign_start_utc=begin.isoformat(), campaign_end_utc=finish.isoformat(),
                   reconstruction_start_utc=pd.Timestamp(data['times_utc'][0], tz='UTC').isoformat(),
                   reconstruction_end_utc=pd.Timestamp(data['times_utc'][-1], tz='UTC').isoformat(),
                   model_identity=saved['model_identity'], context_mode=context_mode,
                   usable_context_hours=usable_context_hours, context_hours_used=len(ci))
    arrays = dict(context_x=cx, context_x_mask=cm, context_y=tar['wind'][ci].copy(),
                  context_y_mask=tar['wind_mask'][ci].copy(), query_x=qx, query_x_mask=qm,
                  query_times_utc=pd.to_datetime(data['times_utc'], utc=True))
    return arrays, details


def reconstruct_wind(model, context_x, context_x_mask, context_y, context_y_mask,
                     query_x, query_x_mask, query_times_utc, normalization, device,
                     query_chunk_size, reconstruction_metadata, calm_speed_mps, direction_vector_eps):
    """Unbatched arrays [C,F]/[C,3]/[Q,F]. No query labels are required."""
    import torch
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("Query chunk size must be a positive integer.")
    times = pd.to_datetime(query_times_utc, utc=True)
    if len(times) != len(query_x) or len(times) == 0 or times.isna().any():
        raise ValueError("Require one valid UTC timestamp per query row.")
    for values, mask in [(context_x, context_x_mask), (context_y, context_y_mask), (query_x, query_x_mask)]:
        if values.ndim != 2 or values.shape != mask.shape:
            raise ValueError("Values and masks must be matching two-dimensional arrays.")
    if len(context_x) == 0 or len(context_x) != len(context_y) or not context_y_mask[:, 0].all():
        raise ValueError("Supply nonempty context rows with observed speed supervision.")

    def tensor(array, mask=False):
        return torch.as_tensor(array, device=device, dtype=torch.bool if mask else torch.float32).unsqueeze(0)

    was_training = model.training
    model.eval()
    outputs = []
    try:
        with torch.no_grad():
            encoded = model.encode_context(tensor(context_x), tensor(context_x_mask, True),
                                           tensor(context_y), tensor(context_y_mask, True))
            for first in range(0, len(query_x), query_chunk_size):
                last = first+query_chunk_size
                prediction = model.predict_queries(encoded, tensor(query_x[first:last]), tensor(query_x_mask[first:last], True))
                outputs.append(prediction.squeeze(0).cpu().numpy())
    finally:
        model.train(was_training)
    prediction = np.concatenate(outputs, axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite reconstruction predictions.")
    stats = restore_normalization(normalization)['wind_speed'].loc['speed_mps']
    speed = np.maximum(0., prediction[:, 0].astype(np.float64)*stats['std']+stats['mean'])
    direction = np.degrees(np.arctan2(prediction[:, 1].astype(np.float64), prediction[:, 2].astype(np.float64))) % 360
    direction[(speed <= calm_speed_mps) | (np.hypot(prediction[:, 1], prediction[:, 2]) <= direction_vector_eps)] = np.nan
    # Conservative per-query completeness; never consult target observations.
    predictors_complete = np.asarray(query_x_mask, dtype=bool).all(axis=1)
    result = pd.DataFrame(dict(time_utc=times, predicted_speed_mps=speed,
                               predicted_direction_deg=direction, predictors_complete=predictors_complete))
    metadata = dict(reconstruction_metadata)
    metadata.update(reconstruction_start_utc=times.min().isoformat(), reconstruction_end_utc=times.max().isoformat())
    return result, metadata

