"""Wind-only loaders and leakage-safe feature preparation for notebook 1.

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
METADATA_COLUMNS = ['site_id', 'latitude_deg', 'longitude_deg', 'ground_elevation_m',
                    'measurement_height_m', 'source', 'data_kind', 'timestamp_convention',
                    'station_history', 'era5_grid_latitude', 'era5_grid_longitude']


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
    """Use archive location evidence. Current height evidence is not a historical height."""
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
                         measurement_height_m=np.nan, source=ARCHIVES[site], data_kind='AZMet hourly summary',
                         timestamp_convention='time_utc = end of preceding hour; original hours 1..24, MST UTC-07',
                         station_history=history, era5_grid_latitude=np.nan, era5_grid_longitude=np.nan,
                         current_documented_height_m=float(height[1]) if height else np.nan,
                         height_source=f'https://azmet.arizona.edu/about/station-metadata/{site}',
                         height_status='unknown for 2005-2019; current page has no dated sensor history',
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
                    description='Power faults and equipment/sensor replacement documented; exact change dates unresolved.'))
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
        if event['kind'] in ('substituted_wind', 'hourly_outage'):
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


def select_common_period(monthly, selection_sites, minimum_speed_coverage=0.9):
    """Longest contiguous run of months passing at every selection site; no data filling."""
    view = monthly.loc[monthly.site_id.isin(selection_sites)].copy()
    view['period'] = pd.PeriodIndex.from_fields(year=view.year, month=view.month, freq='M')
    counts = view.groupby('period').agg(n=('site_id', 'nunique'), worst=('speed_coverage', 'min'))
    good = counts.index[(counts.n == len(selection_sites)) & (counts.worst >= minimum_speed_coverage)]
    if not len(good):
        return None
    runs, current = [], []
    for period in good:
        if current and period.ordinal != current[-1].ordinal + 1:
            runs.append(current)
            current = []
        current.append(period)
    runs.append(current)
    best = max(runs, key=len)
    return best[0].start_time.date().isoformat(), best[-1].end_time.date().isoformat()


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


def construct_pairs(reference_wind, era5, metadata, reference_site, target_sites, start, end):
    """Inputs use ONLY reference observations; target labels are attached separately."""
    if reference_site in target_sites or not set(reference_wind.site_id).issubset({reference_site}):
        raise ValueError('Pass reference observations only; reference cannot also be a target')
    times = expected_hours(start, end)
    ref = reference_wind.set_index('time_utc').reindex(times)
    angle = np.deg2rad(ref.direction_deg.where(ref.direction_valid.fillna(False)))
    base = pd.DataFrame({'time_utc': times, 'reference_speed_mps': ref.speed_mps.where(ref.speed_valid.fillna(False)).to_numpy(),
                         'reference_direction_sin': np.sin(angle).to_numpy(),
                         'reference_direction_cos': np.cos(angle).to_numpy()})
    local = pd.Series(times.tz_convert(TZ)) - pd.Timedelta(minutes=30)  # interval midpoint
    hour_phase = 2 * np.pi * (local.dt.hour + local.dt.minute / 60) / 24
    doy_phase = 2 * np.pi * ((local.dt.dayofyear - 1) + (local.dt.hour + local.dt.minute / 60) / 24) / np.where(local.dt.is_leap_year, 366, 365)
    for name, phase in [('local_hour', hour_phase), ('local_doy', doy_phase)]:
        base[name + '_sin'], base[name + '_cos'] = np.sin(phase).to_numpy(), np.cos(phase).to_numpy()
    er = era5.loc[era5.site_id.eq(reference_site), ['time_utc', *ERA_COLUMNS]].rename(columns={c: 'reference_' + c for c in ERA_COLUMNS})
    base = base.merge(er, on='time_utc', how='left', validate='one_to_one')
    meta = metadata.set_index('site_id')
    frames = []
    for target in target_sites:
        et = era5.loc[era5.site_id.eq(target), ['time_utc', *ERA_COLUMNS]].rename(columns={c: 'target_' + c for c in ERA_COLUMNS})
        task = base.merge(et, on='time_utc', how='left', validate='one_to_one')
        task['target_site_id'], task['reference_site_id'] = target, reference_site
        for prefix, site in [('target_', target), ('reference_', reference_site)]:
            for column in ['latitude_deg', 'longitude_deg', 'ground_elevation_m', 'measurement_height_m']:
                task[prefix + column] = meta.loc[site, column]
        task['input_complete'] = np.isfinite(task.select_dtypes('number')).all(axis=1)
        frames.append(task)
    return pd.concat(frames, ignore_index=True)


def attach_labels(inputs, target_observations):
    """Separate label table, component masks, including missing targets on input timeline."""
    labels = target_observations[['site_id', 'time_utc', 'speed_mps', 'direction_deg', 'speed_valid', 'direction_valid']].rename(columns={'site_id': 'target_site_id'})
    result = inputs[['target_site_id', 'time_utc']].merge(labels, on=['target_site_id', 'time_utc'], how='left', validate='one_to_one')
    result['speed_mask'] = result.speed_valid.astype('boolean').fillna(False).astype(bool) & result.speed_mps.notna()
    result['direction_mask'] = result.direction_valid.astype('boolean').fillna(False).astype(bool) & result.direction_deg.notna()
    result['target_speed_mps'] = result.speed_mps.where(result.speed_mask)
    radians = np.deg2rad(result.direction_deg.where(result.direction_mask))
    result['target_direction_sin'], result['target_direction_cos'] = np.sin(radians), np.cos(radians)
    return result[['target_site_id', 'time_utc', 'target_speed_mps', 'target_direction_sin', 'target_direction_cos', 'speed_mask', 'direction_mask']]


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


def prepare_model_inputs(reference_wind, training_observations, validation_observations,
                         era5, metadata, reference_site, train_sites, validation_sites,
                         test_sites, start, end, fit_end_local, output):
    """Optional data preparation only. Test observations are never accepted as inputs."""
    groups = [set([reference_site]), set(train_sites), set(validation_sites), set(test_sites)]
    if sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError('Reference/train/validation/test sites must be disjoint.')
    if not set(training_observations.site_id).issubset(train_sites):
        raise ValueError('Unexpected training observation sites')
    if not set(validation_observations.site_id).issubset(validation_sites):
        raise ValueError('Unexpected validation observation sites')
    output = Path(output)
    tasks = {name: construct_pairs(reference_wind, era5, metadata, reference_site, sites, start, end)
             for name, sites in [('train', train_sites), ('validation', validation_sites), ('test', test_sites)]}
    labels = {'train': attach_labels(tasks['train'], training_observations),
              'validation': attach_labels(tasks['validation'], validation_observations)}
    local_day = (tasks['train'].time_utc.dt.tz_convert(TZ) - pd.Timedelta(hours=1)).dt.date
    fitting = local_day.le(pd.Timestamp(fit_end_local).date())
    if not fitting.any():
        raise ValueError('No rows in the declared fitting window.')
    scaler = fit_scaler(tasks['train'].loc[fitting])
    save_tables(output / 'features', {'training_scaler': scaler.rename_axis('feature').reset_index()})
    dynamic = ['reference_speed_mps', 'reference_direction_sin', 'reference_direction_cos',
               *['reference_' + c for c in ERA_COLUMNS], *['target_' + c for c in ERA_COLUMNS]]
    summaries = []
    for name, inputs in tasks.items():
        inputs['dynamic_inputs_available'] = np.isfinite(inputs[dynamic]).all(axis=1)
        scaled, observed = apply_scaler(inputs, scaler)
        keys = inputs[['target_site_id', 'reference_site_id', 'time_utc']].reset_index(drop=True)
        save_tables(output / 'features', {
            name + '_inputs': inputs,
            name + '_scaled': pd.concat([keys, scaled.reset_index(drop=True)], axis=1),
            name + '_feature_masks': pd.concat([keys, observed.add_suffix('_observed').reset_index(drop=True)], axis=1)})
        summary = {'split': name, 'rows': len(inputs), 'dynamic_inputs_available': int(inputs.dynamic_inputs_available.sum())}
        if name in labels:
            summary['usable_speed_labels'] = int((inputs.dynamic_inputs_available & labels[name].speed_mask).sum())
            summary['usable_direction_labels'] = int((inputs.dynamic_inputs_available & labels[name].direction_mask).sum())
        summaries.append(summary)
    save_tables(output / 'labels', {name + '_labels': table for name, table in labels.items()})
    save_tables(output / 'features', {'training_time_split': tasks['train'][['target_site_id', 'time_utc']].assign(fit_window=fitting.to_numpy())})
    return pd.DataFrame(summaries)
