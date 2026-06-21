import pandas as pd
import numpy as np
import json
import math
import webbrowser
import os
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.models import (
    ColumnDataSource, CustomJS, HoverTool, Select, RangeSlider,
    MultiChoice, Div, WMTSTileSource, LinearColorMapper,
    ColorBar, BasicTicker, Span, LabelSet, NumeralTickFormatter,
    FixedTicker, Label, LinearAxis, Range1d, CheckboxGroup, CustomJSTickFormatter,
    DatetimeTickFormatter
)
from bokeh.layouts import column, row
from bokeh.palettes import Turbo256
from bokeh.embed import components

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MAIN_MAX_WIDTH = 1200
MIN_PUBLISH_YEAR = 2012
MAX_PUBLISH_YEAR = 2026


# ==========================================
# 1. DATA SETUP
# ==========================================

df_models = pd.read_csv('all_ai_models - Clean.csv')
df_models['pub_date'] = pd.to_datetime(df_models['Publication date'], errors='coerce')
df_models['params'] = pd.to_numeric(df_models['Parameters'], errors='coerce')
df_models['flops'] = pd.to_numeric(df_models['Training compute (FLOP)'], errors='coerce')
df_models = df_models.dropna(subset=['pub_date'])

def clean_org_name(name):
    name = str(name).strip()
    name = ' '.join(name.split())
    return name

df_models['org_list'] = df_models['Organization'].apply(
    lambda x: [clean_org_name(o) for o in str(x).split(',') if clean_org_name(o)] if pd.notna(x) else ['Unknown']
)
df_models = df_models.explode('org_list').reset_index(drop=True)
df_models['org'] = df_models['org_list']

COUNTRY_MAP = {
    'United States of America': 'USA', 'United States': 'USA', 'US': 'USA',
    'United Kingdom of Great Britain and Northern Ireland': 'UK', 'United Kingdom': 'UK',
    'Korea (Republic of)': 'South Korea', 'South Korea': 'South Korea', 'Korea': 'South Korea',
    'Iran (Islamic Republic of)': 'Iran',
    'China': 'China', 'Hong Kong': 'Hong Kong', 'Taiwan': 'Taiwan',
    'Canada': 'Canada', 'Australia': 'Australia', 'France': 'France',
    'Germany': 'Germany', 'Japan': 'Japan', 'India': 'India',
    'Singapore': 'Singapore', 'Switzerland': 'Switzerland', 'Sweden': 'Sweden',
    'Netherlands': 'Netherlands', 'Israel': 'Israel', 'Italy': 'Italy',
    'Spain': 'Spain', 'Russia': 'Russia', 'Brazil': 'Brazil',
    'Denmark': 'Denmark', 'Finland': 'Finland', 'Norway': 'Norway',
    'Poland': 'Poland', 'Belgium': 'Belgium', 'Austria': 'Austria',
    'Ireland': 'Ireland', 'Portugal': 'Portugal', 'Czechia': 'Czechia',
    'Croatia': 'Croatia', 'Saudi Arabia': 'Saudi Arabia',
    'United Arab Emirates': 'UAE', 'New Zealand': 'New Zealand',
    'Thailand': 'Thailand', 'Vietnam': 'Vietnam',
    'Argentina': 'Argentina', 'Bulgaria': 'Bulgaria', 'Greece': 'Greece',
    'Macao': 'Macao', 'Multinational': 'Multinational',
    'Iceland': 'Iceland', 'Ukraine': 'Ukraine', 'Indonesia': 'Indonesia',
}

def normalize_countries(raw_val):
    if pd.isna(raw_val) or str(raw_val).strip() == '':
        return 'Unknown'
    parts = [p.strip() for p in str(raw_val).split(',')]
    normalized = []
    for p in parts:
        if p in COUNTRY_MAP:
            normalized.append(COUNTRY_MAP[p])
        elif p and p != '':
            normalized.append(p)
    if not normalized:
        return 'Unknown'
    seen = set()
    unique = []
    for c in normalized:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return ', '.join(unique)

df_models['country'] = df_models['Country (of organization)'].apply(normalize_countries)
df_models['accessibility_simple'] = df_models['Open model weights?'].apply(
    lambda x: 'Open Weight' if str(x).strip().lower() == 'yes' else 'Closed'
)
df_models['accessibility_detail'] = df_models['Model accessibility'].fillna('Unknown')
df_models['log_params'] = np.where(df_models['params'] > 0, np.log10(df_models['params']), 0)
df_models['log_flops'] = np.where(df_models['flops'] > 0, np.log10(df_models['flops']), 0)
df_models['negative_log_flops'] = -df_models['log_flops']

df_models_clean = df_models.fillna({'params': 0, 'flops': 0, 'log_params': 0, 'log_flops': 0, 'negative_log_flops': 0})
df_models_clean['pub_year'] = df_models_clean['pub_date'].dt.year
df_models_clean['pub_date_ms'] = df_models_clean['pub_date'].astype('datetime64[ms]').astype('int64')
df_models_clean = df_models_clean[(df_models_clean['pub_year'] >= 2012) & (df_models_clean['pub_year'] <= 2026)]

# * Helper formatting for hover text in Chart 00
def format_params(p):
    if pd.isna(p) or p <= 0:
        return "Unknown"
    if p >= 1e12:
        return f"{p/1e12:.1f}T"
    if p >= 1e9:
        return f"{p/1e9:.1f}B"
    if p >= 1e6:
        return f"{p/1e6:.1f}M"
    return f"{p:,}"

def format_flops(f):
    if pd.isna(f) or f <= 0:
        return "Unknown"
    return f"{f:.2e}"

df_models_clean['params_str'] = df_models_clean['params'].apply(format_params)
df_models_clean['flops_str'] = df_models_clean['flops'].apply(format_flops)

# * Color mapping by accessibility
df_models_clean['color'] = df_models_clean['accessibility_simple'].map({
    'Open Weight': '#00f0a0',
    'Closed': '#ff6b6b'
}).fillna('#8b949e')


# --- Data Centers ---
with open('all_datacenter_per_country.json', 'r') as f:
    dc_data = json.load(f)

dc_list = []
for ctry, data in dc_data.items():
    if 'items' in data:
        for item in data['items']:
            lat = item.get('lat', 0)
            lon = item.get('lng', 0)
            if lat and lon:
                x = lon * 20037508.34 / 180.0
                y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
                y = y * 20037508.34 / 180.0
            else:
                x, y = 0, 0
            dc_list.append({
                'title': item.get('title', ''),
                'megawatts': item.get('megawatts', 0) or 0,
                'company': ", ".join(item.get('companies', [])),
                'country': item.get('country', ''),
                'lat': lat, 'lon': lon, 'x': x, 'y': y
            })
df_datacenters = pd.DataFrame(dc_list)

# --- Electricity / Carbon Intensity ---
with open('all_electricity_carbon_intensity_kwh.json', 'r') as f:
    elec_data = json.load(f)

elec_list = []
for ctry, data in elec_data.items():
    if 'data' in data and len(data['data']) > 0:
        latest = sorted(data['data'], key=lambda k: k.get('year', 0))[-1]
        elec_list.append({
            'country': ctry,
            'carbon_intensity': latest.get('year_avg', 0)
        })
df_elec = pd.DataFrame(elec_list)

df_elec['country'] = df_elec['country'].replace({'United States': 'USA', 'United Kingdom': 'UK'})
df_datacenters['country'] = df_datacenters['country'].replace({'United States': 'USA', 'United Kingdom': 'UK'})
df_models_clean['country'] = df_models_clean['country'].replace({'United States of America': 'USA', 'United Kingdom': 'UK'})

df_datacenters = pd.merge(df_datacenters, df_elec, on='country', how='left')
df_datacenters['carbon_intensity'] = df_datacenters['carbon_intensity'].fillna(400)
df_datacenters['co2'] = (df_datacenters['megawatts'] * 1000) * 8760 * df_datacenters['carbon_intensity'] / 1_000_000

# ==========================================
# 2. KPI CALCULATIONS
# ==========================================
total_models_kpi = int(df_models_clean['Model'].nunique())
total_dc_mw_kpi = df_datacenters['megawatts'].sum()
total_co2_mt_kpi = df_datacenters['co2'].sum() / 1_000_000
model_count_2024 = df_models_clean[df_models_clean['pub_year'] == 2024].shape[0]
model_count_2023 = df_models_clean[df_models_clean['pub_year'] == 2023].shape[0]
growth_pct_kpi = ((model_count_2024 - model_count_2023) / max(model_count_2023, 1)) * 100

# ==========================================
# 3. CHART 5 DATA: CO2 per Country
# ==========================================
df_co2_country = df_datacenters.groupby('country').agg(
    total_mw=('megawatts', 'sum'),
    avg_carbon=('carbon_intensity', 'mean'),
    total_co2=('co2', 'sum')
).reset_index()
df_co2_country = df_co2_country[df_co2_country['total_co2'] > 0]
df_co2_country = df_co2_country.sort_values('total_co2', ascending=False).head(15)
df_co2_country = df_co2_country.sort_values('total_co2', ascending=True)
# display_value drives bar length — starts as total_co2, updated by JS sort callback
df_co2_country['display_value'] = df_co2_country['total_co2'].copy()
source_co2_country = ColumnDataSource(df_co2_country)

# ==========================================
# 4. CHART 6 DATA: Compute Demand Timeline
# ==========================================
yearly_records = []
for yr in sorted(df_models_clean[df_models_clean['pub_year'] >= 2012]['pub_year'].unique()):
    df_yr = df_models_clean[df_models_clean['pub_year'] == yr]
    model_count = len(df_yr)

    # Model with largest FLOPs in that year
    df_yr_flops = df_yr[df_yr['flops'] > 0]
    if not df_yr_flops.empty:
        max_flop_idx = df_yr_flops['flops'].idxmax()
        max_flop_row = df_yr_flops.loc[max_flop_idx]
        max_flop_model = max_flop_row['Model']
        max_log_flops = max_flop_row['log_flops']
    else:
        max_flop_model = "-"
        max_log_flops = 0.0

    # Model with largest parameters in that year
    df_yr_params = df_yr[df_yr['params'] > 0]
    if not df_yr_params.empty:
        max_param_idx = df_yr_params['params'].idxmax()
        max_param_row = df_yr_params.loc[max_param_idx]
        max_param_model = max_param_row['Model']
        max_log_params = max_param_row['log_params']
    else:
        max_param_model = "-"
        max_log_params = 0.0

    yearly_records.append({
        'pub_year': yr,
        'model_count': model_count,
        'max_log_flops': max_log_flops,
        'max_log_params': max_log_params,
        'max_flop_model': max_flop_model,
        'max_param_model': max_param_model
    })
df_yearly = pd.DataFrame(yearly_records)
df_yearly['cumulative_models'] = df_yearly['model_count'].cumsum()
source_yearly = ColumnDataSource(df_yearly)

# ==========================================
# 5. EXISTING AGGREGATIONS (enhanced with CO2)
# ==========================================
models_by_company = df_models_clean.groupby('org').size().reset_index(name='model_count')
models_by_country = df_models_clean.groupby('country').size().reset_index(name='model_count')

dc_by_company = df_datacenters.groupby('company').agg(
    megawatts=('megawatts', 'sum'),
    carbon_intensity=('carbon_intensity', 'mean'),
    est_co2=('co2', 'sum')
).reset_index()
dc_by_country = df_datacenters.groupby('country').agg(
    megawatts=('megawatts', 'sum'),
    carbon_intensity=('carbon_intensity', 'mean'),
    est_co2=('co2', 'sum')
).reset_index()

matrix_company = pd.merge(dc_by_company, models_by_company, left_on='company', right_on='org', how='inner')
matrix_company['entity'] = matrix_company['company']
matrix_company['type'] = 'By Company'
matrix_company['company_name'] = matrix_company['company']

matrix_country = pd.merge(dc_by_country, models_by_country, on='country', how='inner')
matrix_country['entity'] = matrix_country['country']
matrix_country['type'] = 'By Country'
matrix_country['company_name'] = ""

# * Individual datacenters sites matrix representation
matrix_dc = df_datacenters.copy()
matrix_dc['entity'] = matrix_dc['title']
matrix_dc['type'] = 'All Datacenters'
matrix_dc['model_count'] = 0
matrix_dc['est_co2'] = matrix_dc['co2']
matrix_dc['company_name'] = matrix_dc['company']

df_matrix = pd.concat([
    matrix_company[['entity', 'megawatts', 'carbon_intensity', 'model_count', 'est_co2', 'type', 'company_name']],
    matrix_country[['entity', 'megawatts', 'carbon_intensity', 'model_count', 'est_co2', 'type', 'company_name']],
    matrix_dc[['entity', 'megawatts', 'carbon_intensity', 'model_count', 'est_co2', 'type', 'company_name']]
])
df_matrix.rename(columns={'megawatts': 'capacity_mw'}, inplace=True)
df_matrix = df_matrix.fillna(0)
df_matrix = df_matrix[df_matrix['capacity_mw'] > 0]
max_co2_matrix = df_matrix['est_co2'].max()
df_matrix['bubble_size'] = np.sqrt(df_matrix['est_co2'].clip(lower=1) / max(max_co2_matrix, 1)) * 44 + 8


source_models_master = ColumnDataSource(df_models_clean)
source_models_view = ColumnDataSource(df_models_clean)
# * Chart 00 source only includes models with recorded parameters
source_chart0_view = ColumnDataSource(df_models_clean[df_models_clean['params'] > 0])


df_map_clean = df_datacenters[(df_datacenters['x'] != 0) & (df_datacenters['y'] != 0)].fillna(0)
df_map_grouped = df_map_clean.groupby(['x', 'y']).agg(
    total_mw=('megawatts', 'sum'),
    facility_count=('title', 'count'),
    facilities=('title', lambda titles: ', '.join(titles.tolist())),
    all_companies=('company', lambda comps: ', '.join(sorted(set(
        c.strip() for cs in comps for c in cs.split(',') if c.strip()
    )))),
    country=('country', 'first'),
    carbon_intensity=('carbon_intensity', 'mean'),
    est_co2=('co2', 'sum'),
).reset_index()
df_map_grouped['bubble_size'] = np.sqrt(df_map_grouped['total_mw'].clip(lower=1)) * 0.2 + 2
source_map_master = ColumnDataSource(df_map_grouped)
source_map_view = ColumnDataSource(df_map_grouped)

source_matrix_master = ColumnDataSource(df_matrix)
source_matrix_view = ColumnDataSource(df_matrix[df_matrix['type'] == 'By Country'].copy())

# Initial data prep for Chart 1 with latest model per organization
idx_latest = df_models_clean.groupby('org')['pub_date'].idxmax()
df_latest_models = df_models_clean.loc[idx_latest, ['org', 'Model', 'pub_date']].rename(
    columns={'Model': 'latest_model', 'pub_date': 'latest_date'}
)
df_latest_models['latest_date_str'] = df_latest_models['latest_date'].dt.strftime('%b %Y')

org_counts_df = df_models_clean.groupby('org').size().reset_index(name='count')
org_summary = pd.merge(org_counts_df, df_latest_models, on='org', how='left')

top_orgs = org_summary.sort_values('count', ascending=False).head(15)
source_chart1 = ColumnDataSource(data=dict(
    org=top_orgs['org'].tolist(),
    count=top_orgs['count'].tolist(),
    latest_model=top_orgs['latest_model'].tolist(),
    latest_date_str=top_orgs['latest_date_str'].tolist()
))

# Pre-compute monthly data for Chart 2
df_models_clean['year_month'] = df_models_clean['pub_date'].dt.to_period('M')
df_models_clean['month_start'] = df_models_clean['year_month'].apply(lambda x: x.start_time)

cutoff = pd.Timestamp(f'{MIN_PUBLISH_YEAR}-01-01')
df_monthly_base = df_models_clean.groupby('month_start').agg(
    monthly_log_params=('log_params', 'sum'),
    monthly_neg_flops=('negative_log_flops', 'sum'),
    model_count=('Model', 'count')
).reset_index()
df_monthly_filt = df_monthly_base[df_monthly_base['month_start'] >= cutoff]
source_monthly_view = ColumnDataSource(data=dict(
    month_start=df_monthly_filt['month_start'].tolist(),
    monthly_log_params=df_monthly_filt['monthly_log_params'].tolist(),
    monthly_neg_flops=df_monthly_filt['monthly_neg_flops'].tolist(),
    model_count=df_monthly_filt['model_count'].tolist()
))


# ==========================================
# 6. BUILD CHARTS
# ==========================================

# ---- CHART 00: Model Landscape Scatter Plot (Brushing & Selection) ----
p0 = figure(
    title="The AI Model Cosmos: Size, Release Date & Accessibility",
    x_axis_type="datetime",
    y_axis_type="log",
    height=400, width=1050,
    sizing_mode="stretch_width",
    toolbar_location="above",
    background_fill_color="#16213e",
    border_fill_color="#16213e",
    tools="lasso_select,box_select,poly_select,pan,wheel_zoom,reset,save",
    active_drag="lasso_select"
)
p0.title.text_color = "#e0e0e0"
p0.title.text_font_size = "15pt"
p0.title.text_font_style = "bold"
p0.title.align = "center"
p0.xaxis.axis_label = "Publication Date"
p0.xaxis.axis_label_text_color = "#b0b0b0"
p0.xaxis.major_label_text_color = "#b0b0b0"
p0.yaxis.axis_label = "Model Parameters (Log\u2081\u2080 Scale)"
p0.yaxis.axis_label_text_color = "#b0b0b0"
p0.yaxis.major_label_text_color = "#b0b0b0"

p0.xaxis.formatter = DatetimeTickFormatter(
    years="%Y",
    months="%b %Y",
    days="%d %b %Y"
)

p0.xgrid.grid_line_color = "#2a2a4a"
p0.ygrid.grid_line_color = "#2a2a4a"
p0.outline_line_color = None

# Custom styling for selected/non-selected glyphs
c0_scatter = p0.scatter(
    x='pub_date', y='params', size=8, color='color',
    source=source_chart0_view, alpha=0.75,
    selection_color='#00d2ff', nonselection_alpha=0.15,
    hover_fill_color='#ffffff', hover_alpha=1.0,
    line_color=None
)

p0.add_tools(HoverTool(
    tooltips=[
        ("Model", "@Model"),
        ("Organization", "@org"),
        ("Date", "@pub_date{%F}"),
        ("Parameters", "@params_str"),
        ("Training Compute", "@flops_str FLOPs"),
        ("Accessibility", "@accessibility_simple")
    ],
    formatters={'@pub_date': 'datetime'},
    renderers=[c0_scatter]
))

# ---- CHART 1: Top Organizations ----
orgs_list = list(source_chart1.data['org'])[::-1]
p1 = figure(
    y_range=orgs_list,
    title="Top 15 Organizations by AI Model Count",
    height=400, width=1050,
    sizing_mode="stretch_width",
    toolbar_location=None,
    background_fill_color="#16213e",
    border_fill_color="#16213e",
)
p1.title.text_color = "#e0e0e0"
p1.title.text_font_size = "15pt"
p1.title.text_font_style = "bold"
p1.title.align = "center"
p1.xaxis.axis_label = "Number of AI Models"
p1.xaxis.axis_label_text_color = "#b0b0b0"
p1.xaxis.major_label_text_color = "#b0b0b0"
p1.yaxis.major_label_text_color = "#d0d0d0"
p1.xgrid.grid_line_color = "#2a2a4a"
p1.ygrid.grid_line_color = None
p1.outline_line_color = None
bars1 = p1.hbar(
    y='org', right='count', source=source_chart1, height=0.68,
    color="#00d2ff", hover_fill_color="#7ee8ff", hover_alpha=0.95,
    line_color=None
)
p1.add_tools(HoverTool(tooltips=[
    ("Organization", "@org"),
    ("AI Models Published", "@count"),
    ("Latest Model", "@latest_model"),
    ("Release Date", "@latest_date_str")
], renderers=[bars1]))

# ---- CHART 2: Monthly Params vs FLOPs ----
monthly_agg_label = Div(text="""
<div style="padding: 4px 20px 0px 0px; display: inline-block;">
    <span style="font-size: 9pt; font-weight: 700; color: #58a6ff; font-family: 'Inter', sans-serif;">
        Aggregation:
    </span>
</div>
""")
monthly_agg_select = Select(title="", options=["Sum", "Average", "Maximum"], value="Sum", width=140)

p2 = figure(
    x_axis_type="datetime",
    title="Parameter Scale vs. Training Compute Cost (Monthly)",
    height=400, width=1050,
    sizing_mode="stretch_width",
    toolbar_location="above",
    background_fill_color="#16213e",
    border_fill_color="#16213e",
    x_range=(pd.Timestamp('2012-01-01'), pd.Timestamp('2026-12-31')),
    tools="xpan,xbox_zoom,xwheel_zoom,reset,save"
)
p2.title.text_color = "#e0e0e0"
p2.title.text_font_size = "15pt"
p2.title.text_font_style = "bold"
p2.title.align = "center"
p2.xaxis.axis_label = "Publication Date"
p2.xaxis.axis_label_text_color = "#b0b0b0"
p2.xaxis.major_label_text_color = "#b0b0b0"
p2.yaxis.axis_label = "Log\u2081\u2080 Scale"
p2.yaxis.axis_label_text_color = "#b0b0b0"
p2.yaxis.major_label_text_color = "#b0b0b0"
p2.yaxis.formatter = CustomJSTickFormatter(code="""
    const absTick = Math.abs(tick);
    if (absTick === 0) return "1";
    const rounded = Math.round(absTick);
    if (Math.abs(absTick - rounded) < 0.0001) {
        const superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        };
        const str = rounded.toString();
        let result = "10";
        for (let i = 0; i < str.length; i++) {
            result += superscripts[str[i]] || str[i];
        }
        return result;
    }
    return "10^" + absTick.toFixed(1);
""")
p2.xgrid.grid_line_color = "#2a2a4a"
p2.ygrid.grid_line_color = "#2a2a4a"
p2.outline_line_color = None
p2.vbar(x='month_start', top='monthly_log_params', bottom=0,
        width=1000*60*60*24*28, source=source_monthly_view,
        color="#00d2ff", legend_label="Parameters (Log\u2081\u2080 — model size)",
        alpha=0.75, line_color=None)
p2.vbar(x='month_start', top=0, bottom='monthly_neg_flops',
        width=1000*60*60*24*28, source=source_monthly_view,
        color="#ff6b6b", legend_label="FLOPs / Training Compute (Log\u2081\u2080 — energy proxy)",
        alpha=0.75, line_color=None)
zero_line = Span(location=0, dimension='width', line_color='#ffffff', line_width=1.5, line_dash='dashed')
p2.add_layout(zero_line)
p2.add_tools(HoverTool(tooltips=[
    ("Month", "@month_start{%b %Y}"),
    ("Params (Log\u2081\u2080)", "@monthly_log_params{0.2f}"),
    ("FLOPs (Log\u2081\u2080)", "@monthly_neg_flops{0.2f}"),
    ("Models", "@model_count")
], formatters={'@month_start': 'datetime'}))
p2.legend.location = "top_left"
p2.legend.background_fill_color = "#1a1a2e"
p2.legend.background_fill_alpha = 0.8
p2.legend.label_text_color = "#d0d0d0"
p2.legend.border_line_color = None

# Aggregation callback will be registered at the bottom using update_downstream_callback
chart2_header = row(monthly_agg_label, monthly_agg_select, margin=(0, 0, 4, 0), sizing_mode="stretch_width")
chart2_block = column(chart2_header, p2, sizing_mode="stretch_width")

# ---- CHART 3: Global Map ----
tile_source = WMTSTileSource(url="https://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png")
p3 = figure(
    x_axis_type="mercator", y_axis_type="mercator",
    title="Global AI Data Center Footprint — Capacity & Carbon Intensity",
    height=500, width=1050,
    sizing_mode="stretch_width",
    x_range=(-15000000, 15000000), y_range=(-5000000, 8000000),
    toolbar_location="below",
    background_fill_color="#16213e",
    border_fill_color="#16213e",
)
p3.title.text_color = "#ffffff"
p3.title.text_font_size = "15pt"
p3.title.text_font_style = "bold"
p3.title.align = "center"
p3.add_tile(tile_source)

# Map Theme Selector Widgets and Callbacks
chart3_theme_label = Div(text="""
<div style="padding: 4px 16px 0px 0px; display: inline-block;">
    <span style="font-size: 9pt; font-weight: 700; color: #58a6ff; font-family: 'Inter', sans-serif;">
        Map Theme:
    </span>
</div>
""")
chart3_theme_select = Select(
    title="",
    options=["Dark Matter", "Light Matter", "Open Street Map", "Satellite Earth"],
    value="Dark Matter",
    width=210
)
chart3_theme_callback = CustomJS(args=dict(
    tile_source=tile_source,
    selector=chart3_theme_select
), code="""
    const theme = selector.value;
    const theme_map = {
        "Dark Matter": "dark",
        "Light Matter": "voyager",
        "Open Street Map": "osm",
        "Satellite Earth": "satellite"
    };
    const theme_val = theme_map[theme] || "dark";
    let url = "";
    if (theme_val === "dark") {
        url = "https://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png";
    } else if (theme_val === "voyager") {
        url = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png";
    } else if (theme_val === "osm") {
        url = "https://a.tile.openstreetmap.org/{Z}/{X}/{Y}.png";
    } else if (theme_val === "satellite") {
        url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}";
    }
    tile_source.url = url;
""")
chart3_theme_select.js_on_change('value', chart3_theme_callback)
chart3_header = row(chart3_theme_label, chart3_theme_select, margin=(0, 0, 4, 0), sizing_mode="stretch_width")
chart3_block = column(chart3_header, p3, sizing_mode="stretch_width")

p3.outline_line_color = None
p3.xaxis.visible = False
p3.yaxis.visible = False
p3.xgrid.grid_line_color = None
p3.ygrid.grid_line_color = None
color_mapper_map = LinearColorMapper(palette=Turbo256, low=100, high=600)
map_scatter = p3.scatter(
    x='x', y='y', size='bubble_size',
    color={'field': 'carbon_intensity', 'transform': color_mapper_map},
    source=source_map_view, alpha=0.72, line_color='#ffffff', line_width=0.4
)
color_bar_map = ColorBar(
    color_mapper=color_mapper_map, ticker=BasicTicker(desired_num_ticks=6),
    title="Carbon Intensity (gCO\u2082/kWh)",
    title_text_color="#e0e0e0", title_text_font_size="10pt", title_text_font_style="bold",
    title_standoff=10,
    major_label_text_color="#e0e0e0", major_label_text_font_size="9pt",
    background_fill_color="#16213e", background_fill_alpha=0.95,
    label_standoff=8, width=22, height=420, padding=8,
)
p3.add_layout(color_bar_map, 'right')
p3.add_tools(HoverTool(
    tooltips=[
        ("Facility", "@facilities"),
        ("Country", "@country"),
        ("Total Capacity", "@total_mw{0,0} MW"),
        ("Sites at Location", "@facility_count"),
        ("Carbon Intensity", "@carbon_intensity{0} gCO\u2082/kWh"),
        ("Est. Annual CO\u2082", "@est_co2{0,0} tons/yr")
    ],
    renderers=[map_scatter]
))

# ---- CHART 4: Capacity vs Carbon (enhanced) ----
df_company_view = df_matrix[df_matrix['type'] == 'By Company']
med_mw = df_company_view['capacity_mw'].mean() if not df_company_view.empty else 1000
# med_carbon = df_company_view['carbon_intensity'].mean() if not df_company_view.empty else 350
med_carbon = 100

p4 = figure(
    title="Compute Capacity vs. Grid Carbon Intensity",
    x_axis_label="Total Capacity (MW) (Log Scale)",
    y_axis_label="Avg Carbon Intensity (gCO\u2082/kWh)",
    x_axis_type="log",
    height=500, width=1050,
    sizing_mode="stretch_width",
    toolbar_location="above",
    background_fill_color="#16213e",
    border_fill_color="#16213e",
)
p4.title.text_color = "#e0e0e0"
p4.title.text_font_size = "15pt"
p4.title.text_font_style = "bold"
p4.title.align = "center"
p4.xaxis.axis_label_text_color = "#b0b0b0"
p4.xaxis.major_label_text_color = "#b0b0b0"
p4.yaxis.axis_label_text_color = "#b0b0b0"
p4.yaxis.major_label_text_color = "#b0b0b0"
p4.xgrid.grid_line_color = "#2a2a4a"
p4.ygrid.grid_line_color = "#2a2a4a"
p4.outline_line_color = None
p4.y_range = Range1d(start=0, end=700)

# Quadrant reference lines
vline4 = Span(location=med_mw, dimension='height', line_color='#484f58', line_width=1.5, line_dash='dashed')
hline4 = Span(location=med_carbon, dimension='width', line_color='#00f0a0', line_width=1.5, line_dash='dashed')
p4.add_layout(vline4)
p4.add_layout(hline4)

# Reference line labels
vline4_label = Label(
    x=med_mw,
    y=550,
    text=f"Average Capacity ({med_mw:.0f} MW)",
    text_color='#8b949e',
    text_font_size='9pt',
    text_font_style='bold',
    angle=-1.5708,
    x_offset=-12,
    y_offset=0,
    text_font='Inter, sans-serif'
)
hline4_label = Label(
    x=30,
    y=med_carbon,
    text="EU Taxonomy Threshold (100 gCO₂/kWh)",
    text_color='#00f0a0',
    text_font_size='9pt',
    text_font_style='bold',
    x_offset=0,
    y_offset=5,
    text_font='Inter, sans-serif'
)
p4.add_layout(vline4_label)
p4.add_layout(hline4_label)

# # ? Diagonal boundary line for "clean energy" line using threshold grid factor
# E_threshold = 100
# # ? Boundary line matrix would be Y = (8760 * E_threshold) * X
# clean_energy_slope = (8760 * E_threshold)
# clean_energy_line_y = clean_energy_slope * df_company_view['capacity_mw']

# # Add clean energy slope line
# clean_energy_df = pd.DataFrame({
#     'x': df_company_view['capacity_mw'],
#     'y': clean_energy_line_y
# }).sort_values('x')

# p4.line(
#     x=clean_energy_df['x'],
#     y=clean_energy_df['y'],
#     line_color='#ff0000',
#     line_width=1.5,
#     line_dash='solid',
# )


# Color by carbon intensity
color_mapper_4 = LinearColorMapper(palette=Turbo256, low=100, high=600)
scatter4 = p4.scatter(
    x='capacity_mw', y='carbon_intensity', size='bubble_size',
    source=source_matrix_view,
    color={'field': 'carbon_intensity', 'transform': color_mapper_4},
    alpha=0.85, line_color='white', line_width=0.6
)
labels4 = LabelSet(
    x='capacity_mw', y='carbon_intensity', text='entity',
    source=source_matrix_view,
    x_offset=7, y_offset=5,
    text_color='#c9d1d9', text_font_size='8pt',
    text_font='Inter, sans-serif'
)
p4.add_layout(labels4)
# ColorBar for Chart 4
color_bar_4 = ColorBar(
    color_mapper=color_mapper_4,
    ticker=BasicTicker(desired_num_ticks=6),
    title="Carbon Intensity (gCO\u2082/kWh) \u2192 bluer/greener = cleaner",
    title_text_color="#e0e0e0", title_text_font_size="9pt", title_text_font_style="bold",
    title_standoff=10,
    major_label_text_color="#e0e0e0", major_label_text_font_size="9pt",
    background_fill_color="#16213e", background_fill_alpha=0.95,
    label_standoff=8, width=22, height=400, padding=8,
)
p4.add_layout(color_bar_4, 'right')
p4.add_tools(HoverTool(
    tooltips=[
        ("Entity", "@entity"),
        ("Capacity", "@capacity_mw{0,0} MW"),
        ("AI Models", "@model_count"),
        ("Carbon Intensity", "@carbon_intensity{0.1f} gCO\u2082/kWh"),
        ("Est. CO\u2082/year", "@est_co2{0,0} tons")
    ],
    renderers=[scatter4]
))

# Chart 4 Toggle Widget and Layout
chart4_label = Div(text="""
<div style="padding: 4px 16px 0px 0px; display: inline-block;">
    <span style="font-size: 9pt; font-weight: 700; color: #58a6ff; font-family: 'Inter', sans-serif;">
        Aggregation:
    </span>
</div>
""")
matrix_toggle = Select(
    title="",
    options=["All Datacenters", "By Company", "By Country"],
    value="By Country",
    width=180,
    margin=(0, 0, 4, 0)
)

# Matrix toggle callback will be registered at the bottom using update_downstream_callback

chart4_header = row(chart4_label, matrix_toggle, margin=(0, 0, 4, 0), sizing_mode="stretch_width")
chart4_block = column(chart4_header, p4, sizing_mode="stretch_width")

# ---- CHART 5: CO2 per Country ----
# Sort selector for Chart 5
chart5_sort_label = Div(text="""
<div style="padding: 4px 20px 0px 0px; display: inline-block;">
    <span style="font-size: 9pt; font-weight: 700; color: #58a6ff; font-family: 'Inter', sans-serif;">
        Sort by:
    </span>
</div>
""")
chart5_sort_select = Select(
    title="",
    options=[
        "CO₂ Emissions (tons/yr)",
        "DC Capacity (MW)",
        "Carbon Intensity (gCO₂/kWh)"
    ],
    value="CO₂ Emissions (tons/yr)",
    width=210
)

co2_countries = df_co2_country['country'].tolist()
p5 = figure(
    y_range=co2_countries,
    title="Estimated Annual CO\u2082 Emissions from AI Data Centers \u2014 Top 15 Countries",
    height=430, width=1050,
    sizing_mode="stretch_width",
    toolbar_location=None,
    background_fill_color="#16213e",
    border_fill_color="#16213e",
)
p5.title.text_color = "#e0e0e0"
p5.title.text_font_size = "15pt"
p5.title.text_font_style = "bold"
p5.title.align = "center"
p5.xaxis.axis_label = "Estimated CO\u2082 Emissions (tons/year)"
p5.xaxis.axis_label_text_color = "#b0b0b0"
p5.xaxis.major_label_text_color = "#b0b0b0"
p5.xaxis.formatter = NumeralTickFormatter(format="0,0")
p5.yaxis.major_label_text_color = "#d0d0d0"
p5.xgrid.grid_line_color = "#2a2a4a"
p5.ygrid.grid_line_color = None
p5.outline_line_color = None
color_mapper_co2 = LinearColorMapper(
    palette=Turbo256,
    low=0,
    high=df_co2_country['display_value'].max()
)
bars5 = p5.hbar(
    y='country', right='display_value', height=0.65,
    source=source_co2_country,
    color={'field': 'display_value', 'transform': color_mapper_co2},
    line_color=None, alpha=0.9
)
color_bar_5 = ColorBar(
    color_mapper=color_mapper_co2,
    ticker=BasicTicker(desired_num_ticks=5),
    title="Ton CO\u2082/year",
    title_text_color="#e0e0e0", title_text_font_size="9pt", title_text_font_style="bold",
    title_standoff=8,
    major_label_text_color="#e0e0e0", major_label_text_font_size="9pt",
    background_fill_color="#16213e", background_fill_alpha=0.95,
    label_standoff=8, width=22, height=330, padding=8,
)
p5.add_layout(color_bar_5, 'right')
p5.add_tools(HoverTool(tooltips=[
    ("Country", "@country"),
    ("Total DC Capacity", "@total_mw{0,0} MW"),
    ("Avg Carbon Intensity", "@avg_carbon{0.0} gCO\u2082/kWh"),
    ("Est. CO\u2082 per Year", "@total_co2{0,0} tons")
], renderers=[bars5]))

# Chart 5 sort callback
# Sort callback will be registered at the bottom using update_downstream_callback
chart5_header = row(chart5_sort_label, chart5_sort_select, margin=(0, 0, 4, 0), sizing_mode="stretch_width")
chart5_block = column(chart5_header, p5, sizing_mode="stretch_width")

# ---- CHART 6: Compute Demand vs Model Growth Timeline ----
max_cumul = int(df_yearly['cumulative_models'].max()) if not df_yearly.empty else 3000
valid_flops = df_yearly['max_log_flops'][df_yearly['max_log_flops'] > 0]
max_flops_val = float(valid_flops.max()) if not valid_flops.empty else 30.0

p6 = figure(
    title="AI Model Growth vs. Maximum Training Compute Per Year",
    x_axis_label="Year",
    y_axis_label="Cumulative Models Published",
    height=430, width=1050,
    sizing_mode="stretch_width",
    toolbar_location="above",
    background_fill_color="#16213e",
    border_fill_color="#16213e",
    x_range=(2011.5, 2026.5),
    y_range=(0, max_cumul * 1.12),
    tools="xpan,xbox_zoom,xwheel_zoom,reset,save"
)
p6.title.text_color = "#e0e0e0"
p6.title.text_font_size = "15pt"
p6.title.text_font_style = "bold"
p6.title.align = "center"
p6.xaxis.axis_label_text_color = "#b0b0b0"
p6.xaxis.major_label_text_color = "#b0b0b0"
p6.xaxis.ticker = FixedTicker(ticks=list(range(2012, 2027)))
p6.yaxis.axis_label_text_color = "#00d2ff"
p6.yaxis.major_label_text_color = "#b0b0b0"
p6.xgrid.grid_line_color = "#2a2a4a"
p6.ygrid.grid_line_color = "#2a2a4a"
p6.outline_line_color = None
# Secondary y-axis for FLOPs
p6.extra_y_ranges = {"flops_range": Range1d(start=0, end=max_flops_val * 1.2)}
p6.add_layout(LinearAxis(
    y_range_name="flops_range",
    axis_label="Max Training Compute (Log\u2081\u2080 FLOPs \u2014 energy proxy)",
    axis_label_text_color="#ff6b6b",
    major_label_text_color="#b0b0b0",
), 'right')
# Cumulative models area + line
p6.varea(x='pub_year', y1=0, y2='cumulative_models',
         source=source_yearly, color="#00d2ff", alpha=0.12)
line6_cumul = p6.line(
    x='pub_year', y='cumulative_models',
    source=source_yearly, color="#00d2ff",
    line_width=3, legend_label="Cumulative AI Models Published"
)
scatter6_cumul = p6.scatter(
    x='pub_year', y='cumulative_models',
    source=source_yearly, color="#00d2ff", size=9
)
# Max FLOPs line (energy proxy)
p6.varea(x='pub_year', y1=0, y2='max_log_flops',
         source=source_yearly, color="#ff6b6b", alpha=0.10, y_range_name="flops_range")
line6_flops = p6.line(
    x='pub_year', y='max_log_flops',
    source=source_yearly, color="#ff6b6b",
    line_width=3, y_range_name="flops_range",
    legend_label="Max Training FLOPs (Log\u2081\u2080 \u2014 energy demand proxy)"
)
scatter6_flops = p6.scatter(
    x='pub_year', y='max_log_flops',
    source=source_yearly, color="#ff6b6b", size=9, y_range_name="flops_range"
)
# Milestone vertical lines
milestones = [(2020, 'GPT-3'), (2022, 'ChatGPT'), (2023, 'GPT-4')]
for yr, lbl in milestones:
    ms_span = Span(location=yr, dimension='height', line_color='#f0c040', line_width=1.5, line_dash='dotted')
    p6.add_layout(ms_span)
    ms_lbl = Label(
        x=yr + 0.04, y=max_cumul * 0.96,
        text=lbl, text_color='#f0c040',
        text_font_size='9pt', text_font_style='bold',
        text_font='Inter, sans-serif',
        angle=1.5708, # 90 degrees in radians
        background_fill_color='#0d1117', background_fill_alpha=0.7
    )
    p6.add_layout(ms_lbl)
p6.legend.location = "top_left"
p6.legend.background_fill_color = "#1a1a2e"
p6.legend.background_fill_alpha = 0.85
p6.legend.label_text_color = "#d0d0d0"
p6.legend.border_line_color = None
p6.legend.click_policy = "hide"  # click legend to toggle lines

# Separate hover tooltips for blue (models) and red (flops)
hover_cumul = HoverTool(
    tooltips=[
        ("Year", "@pub_year"),
        ("New Models That Year", "@model_count"),
        ("Cumulative Total", "@cumulative_models")
    ],
    renderers=[scatter6_cumul],
    mode="mouse"
)
hover_flops = HoverTool(
    tooltips=[
        ("Year", "@pub_year"),
        ("Max FLOPs (Log₁₀)", "@max_log_flops{0.1f}"),
        ("Largest Model (by FLOPs)", "@max_flop_model")
    ],
    renderers=[scatter6_flops],
    mode="mouse"
)
p6.add_tools(hover_cumul, hover_flops)

# Chart 6 line visibility toggle
chart6_toggle_label = Div(text="""
<div style="padding: 4px 16px 0px 0px; display: inline-block;">
    <span style="font-size: 9pt; font-weight: 700; color: #58a6ff; font-family: 'Inter', sans-serif;">
        Show lines:
    </span>
</div>
""")
chart6_toggle = CheckboxGroup(
    labels=["Cumulative AI Models (blue)", "Max Training FLOPs / Energy (red)"],
    active=[0, 1],
    inline=True,
    stylesheets=["""
        :host { font-family: 'Inter', sans-serif; }
        .bk-input-group label { color: #c9d1d9 !important; font-size: 9pt !important; font-weight: 500 !important; }
    """]
)

# Collect all 6 glyphs that belong to cumul (blue) and flops (red) groups
_all_p6_renderers = p6.renderers
# The renderers in order: varea_cumul, line_cumul, scatter_cumul, varea_flops, line_flops, scatter_flops
_p6_cumul_renderers = [r for r in _all_p6_renderers if hasattr(r, 'glyph') and
    getattr(getattr(r, 'glyph', None), 'line_color', None) == '#00d2ff' or
    (hasattr(r, 'glyph') and type(r.glyph).__name__ == 'VArea' and
     getattr(r.glyph, 'fill_color', '') == '#00d2ff')]

chart6_toggle_callback = CustomJS(args=dict(
    renderers=p6.renderers,
    toggle=chart6_toggle
), code="""
    const active = toggle.active;
    const show_cumul = active.includes(0);
    const show_flops = active.includes(1);
    // renderers order: varea_cumul, line_cumul, scatter_cumul, varea_flops, line_flops, scatter_flops
    // plus milestone spans & labels at end — only touch first 6
    const n = renderers.length;
    const chart_renderers = renderers.slice(0, 6);
    for (let i = 0; i < chart_renderers.length; i++) {
        if (i < 3) {
            chart_renderers[i].visible = show_cumul;
        } else {
            chart_renderers[i].visible = show_flops;
        }
    }
""")
chart6_toggle.js_on_change('active', chart6_toggle_callback)
chart6_header = row(chart6_toggle_label, chart6_toggle, margin=(0, 0, 4, 0), sizing_mode="stretch_width")
chart6_block = column(chart6_header, p6, sizing_mode="stretch_width")

# ==========================================
# 7. SIDEBAR
# ==========================================

kpi_cards = Div(text=f"""
<div style="padding: 12px 14px 6px 14px;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-bottom: 2px;">
        <div style="background: linear-gradient(135deg,#1a2d4a,#0d1117); border:1px solid #1e4976;
                    border-radius:8px; padding:10px 10px; text-align:center;">
            <div style="font-size:17pt;font-weight:800;color:#00d2ff;font-family:'Inter',sans-serif;line-height:1.1;">{total_models_kpi:,}</div>
            <div style="font-size:7pt;color:#8b949e;font-family:'Inter',sans-serif;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px;">AI Models</div>
        </div>
        <div style="background:linear-gradient(135deg,#1a2d4a,#0d1117);border:1px solid #1e4976;
                    border-radius:8px;padding:10px 10px;text-align:center;">
            <div style="font-size:17pt;font-weight:800;color:#00d2ff;font-family:'Inter',sans-serif;line-height:1.1;">{int(total_dc_mw_kpi/1000):,}GW</div>
            <div style="font-size:7pt;color:#8b949e;font-family:'Inter',sans-serif;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px;">DC Capacity</div>
        </div>
        <div style="background:linear-gradient(135deg,#2d1a1a,#0d1117);border:1px solid #762a1e;
                    border-radius:8px;padding:10px 10px;text-align:center;">
            <div style="font-size:17pt;font-weight:800;color:#ff6b6b;font-family:'Inter',sans-serif;line-height:1.1;">{total_co2_mt_kpi:.0f}Mt</div>
            <div style="font-size:7pt;color:#8b949e;font-family:'Inter',sans-serif;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px;">Est. CO\u2082/yr</div>
        </div>
        <div style="background:linear-gradient(135deg,#2a2d1a,#0d1117);border:1px solid #5a6e1e;
                    border-radius:8px;padding:10px 10px;text-align:center;">
            <div style="font-size:17pt;font-weight:800;color:#a8d862;font-family:'Inter',sans-serif;line-height:1.1;">+{growth_pct_kpi:.0f}%</div>
            <div style="font-size:7pt;color:#8b949e;font-family:'Inter',sans-serif;margin-top:2px;text-transform:uppercase;letter-spacing:0.5px;">Growth '23\u2192'24</div>
        </div>
    </div>
</div>
""", width=310)

sidebar_header = Div(text="""
<div style="background:#161b22;padding:6px 0;border:1px solid #30363d;border-radius:6px;width:240px;text-align:center;box-sizing:border-box;">
    <span style="font-size:11pt;font-weight:700;color:#e6edf3;font-family:'Inter',sans-serif;display:block;text-align:center;width:100%;">
        &#9881;&nbsp; Dashboard Controls
    </span>
</div>
""", width=240, margin=(4, 0, 4, 20))


filter_group1_label = Div(text="""
<div style="padding:0px 20px 2px 20px;">
    <span style="font-size:8.5pt;font-weight:700;color:#58a6ff;text-transform:uppercase;letter-spacing:1.5px;font-family:'Inter',sans-serif;">
        &#129482; Model Filters
    </span>
</div>
""", height=20)

_mc_style = ["""
    .choices__inner {
        background-color: white !important; color: black !important;
        border: 1px solid #ccc !important; position: relative !important;
        padding-right: 30px !important;
    }
    .choices__inner::after {
        content: "▼" !important; font-size: 10px !important; color: #555 !important;
        position: absolute !important; right: 12px !important; top: 50% !important;
        transform: translateY(-50%) !important; pointer-events: none !important;
    }
    .choices__list--dropdown { background-color: white !important; }
    .choices__list--dropdown .choices__item { color: black !important; }
    .choices__input { color: black !important; background-color: white !important; }
"""]

access_filter = MultiChoice(
    title="Accessibility:", options=["Open Weight", "Closed"],
    value=[], width=240, stylesheets=_mc_style,
    margin=(0, 0, 2, 20)
)

filter_group2_label = Div(text="""
<div style="padding:14px 20px 2px 20px;">
    <span style="font-size:8.5pt;font-weight:700;color:#58a6ff;text-transform:uppercase;letter-spacing:1.5px;font-family:'Inter',sans-serif;">
        &#128207; Parameters Settings
    </span>
</div>
""", height=32)

min_y = MIN_PUBLISH_YEAR
max_y = MAX_PUBLISH_YEAR
date_slider = RangeSlider(start=min_y, end=max_y, value=(min_y, max_y), step=1, title="Publish Year", width=240, margin=(0, 0, 2, 20))
max_p = df_models_clean['params'].max() / 1e9 if not df_models_clean.empty else 1000
param_slider = RangeSlider(start=0, end=max_p, value=(0, max_p), step=1, title="Parameters (Billions)", width=240, margin=(0, 0, 2, 20))

filter_group3_label = Div(text="""
<div style="padding:4px 20px 2px 20px;">
    <span style="font-size:8.5pt;font-weight:700;color:#58a6ff;text-transform:uppercase;letter-spacing:1.5px;font-family:'Inter',sans-serif;">
        &#127760; Geographic Filters
    </span>
</div>
""", height=25)

all_countries_raw = df_models_clean['country'].dropna().tolist()
all_countries_set = set()
for entry in all_countries_raw:
    for c in str(entry).split(', '):
        c = c.strip()
        if c and c != 'Unknown':
            all_countries_set.add(c)
all_countries = sorted(list(all_countries_set))
country_select = MultiChoice(
    title="Countries:", options=all_countries, value=[],
    width=240, stylesheets=_mc_style,
    margin=(0, 0, 2, 20)
)
# * Centralized update callback for all downstream charts
update_downstream_callback = CustomJS(args=dict(
    s_models_view=source_models_view,
    s_chart0_view=source_chart0_view,
    s_monthly_view=source_monthly_view,
    s_chart1=source_chart1,
    s_map_master=source_map_master,
    s_map_view=source_map_view,
    s_matrix_master=source_matrix_master,
    s_matrix_view=source_matrix_view,
    s_co2_country=source_co2_country,
    s_yearly=source_yearly,
    monthly_sel=monthly_agg_select,
    matrix_sel=matrix_toggle,
    chart5_sort_sel=chart5_sort_select,
    p1_y_range=p1.y_range,
    p2_x_range=p2.x_range,
    p2_y_range=p2.y_range,
    p4_x_range=p4.x_range,
    p4_y_range=p4.y_range,
    p5_x_range=p5.x_range,
    p5_y_range=p5.y_range,
    p5_xaxis=p5.xaxis[0],
    p6_y_range=p6.y_range,
    p6_flops_range=p6.extra_y_ranges['flops_range'],
    vline4=vline4,
    vline4_label=vline4_label,
    labels4=labels4,
    color_mapper_co2=color_mapper_co2,
    date_sl=date_slider,
    countries=country_select
), code=r"""
    function matchCompany(active_orgs, dc_str) {
        if (!dc_str) return false;
        const dc_lower = dc_str.toLowerCase();
        const dc_tokens = dc_lower.replace(/[^a-z0-9]/g, ' ').split(/\s+/).filter(x => x.length > 2);
        
        for (const active_org of active_orgs) {
            const org_lower = active_org.toLowerCase();
            
            // Substring checks for major companies
            if (org_lower.includes("google") && dc_lower.includes("google")) return true;
            if (org_lower.includes("meta") && dc_lower.includes("meta")) return true;
            if (org_lower.includes("facebook") && dc_lower.includes("meta")) return true;
            if (org_lower.includes("facebook") && dc_lower.includes("facebook")) return true;
            if (org_lower.includes("openai") && dc_lower.includes("openai")) return true;
            if (org_lower.includes("microsoft") && dc_lower.includes("microsoft")) return true;
            if (org_lower.includes("amazon") && dc_lower.includes("amazon")) return true;
            if (org_lower.includes("nvidia") && dc_lower.includes("nvidia")) return true;
            if (org_lower.includes("apple") && dc_lower.includes("apple")) return true;
            if (org_lower.includes("tencent") && dc_lower.includes("tencent")) return true;
            if (org_lower.includes("baidu") && dc_lower.includes("baidu")) return true;
            if (org_lower.includes("huawei") && dc_lower.includes("huawei")) return true;
            if (org_lower.includes("bytedance") && dc_lower.includes("bytedance")) return true;
            if (org_lower.includes("ibm") && dc_lower.includes("ibm")) return true;
            if (org_lower.includes("xai") && dc_lower.includes("xai")) return true;
            if (org_lower.includes("anthropic") && dc_lower.includes("anthropic")) return true;
            if (org_lower.includes("mistral") && dc_lower.includes("mistral")) return true;
            
            // Word-by-word comparison for other companies
            const org_tokens = org_lower.replace(/[^a-z0-9]/g, ' ').split(/\s+/).filter(x => x.length > 2);
            for (let k = 0; k < org_tokens.length; k++) {
                const ot = org_tokens[k];
                if (ot === "university" || ot === "research" || ot === "corporation" || ot === "technology" || ot === "institute" || ot === "labs") continue;
                if (dc_tokens.includes(ot)) return true;
            }
        }
        return false;
    }

    const indices = s_chart0_view.selected.indices || [];
    const v_data = s_models_view.data;
    const c0_data = s_chart0_view.data;
    const aggMethod = monthly_sel.value;
    const selected_countries = countries.value || [];

    // 1. Determine active data based on selection in Chart 00
    let active_data = v_data;
    const has_selection = (indices.length > 0);
    const active_companies = new Set();
    const active_countries = new Set();

    if (has_selection) {
        const selected_models = new Set();
        for (let i = 0; i < indices.length; i++) {
            selected_models.add(c0_data['Model'][indices[i]]);
        }
        active_data = {};
        for (const key in v_data) { active_data[key] = []; }
        for (let i = 0; i < v_data['Model'].length; i++) {
            if (selected_models.has(v_data['Model'][i])) {
                for (const key in v_data) { active_data[key].push(v_data[key][i]); }
            }
        }
    }

    // Collect unique companies and countries from active data
    for (let i = 0; i < active_data['org'].length; i++) {
        active_companies.add(active_data['org'][i]);
    }
    for (let i = 0; i < active_data['country'].length; i++) {
        active_countries.add(active_data['country'][i]);
    }

    // 2. Update Chart 1 (Top Organizations)
    const org_counts = {};
    const org_latest_info = {};
    for (let i = 0; i < active_data['Model'].length; i++) {
        const org = active_data['org'][i];
        const model_name = active_data['Model'][i];
        const date_ms = active_data['pub_date_ms'][i];
        
        org_counts[org] = (org_counts[org] || 0) + 1;
        
        if (!org_latest_info[org] || date_ms > org_latest_info[org].max_date_ms) {
            const d = new Date(date_ms);
            const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            const date_str = monthNames[d.getUTCMonth()] + " " + d.getUTCFullYear();
            org_latest_info[org] = {
                max_date_ms: date_ms,
                model_name: model_name,
                date_str: date_str
            };
        }
    }

    let org_arr = [];
    for (const org in org_counts) {
        org_arr.push({
            org: org,
            count: org_counts[org],
            latest_model: org_latest_info[org] ? org_latest_info[org].model_name : '-',
            latest_date_str: org_latest_info[org] ? org_latest_info[org].date_str : '-'
        });
    }
    org_arr.sort((a, b) => b.count - a.count);
    org_arr = org_arr.slice(0, 15);

    const c1_data = {org: [], count: [], latest_model: [], latest_date_str: []};
    const new_y_range = [];
    for (let i = org_arr.length - 1; i >= 0; i--) {
        c1_data.org.push(org_arr[i].org);
        c1_data.count.push(org_arr[i].count);
        c1_data.latest_model.push(org_arr[i].latest_model);
        c1_data.latest_date_str.push(org_arr[i].latest_date_str);
        new_y_range.push(org_arr[i].org);
    }
    s_chart1.data = c1_data;
    s_chart1.change.emit();
    p1_y_range.factors = new_y_range;

    // 3. Update Chart 2 (Monthly Scale & Compute Timeline)
    const monthlyMap = {};
    for (let i = 0; i < active_data['pub_date_ms'].length; i++) {
        const d = new Date(active_data['pub_date_ms'][i]);
        const monthKey = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1);
        if (!monthlyMap[monthKey]) {
            monthlyMap[monthKey] = {params_sum: 0, flops_sum: 0, count: 0, params_max: -Infinity, flops_min: Infinity};
        }
        const lp = active_data['log_params'][i] || 0;
        const nf = active_data['negative_log_flops'][i] || 0;
        monthlyMap[monthKey].params_sum += lp;
        monthlyMap[monthKey].flops_sum += nf;
        monthlyMap[monthKey].count += 1;
        if (lp > monthlyMap[monthKey].params_max) monthlyMap[monthKey].params_max = lp;
        if (nf < monthlyMap[monthKey].flops_min) monthlyMap[monthKey].flops_min = nf;
    }

    const mKeys = Object.keys(monthlyMap).sort((a,b) => a - b);
    const monthData = {month_start:[], monthly_log_params:[], monthly_neg_flops:[], model_count:[]};
    for (const mk of mKeys) {
        const bucket = monthlyMap[mk];
        monthData.month_start.push(Number(mk));
        monthData.model_count.push(bucket.count);
        if (aggMethod === 'Sum') {
            monthData.monthly_log_params.push(bucket.params_sum);
            monthData.monthly_neg_flops.push(bucket.flops_sum);
        } else if (aggMethod === 'Maximum') {
            monthData.monthly_log_params.push(bucket.params_max === -Infinity ? 0 : bucket.params_max);
            monthData.monthly_neg_flops.push(bucket.flops_min === Infinity ? 0 : bucket.flops_min);
        } else {
            monthData.monthly_log_params.push(bucket.count > 0 ? bucket.params_sum/bucket.count : 0);
            monthData.monthly_neg_flops.push(bucket.count > 0 ? bucket.flops_sum/bucket.count : 0);
        }
    }
    s_monthly_view.data = monthData;
    s_monthly_view.change.emit();

    // Sync p2's x_range with the slider to match start date and end date
    const min_year = date_sl.value ? date_sl.value[0] : 2012;
    const max_year = date_sl.value ? date_sl.value[1] : 2026;
    p2_x_range.start = Date.UTC(min_year, 0, 1);
    p2_x_range.end = Date.UTC(max_year, 11, 31, 23, 59, 59);

    // 4. Update Chart 3 (Map Datacenters)
    const map_m = s_map_master.data;
    const map_v = {};
    for (const key in map_m) { map_v[key] = []; }

    for (let i = 0; i < map_m['country'].length; i++) {
        let ctry = map_m['country'][i];
        let comps_str = map_m['all_companies'][i] || '';
        
        let pass_ctry = (selected_countries.length === 0 || selected_countries.includes(ctry));
        
        let pass_comp = true;
        if (has_selection) {
            pass_comp = matchCompany(active_companies, comps_str);
        }
        
        if (pass_ctry && pass_comp) {
            for (const key in map_m) { map_v[key].push(map_m[key][i]); }
        }
    }
    s_map_view.data = map_v;
    s_map_view.change.emit();

    // 5. Update Chart 4 (Capacity vs Carbon Intensity Matrix)
    const t_val = matrix_sel.value;
    const matrix_m = s_matrix_master.data;
    const matrix_v = {};
    for (const key in matrix_m) { matrix_v[key] = []; }

    for (let i = 0; i < matrix_m['type'].length; i++) {
        if (matrix_m['type'][i] === t_val) {
            let entity = matrix_m['entity'][i];
            let pass_entity = true;
            if (has_selection) {
                if (t_val === 'By Company') {
                    pass_entity = matchCompany(active_companies, entity);
                } else if (t_val === 'By Country') {
                    pass_entity = active_countries.has(entity);
                } else if (t_val === 'All Datacenters') {
                    const comps_str = matrix_m['company_name'][i] || '';
                    pass_entity = matchCompany(active_companies, comps_str);
                }
            }
            if (pass_entity) {
                for (const key in matrix_m) { matrix_v[key].push(matrix_m[key][i]); }
            }
        }
    }
    s_matrix_view.data = matrix_v;
    s_matrix_view.change.emit();
    labels4.visible = (t_val !== 'All Datacenters');

    // Recalculate average capacity dynamically
    let sum = 0;
    let count = 0;
    for (let i = 0; i < matrix_v['capacity_mw'].length; i++) {
        const val = matrix_v['capacity_mw'][i];
        if (val > 0) {
            sum += val;
            count++;
        }
    }
    const avg_mw = count > 0 ? sum / count : 1000;
    vline4.location = avg_mw;
    vline4_label.x = avg_mw;
    vline4_label.text = "Average Capacity (" + Math.round(avg_mw) + " MW)";

    // 6. Update Chart 5 (CO2 by Country)
    const map_data = s_map_view.data;
    const countryMap = {};
    for (let i = 0; i < map_data['country'].length; i++) {
        const ctry = map_data['country'][i];
        if (!countryMap[ctry]) {
            countryMap[ctry] = {total_mw: 0, carbon_sum: 0, carbon_count: 0, total_co2: 0};
        }
        countryMap[ctry].total_mw += map_data['total_mw'][i] || 0;
        countryMap[ctry].carbon_sum += map_data['carbon_intensity'][i] || 0;
        countryMap[ctry].carbon_count += 1;
        countryMap[ctry].total_co2 += map_data['est_co2'][i] || 0;
    }

    let co2_arr = [];
    for (const ctry in countryMap) {
        if (countryMap[ctry].total_co2 > 0) {
            co2_arr.push({
                country: ctry,
                total_mw: countryMap[ctry].total_mw,
                avg_carbon: countryMap[ctry].carbon_count > 0 ? countryMap[ctry].carbon_sum / countryMap[ctry].carbon_count : 0,
                total_co2: countryMap[ctry].total_co2
            });
        }
    }

    const sort_map = {
        "CO₂ Emissions (tons/yr)": "total_co2",
        "DC Capacity (MW)": "total_mw",
        "Carbon Intensity (gCO₂/kWh)": "avg_carbon"
    };
    const sort_field = sort_map[chart5_sort_sel.value] || "total_co2";
    co2_arr.sort((a, b) => a[sort_field] - b[sort_field]);

    if (co2_arr.length > 15) {
        co2_arr = co2_arr.slice(co2_arr.length - 15);
    }

    const c5_data = {
        country: [],
        total_mw: [],
        avg_carbon: [],
        total_co2: [],
        display_value: []
    };
    const new_p5_y_factors = [];
    for (let i = 0; i < co2_arr.length; i++) {
        c5_data.country.push(co2_arr[i].country);
        c5_data.total_mw.push(co2_arr[i].total_mw);
        c5_data.avg_carbon.push(co2_arr[i].avg_carbon);
        c5_data.total_co2.push(co2_arr[i].total_co2);
        c5_data.display_value.push(co2_arr[i][sort_field]);
        new_p5_y_factors.push(co2_arr[i].country);
    }

    s_co2_country.data = c5_data;
    s_co2_country.change.emit();
    p5_y_range.factors = new_p5_y_factors;

    const maxVal = Math.max(...c5_data.display_value) || 10;
    p5_x_range.start = 0;
    p5_x_range.end = maxVal * 1.08;
    
    const labels = {
        'total_co2': 'Estimated CO\u2082 Emissions (tons/year)',
        'total_mw': 'Total DC Capacity (MW)',
        'avg_carbon': 'Avg Carbon Intensity (gCO\u2082/kWh)'
    };
    p5_xaxis.axis_label = labels[sort_field] || '';

    color_mapper_co2.low = 0;
    color_mapper_co2.high = maxVal;

    // 7. Update Chart 6 (Model Compute Timeline)
    const yearlyMap = {};
    for (let i = 0; i < active_data['pub_year'].length; i++) {
        const yr = active_data['pub_year'][i];
        if (!yearlyMap[yr]) {
            yearlyMap[yr] = {model_count: 0, max_log_flops: 0, max_flop_model: '-', max_log_params: 0, max_param_model: '-'};
        }
        yearlyMap[yr].model_count += 1;
        
        const log_flops = active_data['log_flops'][i] || 0;
        if (log_flops > yearlyMap[yr].max_log_flops) {
            yearlyMap[yr].max_log_flops = log_flops;
            yearlyMap[yr].max_flop_model = active_data['Model'][i];
        }
        
        const log_params = active_data['log_params'][i] || 0;
        if (log_params > yearlyMap[yr].max_log_params) {
            yearlyMap[yr].max_log_params = log_params;
            yearlyMap[yr].max_param_model = active_data['Model'][i];
        }
    }

    const yrs = [];
    for (let y = 2012; y <= 2026; y++) { yrs.push(y); }

    let cumulative = 0;
    const y_data = {
        pub_year: [],
        model_count: [],
        cumulative_models: [],
        max_log_flops: [],
        max_log_params: [],
        max_flop_model: [],
        max_param_model: []
    };

    for (let i = 0; i < yrs.length; i++) {
        const yr = yrs[i];
        const bucket = yearlyMap[yr] || {model_count: 0, max_log_flops: 0, max_flop_model: '-', max_log_params: 0, max_param_model: '-'};
        cumulative += bucket.model_count;
        y_data.pub_year.push(yr);
        y_data.model_count.push(bucket.model_count);
        y_data.cumulative_models.push(cumulative);
        y_data.max_log_flops.push(bucket.max_log_flops);
        y_data.max_log_params.push(bucket.max_log_params);
        y_data.max_flop_model.push(bucket.max_flop_model);
        y_data.max_param_model.push(bucket.max_param_model);
    }

    s_yearly.data = y_data;
    s_yearly.change.emit();

    p6_y_range.start = 0;
    p6_y_range.end = cumulative * 1.12 + 10;
    
    const max_f = Math.max(...y_data.max_log_flops) || 30;
    p6_flops_range.start = 0;
    p6_flops_range.end = max_f * 1.2;
""")

# * Callback triggered by sidebar filters
filter_models_callback = CustomJS(args=dict(
    s_models_master=source_models_master,
    s_models_view=source_models_view,
    s_chart0_view=source_chart0_view,
    date_sl=date_slider,
    param_sl=param_slider,
    countries=country_select,
    access=access_filter,
    update_downstream=update_downstream_callback
), code="""
    const min_year = date_sl.value ? date_sl.value[0] : 2012;
    const max_year = date_sl.value ? date_sl.value[1] : 2026;
    const min_param = param_sl.value ? param_sl.value[0] * 1e9 : 0;
    const max_param = param_sl.value ? param_sl.value[1] * 1e9 : 1e15;
    const selected_countries = countries.value || [];
    const selected_access = access.value || [];

    const m_data = s_models_master.data;
    const v_data = {};
    for (const key in m_data) { v_data[key] = []; }

    for (let i = 0; i < m_data['pub_date_ms'].length; i++) {
        let year = m_data['pub_year'][i];
        let param = m_data['params'][i];
        let ctry = m_data['country'][i] || '';
        let acc = m_data['accessibility_simple'][i] || '';
        
        let pass_date = (year >= min_year && year <= max_year);
        let pass_param = (param >= min_param && param <= max_param);
        let pass_ctry = (selected_countries.length === 0 || selected_countries.includes(ctry));
        let pass_acc = (selected_access.length === 0 || selected_access.includes(acc));
        
        if (pass_date && pass_param && pass_ctry && pass_acc) {
            for (const key in m_data) { v_data[key].push(m_data[key][i]); }
        }
    }
    s_models_view.data = v_data;
    s_models_view.change.emit();

    // Filter s_models_view where params > 0 for Chart 00
    const c0_data = {};
    for (const key in v_data) { c0_data[key] = []; }
    for (let i = 0; i < v_data['params'].length; i++) {
        if (v_data['params'][i] > 0) {
            for (const key in v_data) { c0_data[key].push(v_data[key][i]); }
        }
    }
    s_chart0_view.data = c0_data;
    s_chart0_view.change.emit();

    // Reset selection on Chart 00
    s_chart0_view.selected.indices = [];

    // Trigger downstream updates
    update_downstream.execute();
""")

# * Callback triggered by selecting points in Chart 00
chart0_selection_callback = CustomJS(args=dict(
    update_downstream=update_downstream_callback
), code="""
    update_downstream.execute();
""")

# Bind filters and selectors to callbacks
date_slider.js_on_change('value', filter_models_callback)
param_slider.js_on_change('value', filter_models_callback)
country_select.js_on_change('value', filter_models_callback)
access_filter.js_on_change('value', filter_models_callback)

source_chart0_view.selected.js_on_change('indices', chart0_selection_callback)

monthly_agg_select.js_on_change('value', update_downstream_callback)
matrix_toggle.js_on_change('value', update_downstream_callback)
chart5_sort_select.js_on_change('value', update_downstream_callback)

def _divider():
    return Div(text="""<hr style="border:0;border-top:1px solid #21262d;margin:4px 16px;">""", height=10)

sidebar = column(
    kpi_cards,
    _divider(),
    sidebar_header,
    _divider(),
    filter_group1_label,
    access_filter,
    filter_group2_label,
    date_slider,
    param_slider,
    _divider(),
    filter_group3_label,
    country_select,
    width=290,
    css_classes=["sidebar-panel"]
)

# ==========================================
# 8. NARRATIVE BLOCKS (placed BELOW each chart)
# ==========================================

def make_narrative(number, title, body, finding):
    return f"""
<div class="narrative-block">
    <div class="narrative-number">{number}</div>
    <div class="narrative-content">
        <h3 class="narrative-title">{title}</h3>
        <p class="narrative-body">{body}</p>
        <div class="narrative-finding">
            <span class="finding-label">&#128161; Key Finding</span>
            <span class="finding-text">{finding}</span>
        </div>
    </div>
</div>
"""

narrative_0 = make_narrative(
    "00",
    "The AI Model Cosmos: Size, Timeline, and Openness",
    "This interactive scatter plot maps individual AI models by their release date and parameter scale (log scale). The points are color-coded by openness: green for open weights and red for closed models. Use the free-form lasso or box select tools to drag and select a cluster of models; this will dynamically filter all subsequent charts to analyze only the selected models.",
    "Hover over any model to see details. Drag a lasso or box around points to filter the entire dashboard by those specific models."
)

narrative_1 = make_narrative(
    "01",
    "Who is Building Artificial Intelligence?",
    "Google DeepMind, Google, and NVIDIA lead global AI model development. This dominance is driven by direct ownership and access to large-scale compute infrastructure, rather than just algorithmic research superiority.",
    "Just 15 organizations produce the majority of global AI models &mdash; and all of them own or rely on massive data center infrastructure."
)

narrative_2 = make_narrative(
    "02",
    "The Explosion of AI Compute Costs",
    "This visualization compares AI model size (blue bars) against training compute demand (red bars). Since 2022, the energy consumption required to train new models has risen exponentially.",
    "AI training compute costs have exploded since 2022 &mdash; not only are there more models, but each new model is exponentially more energy-hungry."
)

narrative_3 = make_narrative(
    "03",
    "Physical Footprint of Global AI Infrastructure",
    "The global AI data center distribution map displays megawatt capacity (bubble size) and local grid emission levels (color). Currently, aggressive data center growth is expanding into countries with high-carbon power grids.",
    "344 data centers across 64 countries consume a total of ~154,000 MW &mdash; equivalent to running 154 large-scale power plants continuously, 24 hours a day."
)

narrative_4 = make_narrative(
    "04",
    "The Capacity vs. Carbon Intensity Matrix",
    "This scatter plot maps total compute capacity (log scale) against grid carbon intensity, utilizing a 100 gCO₂e/kWh horizontal line as the EU Taxonomy threshold for substantial climate mitigation contribution. Entities in the bottom-right quadrant represent high-capacity operations running on cleaner grids.",
    "The 100 gCO₂e/kWh line marks the threshold for clean energy; entities below this line utilize grids that meet strict low-carbon standards, while those to the right operate at massive scale."
)

narrative_5 = make_narrative(
    "05",
    "AI's Hidden Carbon Bill by Country",
    "Estimated annual carbon emissions are calculated based on data center capacity and local grid carbon intensity. Dominant capacity makes the USA the largest emitter, followed by countries relying heavily on fossil fuel grids.",
    "The USA accounts for the largest share of global AI infrastructure CO\u2082 emissions &mdash; but developing nations with dirty grids emit far more carbon per megawatt."
)

narrative_6 = make_narrative(
    "06",
    "Approaching the Energy Wall: AI Models vs. Compute Needs",
    "The trend lines compare the cumulative number of AI models (blue) with the peak training compute (red) per year. Compute demand is rising faster, driven by the release of large language models like GPT-3 and GPT-4.",
    "While the number of AI models grows rapidly, the energy required per model grows exponentially &mdash; we are racing toward a global compute energy wall."
)

# ==========================================
# 9. EMBED & HTML TEMPLATE
# ==========================================

script, divs = components({
    'sidebar': sidebar,
    'chart0': p0,
    'p1': p1,
    'chart2': chart2_block,
    'p3': chart3_block,
    'p4': chart4_block,
    'chart5': chart5_block,
    'chart6': chart6_block,
})

for key in divs:
    divs[key] = divs[key].replace('style="display: contents;"', '')

script = script.replace("docs_json = '", "docs_json = `").replace(
    "';\n        const render_items", "`;\n        const render_items"
)

footer_sources = "Data Sources: Epoch AI &nbsp;&middot;&nbsp; Electricity Maps &nbsp;&middot;&nbsp; AI Data Center Index"
footer_project = "Advanced Data Visualization &mdash; Final Project &mdash; 2026"

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Hidden Cost of Intelligence &mdash; AI Environmental Impact Dashboard</title>
    <meta name="description" content="Visualizing the environmental cost of AI's exponential growth: data centers, carbon footprint, and compute scale analysis.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&amp;display=swap" rel="stylesheet">
    {CDN.render()}
    <style>
        *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}

        html, body {{
            height: 100%;
            overflow: hidden;
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
        }}

        /* ===== GRID LAYOUT ===== */
        .dashboard-container {{
            display: grid;
            grid-template-columns: 290px 1fr;
            grid-template-rows: auto 1fr auto;
            grid-template-areas:
                "header header"
                "sidebar main"
                "footer footer";
            height: 100vh;
            width: 100vw;
        }}

        /* ===== HEADER ===== */
        .header {{
            grid-area: header;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
            border-bottom: 2px solid #00d2ff;
            padding: 20px 36px 18px 36px;
            text-align: center;
            z-index: 10;
        }}
        .header h1 {{
            font-size: 26pt; font-weight: 800; letter-spacing: -0.5px;
            color: #ffffff; margin-bottom: 4px;
        }}
        .header h1 .accent {{ color: #00d2ff; }}
        .header p {{ font-size: 11pt; font-weight: 400; color: #8b949e; }}

        /* ===== SIDEBAR ===== */
        .sidebar {{
            grid-area: sidebar;
            background: #0d1117;
            border-right: 1px solid #30363d;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        .sidebar::-webkit-scrollbar {{ width: 6px; }}
        .sidebar::-webkit-scrollbar-track {{ background: #0d1117; }}
        .sidebar::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
        .sidebar::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
        .sidebar .bk-input-group {{ margin-bottom: 1px !important; }}
        .sidebar .bk-input-group label {{ margin-bottom: 0 !important; }}
        .sidebar .bk-panel-model {{ gap: 2px !important; }}
        .sidebar .bk-root > div, .sidebar .bk-root > bk-column, .sidebar .bk-root > bk-row {{ gap: 2px !important; }}
        .bk-root .choices__item--choice {{ color: #000000 !important; }}
        .bk-root .choices__item--choice.is-highlighted {{ background: #0366d6 !important; color: #ffffff !important; }}
        .sidebar .choices__inner {{ position: relative; padding-right: 28px !important; }}
        .sidebar .choices__inner::after {{
            content: '▼'; position: absolute; right: 12px; top: 50%;
            transform: translateY(-50%); color: #8b949e; font-size: 9px; pointer-events: none;
        }}
        .bk-root select option {{ color: #000000 !important; background: #ffffff !important; }}

        /* ===== MAIN CONTENT ===== */
        .main-content {{
            grid-area: main;
            background: #0d1117;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px 14px 8px 14px;
        }}
        .main-content::-webkit-scrollbar {{ width: 8px; }}
        .main-content::-webkit-scrollbar-track {{ background: #0d1117; }}
        .main-content::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 4px; }}
        .main-content::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}

        /* ===== CHART WRAPPER ===== */
        .chart-wrapper {{
            margin-bottom: 0;
            max-width: 100%;
            overflow-x: auto;
        }}

        /* ===== CHART GROUP CONTAINER ===== */
        .chart-group-container {{
            max-width: {MAIN_MAX_WIDTH}px;
            width: 100%;
            margin: 0 auto;
        }}

        /* ===== NARRATIVE BLOCKS ===== */
        .narrative-block {{
            display: flex;
            gap: 18px;
            align-items: flex-start;
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #21262d;
            border-top: none;
            border-radius: 0 0 10px 10px;
            padding: 18px 24px 18px 20px;
            margin-bottom: 28px;
        }}
        .narrative-number {{
            font-size: 28pt;
            font-weight: 800;
            color: #ffffff;
            line-height: 1;
            min-width: 50px;
            letter-spacing: -2px;
            font-family: 'Inter', sans-serif;
            user-select: none;
            text-align: center;
            opacity: 0.15;
            padding-top: 4px;
        }}
        .narrative-content {{ flex: 1; }}
        .narrative-title {{
            font-size: 13pt;
            font-weight: 800;
            color: #e6edf3;
            margin-bottom: 6px;
            letter-spacing: -0.2px;
            font-family: 'Inter', sans-serif;
        }}
        .narrative-body {{
            font-size: 9.5pt;
            color: #8b949e;
            line-height: 1.65;
            margin-bottom: 12px;
            font-family: 'Inter', sans-serif;
        }}
        .narrative-finding {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            background: rgba(0, 210, 255, 0.06);
            border: 1px solid rgba(0, 210, 255, 0.2);
            border-radius: 6px;
            padding: 9px 14px;
        }}
        .finding-label {{
            font-size: 8.5pt;
            font-weight: 700;
            color: #00d2ff;
            white-space: nowrap;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding-top: 1px;
            font-family: 'Inter', sans-serif;
        }}
        .finding-text {{
            font-size: 10pt;
            color: #c9d1d9;
            line-height: 1.5;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
        }}

        /* ===== FOOTER ===== */
        .footer {{
            grid-area: footer;
            background: #0d1117;
            border-top: 1px solid #30363d;
            padding: 12px 36px;
            text-align: center;
            z-index: 10;
        }}
        .footer p {{ font-size: 10pt; font-weight: 500; color: #8b949e; margin: 0 0 2px 0; }}
        .footer .footer-sub {{ font-size: 9pt; color: #484f58; margin: 0; }}

        /* ===== RESPONSIVE ===== */
        @media (max-width: 900px) {{
            .dashboard-container {{
                grid-template-columns: 1fr;
                grid-template-rows: auto auto 1fr auto;
                grid-template-areas: "header" "sidebar" "main" "footer";
            }}
            .sidebar {{ max-height: 250px; border-right: none; border-bottom: 1px solid #30363d; }}
            .header h1 {{ font-size: 18pt; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">

        <!-- HEADER -->
        <header class="header">
            <h1><span class="accent">The Hidden Cost</span> of Intelligence</h1>
            <p>Exploring the Environmental Impact of AI&rsquo;s Exponential Growth &mdash; Data Centers, Carbon Footprint &amp; Compute Scale Analysis</p>
        </header>

        <!-- SIDEBAR -->
        <aside class="sidebar">
            {divs['sidebar']}
        </aside>

        <!-- MAIN CONTENT -->
        <main class="main-content">

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['chart0']}</div>
                {narrative_0}
            </div>


            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['p1']}</div>
                {narrative_1}
            </div>

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['chart2']}</div>
                {narrative_2}
            </div>

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['p3']}</div>
                {narrative_3}
            </div>

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['p4']}</div>
                {narrative_4}
            </div>

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['chart5']}</div>
                {narrative_5}
            </div>

            <div class="chart-group-container">
                <div class="chart-wrapper">{divs['chart6']}</div>
                {narrative_6}
            </div>

        </main>

        <!-- FOOTER -->
        <footer class="footer">
            <p>{footer_sources}</p>
            <p class="footer-sub">{footer_project}</p>
        </footer>

    </div>

    {script}
</body>
</html>
"""

with open('hidden_cost_of_ai_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Dashboard generated: hidden_cost_of_ai_dashboard.html")
webbrowser.open('file://' + os.path.realpath('hidden_cost_of_ai_dashboard.html'))
