import pandas as pd
import numpy as np
import json
import math
from datetime import datetime
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import (
    ColumnDataSource, CustomJS, HoverTool, Select, RangeSlider, 
    MultiChoice, CheckboxGroup, Div, WMTSTileSource, LinearColorMapper,
    ColorBar, BasicTicker, Span
)
from bokeh.layouts import column, row, layout
from bokeh.palettes import Magma256, Category10, Spectral10
from bokeh.transform import factor_cmap, linear_cmap

# ==========================================
# 1. DATA SETUP
# ==========================================

df_models = pd.read_csv('all_ai_models - Clean.csv')
df_models['pub_date'] = pd.to_datetime(df_models['Publication date'], errors='coerce')
df_models['params'] = pd.to_numeric(df_models['Parameters'], errors='coerce')
df_models['flops'] = pd.to_numeric(df_models['Training compute (FLOP)'], errors='coerce')

df_models = df_models.dropna(subset=['pub_date'])
df_models['org'] = df_models['Organization']
df_models['country'] = df_models['Country (of organization)'].fillna('Unknown')
df_models['accessibility_simple'] = df_models['Open model weights?'].apply(lambda x: 'Open Weight' if str(x).strip().lower() == 'yes' else 'Closed')
df_models['accessibility_detail'] = df_models['Model accessibility'].fillna('Unknown')

df_models['log_params'] = np.where(df_models['params'] > 0, np.log10(df_models['params']), 0)
df_models['log_flops'] = np.where(df_models['flops'] > 0, np.log10(df_models['flops']), 0)
df_models['negative_log_flops'] = -df_models['log_flops']

df_models_clean = df_models.fillna({'params': 0, 'flops': 0, 'log_params': 0, 'log_flops': 0, 'negative_log_flops': 0})
df_models_clean['pub_year'] = df_models_clean['pub_date'].dt.year
df_models_clean['pub_date_ms'] = df_models_clean['pub_date'].astype('int64') // 10**6

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
                'megawatts': item.get('megawatts', 0),
                'company': ", ".join(item.get('companies', [])),
                'country': item.get('country', ''),
                'lat': lat, 'lon': lon, 'x': x, 'y': y
            })
df_datacenters = pd.DataFrame(dc_list)

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

models_by_company = df_models_clean.groupby('org').size().reset_index(name='model_count')
models_by_country = df_models_clean.groupby('country').size().reset_index(name='model_count')

dc_by_company = df_datacenters.groupby('company').agg({'megawatts': 'sum', 'carbon_intensity': 'mean'}).reset_index()
dc_by_country = df_datacenters.groupby('country').agg({'megawatts': 'sum', 'carbon_intensity': 'mean'}).reset_index()

matrix_company = pd.merge(dc_by_company, models_by_company, left_on='company', right_on='org', how='inner')
matrix_company['entity'] = matrix_company['company']
matrix_company['type'] = 'By Company'

matrix_country = pd.merge(dc_by_country, models_by_country, on='country', how='inner')
matrix_country['entity'] = matrix_country['country']
matrix_country['type'] = 'By Country'

df_matrix = pd.concat([matrix_company[['entity', 'megawatts', 'carbon_intensity', 'model_count', 'type']], 
                       matrix_country[['entity', 'megawatts', 'carbon_intensity', 'model_count', 'type']]])
df_matrix.rename(columns={'megawatts': 'capacity_mw'}, inplace=True)

source_models_master = ColumnDataSource(df_models_clean)
source_models_view = ColumnDataSource(df_models_clean)

df_map_clean = df_datacenters[(df_datacenters['x'] != 0) & (df_datacenters['y'] != 0)].fillna(0)
source_map_master = ColumnDataSource(df_map_clean)
source_map_view = ColumnDataSource(df_map_clean)

source_matrix_master = ColumnDataSource(df_matrix.fillna(0))
source_matrix_view = ColumnDataSource(df_matrix[df_matrix['type'] == 'By Company'].fillna(0))

top_orgs = df_models_clean.groupby('org').size().reset_index(name='count').sort_values('count', ascending=False).head(15)
source_chart1 = ColumnDataSource(top_orgs)

# ==========================================
# 2. CHARTS
# ==========================================

# CHART 1
orgs_list = source_chart1.data['org'].tolist()[::-1]
p1 = figure(y_range=orgs_list, title="Top 15 Organizations by Model Count", 
            height=350, width=500, toolbar_location=None)
p1.hbar(y='org', right='count', source=source_chart1, height=0.8, color="teal")
p1.add_tools(HoverTool(tooltips=[("Org", "@org"), ("Count", "@count")]))

# CHART 2
p2 = figure(x_axis_type="datetime", title="Parameter Scale vs. Compute Cost (Log10)",
            height=350, width=700)
p2.vbar(x='pub_date', top='log_params', bottom=0, width=1000*60*60*24*7,
        source=source_models_view, color="blue", legend_label="Parameters (Log10)", alpha=0.6)
p2.vbar(x='pub_date', top=0, bottom='negative_log_flops', width=1000*60*60*24*7,
        source=source_models_view, color="red", legend_label="FLOPs (Log10)", alpha=0.6)
zero_line = Span(location=0, dimension='width', line_color='black', line_width=1)
p2.add_layout(zero_line)
p2.add_tools(HoverTool(tooltips=[
    ("Model", "@Model"), ("Org", "@org"), ("Date", "@pub_date{%F}"), 
    ("Params", "@params"), ("FLOPs", "@flops")
], formatters={'@pub_date': 'datetime'}))
p2.legend.location = "top_left"

# CHART 3
tile_options = WMTSTileSource(url="https://a.basemaps.cartocdn.com/light_all/{Z}/{X}/{Y}.png")
p3 = figure(x_axis_type="mercator", y_axis_type="mercator", 
            title="Global Data Center Footprint", height=400, width=600,
            x_range=(-15000000, 15000000), y_range=(-5000000, 8000000))
p3.add_tile(tile_options)

color_mapper = LinearColorMapper(palette=Magma256[::-1], low=100, high=600)
p3.scatter(x='x', y='y', size=10, color={'field': 'carbon_intensity', 'transform': color_mapper}, 
           source=source_map_view, alpha=0.7)
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title="Grid CO2 (gCO2/kWh)")
p3.add_layout(color_bar, 'right')
p3.add_tools(HoverTool(tooltips=[
    ("Facility", "@title"), ("Company", "@company"), 
    ("Capacity", "@megawatts MW"), ("Grid Carbon", "@carbon_intensity gCO2/kWh"),
    ("Estimated Total CO2", "@co2{0,0} tons/yr")
]))

# CHART 4
p4 = figure(title="Compute Capacity vs. Grid Carbon Intensity", 
            x_axis_label="Total Capacity (MW)", y_axis_label="Avg Carbon Intensity (gCO2/kWh)",
            height=400, width=600)
p4.scatter(x='capacity_mw', y='carbon_intensity', size=15, 
           source=source_matrix_view, alpha=0.6, color="green")
p4.add_tools(HoverTool(tooltips=[
    ("Entity", "@entity"), ("MW", "@capacity_mw"), ("Models", "@model_count")
]))

# ==========================================
# 3. SIDEBAR AND LAYOUT
# ==========================================

sidebar_title = Div(text="<h2>Global Filters</h2>", width=250)
access_filter = MultiChoice(title="Filter Accessibility:", options=["Open Weight", "Closed"], value=[])

min_y = df_models_clean['pub_year'].min() if not df_models_clean.empty else 2018
max_y = df_models_clean['pub_year'].max() if not df_models_clean.empty else 2026
date_slider = RangeSlider(start=min_y, end=max_y, value=(min_y, max_y), step=1, title="Publish Year")

max_p = df_models_clean['params'].max() / 1e9 if not df_models_clean.empty else 1000
param_slider = RangeSlider(start=0, end=max_p, value=(0, max_p), step=1, title="Parameters (Billions)")

all_countries = sorted(list(set(df_models_clean['country'].dropna().tolist())))
country_select = MultiChoice(title="Select Countries:", options=all_countries, value=[])

matrix_toggle = Select(title="Matrix Aggregation (Chart 4):", options=["By Company", "By Country"], value="By Company")

global_filter_callback = CustomJS(args=dict(
    s_models_master=source_models_master, s_models_view=source_models_view,
    s_map_master=source_map_master, s_map_view=source_map_view,
    s_chart1=source_chart1,
    date_sl=date_slider, param_sl=param_slider, countries=country_select,
    access=access_filter, p1_y_range=p1.y_range
), code="""
    const min_year = date_sl.value[0];
    const max_year = date_sl.value[1];
    const min_param = param_sl.value[0] * 1e9;
    const max_param = param_sl.value[1] * 1e9;
    const selected_countries = countries.value; 
    const selected_access = access.value;

    const m_data = s_models_master.data;
    const v_data = {};
    for (const key in m_data) { v_data[key] = []; }

    const org_counts = {};

    for (let i = 0; i < m_data['pub_date_ms'].length; i++) {
        let year = m_data['pub_year'][i];
        let param = m_data['params'][i];
        let ctry = m_data['country'][i];
        let acc = m_data['accessibility_simple'][i];
        let org = m_data['org'][i];

        let pass_date = (year >= min_year && year <= max_year);
        let pass_param = (param >= min_param && param <= max_param);
        let pass_ctry = (selected_countries.length === 0 || selected_countries.includes(ctry));
        let pass_acc = (selected_access.length === 0 || selected_access.includes(acc));

        if (pass_date && pass_param && pass_ctry && pass_acc) {
            for (const key in m_data) { v_data[key].push(m_data[key][i]); }
            org_counts[org] = (org_counts[org] || 0) + 1;
        }
    }
    s_models_view.data = v_data;
    s_models_view.change.emit();

    let org_arr = [];
    for (const org in org_counts) {
        org_arr.push({org: org, count: org_counts[org]});
    }
    org_arr.sort((a, b) => b.count - a.count);
    org_arr = org_arr.slice(0, 15);
    
    const c1_data = {org: [], count: []};
    const new_y_range = [];
    for (let i = org_arr.length - 1; i >= 0; i--) { 
        c1_data.org.push(org_arr[i].org);
        c1_data.count.push(org_arr[i].count);
        new_y_range.push(org_arr[i].org);
    }
    s_chart1.data = c1_data;
    s_chart1.change.emit();
    p1_y_range.factors = new_y_range;

    const map_m = s_map_master.data;
    const map_v = {};
    for (const key in map_m) { map_v[key] = []; }

    for (let i = 0; i < map_m['country'].length; i++) {
        let ctry = map_m['country'][i];
        if (selected_countries.length === 0 || selected_countries.includes(ctry)) {
            for (const key in map_m) { map_v[key].push(map_m[key][i]); }
        }
    }
    s_map_view.data = map_v;
    s_map_view.change.emit();
""")

date_slider.js_on_change('value', global_filter_callback)
param_slider.js_on_change('value', global_filter_callback)
country_select.js_on_change('value', global_filter_callback)
access_filter.js_on_change('value', global_filter_callback)

matrix_toggle_callback = CustomJS(args=dict(
    s_matrix_master=source_matrix_master, s_matrix_view=source_matrix_view,
    toggle=matrix_toggle
), code="""
    const t_val = toggle.value;
    const m_data = s_matrix_master.data;
    const v_data = {};
    for (const key in m_data) { v_data[key] = []; }

    for (let i = 0; i < m_data['type'].length; i++) {
        if (m_data['type'][i] === t_val) {
            for (const key in m_data) { v_data[key].push(m_data[key][i]); }
        }
    }
    s_matrix_view.data = v_data;
    s_matrix_view.change.emit();
""")
matrix_toggle.js_on_change('value', matrix_toggle_callback)

sidebar = column(
    sidebar_title, 
    access_filter, 
    date_slider, param_slider, 
    country_select, matrix_toggle, 
    width=300, 
    styles={'position': 'sticky', 'top': '0px'}
)

dashboard_layout = row(
    sidebar,
    column(
        row(p1, p2),
        row(p3, p4)
    )
)

output_file("hidden_cost_of_ai_dashboard.html", title="Advanced Data Vis Final")
show(dashboard_layout)
