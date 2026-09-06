# The Hidden Cost of Intelligence — AI Environmental Impact Dashboard

Live Demo: [https://ketanbasi.github.io/adv_vis_dat_final_project/](https://ketanbasi.github.io/adv_vis_dat_final_project/)

---

## Project Overview
This interactive dashboard explores the environmental impact of Artificial Intelligence's exponential growth. It highlights the tension between the development of large AI models ("the brains") and the massive physical infrastructure, energy requirements, and carbon emissions ("the cost") needed to sustain them.

---

## Key Visualizations

### Chart 00: The AI Model Cosmos
* **Type:** Interactive 2D Scatter Plot (Date vs. Parameter Count).
* **Details:** Maps individual AI models color-coded by weights accessibility (green for open weights, red for closed).
* **Interactivity:** Features a free-form **Lasso Selection** and **Box Selection** tool. Selecting a group of models dynamically filters all subsequent charts to analyze only the chosen subset.

### Chart 01: Top Builders of AI
* **Type:** Horizontal Bar Chart.
* **Details:** Visualizes the top 15 organizations by model count.
* **Interactivity:** Tooltips reveal the total models and details of the latest model published by each organization.

### Chart 02: The Exponential Cost (Monthly Timeline)
* **Type:** Diverging Monthly Bar Chart.
* **Details:** Compares model size (positive Y-axis, parameter counts) against training compute demand (negative Y-axis, FLOPs) on a Log₁₀ scale.
* **Interactivity:** Toggle aggregation methods dynamically between **Sum**, **Average**, and **Maximum** in the client browser.

### Chart 03: Physical Footprint Geospatial Map
* **Type:** Interactive Mercator Map (`WMTSTileSource`).
* **Details:** Plots data center clusters globally.
  * **Bubble Size:** Megawatts capacity.
  * **Bubble Color:** Local grid carbon intensity (gCO₂/kWh) from green (cleaner) to red (dirtier).
* **Interactivity:** Map theme toggle (Dark Matter, Light Matter, OSM, Satellite Earth). Dynamically filtered by the global year/parameter controls and active selections.

### Chart 04: Compute Capacity vs. Grid Carbon Intensity Matrix
* **Type:** 2D Scatter Plot (Log MW Capacity vs. Grid Intensity).
* **Details:** Evaluates organizations' and regions' alignment with sustainable standards. Features reference lines for average capacity and the **EU Taxonomy low-carbon threshold** (100 gCO₂/kWh).
* **Interactivity:** Switches aggregation levels on the fly:
  * **By Country:** National compute footprints.
  * **By Company:** Specific corporate compute footprint.
  * **All Datacenters:** Exploded view of all 344 individual datacenters (labels automatically hide in this view to maintain visual clarity).

### Chart 05: Estimated Annual CO₂ Emissions
* **Type:** Horizontal Bar Chart.
* **Details:** Shows the estimated annual CO₂ output of data centers for the top 15 countries.
* **Interactivity:** Sorts dynamically by **CO₂ Emissions (tons/year)**, **DC Capacity (MW)**, or **Carbon Intensity (gCO₂/kWh)**.

### Chart 06: Approaching the Energy Wall
* **Type:** Multi-Axis Trend Plot.
* **Details:** Tracks the cumulative models published (left Y-axis) against maximum training compute (right Y-axis) per year.
* **Interactivity:** Interactive checkboxes show/hide individual trendlines.

---

## Setup & Execution

### Prerequisites
Ensure you have Python 3.10+ and a virtual environment set up. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Files Included
* [dashboard.py](dashboard.py): Main script containing data preprocessing and Bokeh visualization layouts.
* [scrape_datacenter.ipynb](scrape_datacenter.ipynb): Notebook used to collect data from AI Data Center Index and Electricity Maps (API key required), and generate the json files.
* `all_ai_models - Clean.csv`: Structured model metadata from Epoch AI, manually cleaned and preprocessed.
* `all_datacenter_per_country.json`: Datacenter coordinates, company, and megawatt metrics.
* `all_electricity_carbon_intensity_kwh.json`: Dynamic electricity grid carbon mappings.

### Compile the Dashboard
Run the following command to process the datasets and output the final single-page HTML application:
```bash
python dashboard.py
```
This generates [hidden_cost_of_ai_dashboard.html](hidden_cost_of_ai_dashboard.html) and automatically opens it in your default web browser.

---

## Data Sources
1. **Model Timeline & Compute:** [Epoch AI database](https://epoch.ai/)
2. **Global Grid Emissions:** [Electricity Maps](https://electricitymaps.com/)
3. **Data Centers:** [AI Data Center Index](https://aidatacenterindex.com/)
