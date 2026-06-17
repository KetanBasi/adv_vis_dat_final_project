# Project Design Plan: The Hidden Cost of Intelligence
**Class:** Advanced Data Visualization
**Tech Stack:** Python, Pandas (Preprocessing), Bokeh (Visualization), CustomJS (Interactivity), HTML (Deployment)
**Deployment:** Static Site Hosting (GitHub Pages / Vercel)

---

## 1. Narrative & Objective
The dashboard tells the story of AI's environmental impact, creating a narrative tension between the exponential growth of AI models ("the brains") and the massive physical infrastructure and carbon footprint required to sustain them.

## 2. Datasets & Feature Engineering
1. **Epoch AI (`all_ai_models.csv`):** Tracks model evolution (Org, Publication Date, Parameters, FLOPs, Accessibility, Country).
2. **Electricity Maps:** Tracks grid carbon intensity (`gCO2/kWh`) globally.
3. **AI Data Center Index:** Tracks current physical footprint (Title, MW capacity, Company, Lat/Lng).
4. **Engineered Feature (Calculated Total CO₂):** `Total CO₂ (tons/year) = Power (kW) * Hours per year * Grid Carbon Intensity (gCO₂/kWh) / 1,000,000`

---

## 3. Dashboard Architecture
Because the dashboard will be hosted statically, there is no Python backend (Bokeh Server). 
* All heavy data aggregations (e.g., grouping by Country vs. Company) will be preprocessed in Pandas and saved as distinct `ColumnDataSource` objects.
* All interactivity, cross-filtering, and dynamic updates will be driven entirely client-side using **Bokeh's `CustomJS` callbacks**.

---

## 4. Visualizations (The Charts)

### Chart 1: Who is Building the Brains? (Basic Plot)
* **Type:** Horizontal Stacked Bar Chart.
* **Data:** Top Organizations by Model Count.
* **Focus:** Model Accessibility.
* **Interactivity:** A chart-specific toggle to switch views between **Simple** (Open Weight vs. Closed) and **Detailed** (Unrestricted, Restricted, API access, Hosted no API). 

### Chart 2: The Exponential Cost (Time-Series & Complex Plot)
* **Type:** Diverging Bar Chart.
* **Data:** X-axis = Publication Date (grouped by day/week/month). 
* **Focus:** Visualizing "Scale" vs. "Cost". 
    * Positive Y-axis = Parameter Count (Brain Size).
    * Negative Y-axis = Training Compute / FLOPs (Compute Cost).
* **Note:** Consider a logarithmic transformation during preprocessing to prevent recent models from dwarfing older models.

### Chart 3: The Physical Footprint (Geospatial Map)
* **Type:** Interactive Map (`WMTSTileSource`).
* **Data:** Data Centers plotted via Web Mercator Lat/Lng.
* **Focus:** Geographic distribution and environmental load.
    * **Bubble Size:** Megawatts (Capacity scale).
    * **Bubble Color:** Grid Carbon Intensity / Total CO₂ (Color mapped from green to red).
* **Interactivity:** Tooltips showing Facility, Capacity, and calculated Carbon. Affected by global Country filter (highlights selected, grays out others).

### Chart 4: The Carbon-Compute Matrix (Complex Plot)
* **Type:** 2D Scatter/Bubble Plot.
* **Data:** X = Total Capacity (MW), Y = Avg Carbon Intensity, Bubble Size = AI Model Count.
* **Focus:** Identifying "Green AI" leaders (high compute, low carbon) vs. worst offenders.
* **Interactivity:** Chart-specific dropdown toggle to switch aggregation between **By Company** and **By Country** (swapping pre-calculated data sources via JS).

---

## 5. Global Sticky Sidebar (Client-Side Filters)
A layout column anchored to the side (`position: sticky`), driving `CustomJS` callbacks that filter a "Master" data source and update the "View" data sources for all charts:
1. **Model Accessibility Filter:** Multi-select/checkboxes for Open, Closed, and detailed access levels.
2. **Date Range Slider:** To filter models by publication year/date.
3. **Parameter Count Slider:** Number range input to filter by model size.
4. **Country Multi-select:** Shows models from specific countries, and specifically cross-links to **Chart 3** to visually highlight selected regions while graying out unselected ones.