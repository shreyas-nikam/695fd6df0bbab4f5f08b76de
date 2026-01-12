
# Streamlit Application Specification: PE Sector Configuration Manager

## 1. Application Overview

### Purpose
The "PE Sector Configuration Manager" Streamlit application serves as an administrative user interface for private equity (PE) firms to manage the critical configuration parameters across various investment sectors. Built for Software Developers and Data Engineers, this tool demonstrates a real-world workflow for maintaining a unified data architecture where sector-specific logic is driven by configuration, not schema variations. It allows users to view and propose updates to dimension weights and calibration parameters, and illustrates the role of caching in data delivery and consistency, particularly focusing on how cache invalidation ensures that client applications retrieve the most current configuration.

### Story Flow
The application guides the user through the following workflow:

1.  **Welcome & Overview**: Upon launching, the user is greeted with an introduction to the application and a high-level overview of the configuration-driven architecture concept. They can then navigate to view all defined PE sectors.
2.  **Sector Discovery**: The user can browse a table of all available PE sectors, understanding their basic properties. From this overview, they can select a specific sector to delve into its detailed configuration.
3.  **Configuration Management**: For a chosen sector, the user can inspect its current dimension weights (e.g., how important 'Data Infrastructure' is for 'Manufacturing' vs. 'Healthcare') and calibration parameters (e.g., 'H^R Baseline', 'EBITDA Multiplier'). An interactive form allows the user to *propose* new weights and calibrations. Client-side validation ensures dimension weights sum to 1.0. Upon "submitting" these proposed changes, the application simulates the interaction with a backend API (which would handle persistence) and demonstrates the critical step of *cache invalidation* to ensure that subsequent data requests fetch the (simulated) fresh configuration.
4.  **Organization Data (Conceptual)**: A dedicated section illustrates how organizations are linked to sectors and possess sector-specific attributes. Due to the limited scope of available backend functions in `source.py` for direct organization queries, this section will primarily serve to conceptualize this integration, explaining how a full backend would expose such data. This highlights architectural boundaries and the need for comprehensive API endpoints.

This flow emphasizes the `Apply` Bloom's level objective from the lab preamble, showing how to implement focus group configuration loading with caching in an interactive administrative context.

## 2. Code Requirements

### Import Statement
The application will begin with the following imports:

```python
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
from decimal import Decimal

# Import all available components from the provided source.py file
from source import SectorConfig, SectorConfigService, sector_service, cache
```

### `st.session_state` Design

#### Initialization
`st.session_state` will be initialized at the start of `app.py` to ensure state persistence across user interactions and page changes.

```python
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'
if 'all_sectors_data' not in st.session_state:
    st.session_state.all_sectors_data = [] # List[SectorConfig]
if 'selected_sector_id' not in st.session_state:
    st.session_state.selected_sector_id = None
if 'selected_sector_config' not in st.session_state:
    st.session_state.selected_sector_config = None # SectorConfig object
if 'editing_weights' not in st.session_state:
    st.session_state.editing_weights = {} # Temporary store for proposed weights
if 'editing_calibrations' not in st.session_state:
    st.session_state.editing_calibrations = {} # Temporary store for proposed calibrations
if 'weights_rationale' not in st.session_state:
    st.session_state.weights_rationale = ""
if 'calibrations_rationale' not in st.session_state:
    st.session_state.calibrations_rationale = ""
if 'show_weight_validation_error' not in st.session_state:
    st.session_state.show_weight_validation_error = False
```

#### Update & Read Patterns
*   `st.session_state.current_page`: Updated via sidebar dropdown. Read to conditionally render page content.
*   `st.session_state.all_sectors_data`: Updated once on app startup (or first access to Overview page) by calling `asyncio.run(sector_service.get_all_configs())`. Read to display the list of sectors and populate sector selection widgets.
*   `st.session_state.selected_sector_id`: Updated when a user selects a sector from a list or dropdown. Read to fetch and display details for that specific sector.
*   `st.session_state.selected_sector_config`: Updated after `selected_sector_id` changes, by calling `asyncio.run(sector_service.get_config(st.session_state.selected_sector_id))`. This will leverage the caching mechanism. Read to populate display tables and editing forms for the selected sector's configuration. This will be re-fetched after a simulated update and cache invalidation.
*   `st.session_state.editing_weights` and `st.session_state.editing_calibrations`: Updated by Streamlit input widgets (e.g., `st.number_input`) as the user proposes changes. Read during the "Propose Changes" action to gather data for validation and the simulated backend call.
*   `st.session_state.weights_rationale` and `st.session_state.calibrations_rationale`: Updated by `st.text_area` for audit trail.
*   `st.session_state.show_weight_validation_error`: Boolean flag, set to `True` if `validate_weights_sum` fails, cleared otherwise.

### Application Pages

#### 1. Sidebar Navigation
The sidebar will provide navigation for a multi-page experience.

```python
with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.svg", width=50) # Placeholder image
    st.markdown(f"# PE Config Admin")
    page_selection = st.radio(
        "Navigate",
        ['Overview', 'Manage Sector Configuration', 'Organization Browser (Conceptual)']
    )
    if page_selection != st.session_state.current_page:
        st.session_state.current_page = page_selection
        # Reset specific session state for new page context if necessary
        if page_selection == 'Manage Sector Configuration':
            if st.session_state.all_sectors_data:
                # Set default selected sector if not already set
                if not st.session_state.selected_sector_id:
                    st.session_state.selected_sector_id = st.session_state.all_sectors_data[0].focus_group_id
                    st.session_state.selected_sector_config = asyncio.run(sector_service.get_config(st.session_state.selected_sector_id))
                # Populate editing fields with current values for default sector
                if st.session_state.selected_sector_config:
                    st.session_state.editing_weights = {k: float(v) for k, v in st.session_state.selected_sector_config.dimension_weights.items()}
                    st.session_state.editing_calibrations = {k: float(v) for k, v in st.session_state.selected_sector_config.calibrations.items()}
        st.experimental_rerun() # Rerun to switch page content immediately
```

#### 2. Page: Sector Overview
This page will provide a high-level view of all PE sectors.

**Layout**:
*   Main column for title and introductory markdown.
*   Table displaying all sectors.

**Markdown Content**:
```python
st.markdown(f"# Sector Overview & Discovery")
st.markdown(f"Welcome to the **PE Sector Configuration Manager**.")
st.markdown(f"This application serves as an administrative interface for managing the critical configuration parameters that define our investment strategy across various private equity sectors.")
st.markdown(f"Our core principle is **'One Schema, Many Configurations'**. Instead of creating separate database schemas for each sector, we differentiate sector-specific logic through flexible configuration tables.")
st.markdown(f"This approach avoids schema proliferation, simplifies database management, and enhances agility in adapting to new investment thesis or market conditions.")
st.markdown(f"Below is a list of all currently configured PE sectors:")

# Function Invocation:
# Fetch all sector configurations
all_sectors_configs = asyncio.run(sector_service.get_all_configs())
if all_sectors_configs:
    st.session_state.all_sectors_data = all_sectors_configs
    # Convert to DataFrame for display
    sectors_df = pd.DataFrame([
        {'ID': s.focus_group_id, 'Name': s.group_name, 'Code': s.group_code,
         'H^R Baseline': s.h_r_baseline, 'EBITDA Multiplier': s.ebitda_multiplier,
         'Talent Threshold': s.talent_concentration_threshold}
        for s in st.session_state.all_sectors_data
    ])
    st.dataframe(sectors_df, use_container_width=True)
else:
    st.warning("No sector configurations found. Please ensure the backend is populated.")

st.markdown(f"Select a sector from the sidebar or the 'Manage Sector Configuration' page to view and modify its detailed parameters.")
```

#### 3. Page: Manage Sector Configuration
This page allows detailed viewing and *simulated* editing of a selected sector's dimension weights and calibrations.

**Layout**:
*   Sidebar for sector selection.
*   Main column divided into:
    *   Sector details section.
    *   Dimension Weights section (table, edit form, validation, visualization).
    *   Calibrations section (table, edit form).
    *   Simulated Update & Cache Invalidation section.

**Markdown Content & Widgets**:
```python
st.markdown(f"# Manage Sector Configuration")
st.markdown(f"Adjust dimension weights and calibration parameters for the selected PE sector. These configurations drive how we evaluate companies within each sector, ensuring our models are tailored and accurate.")

# Widget: Sector Selector
sector_ids = [s.focus_group_id for s in st.session_state.all_sectors_data]
if not sector_ids:
    st.warning("No sectors available to configure. Please check the backend.")
else:
    selected_sector_id_from_widget = st.selectbox(
        "Select a PE Sector to configure:",
        sector_ids,
        index=sector_ids.index(st.session_state.selected_sector_id) if st.session_state.selected_sector_id in sector_ids else 0,
        key='sector_config_selector'
    )

    if selected_sector_id_from_widget != st.session_state.selected_sector_id:
        st.session_state.selected_sector_id = selected_sector_id_from_widget
        st.session_state.selected_sector_config = asyncio.run(sector_service.get_config(st.session_state.selected_sector_id))
        # Initialize editing fields with current values when sector changes
        if st.session_state.selected_sector_config:
            st.session_state.editing_weights = {k: float(v) for k, v in st.session_state.selected_sector_config.dimension_weights.items()}
            st.session_state.editing_calibrations = {k: float(v) for k, v in st.session_state.selected_sector_config.calibrations.items()}
            st.session_state.weights_rationale = ""
            st.session_state.calibrations_rationale = ""
            st.session_state.show_weight_validation_error = False
        st.experimental_rerun() # Rerun to update page content with new sector

    if st.session_state.selected_sector_config:
        current_sector = st.session_state.selected_sector_config
        st.markdown(f"---")
        st.markdown(f"## Current Configuration for {current_sector.group_name} ({current_sector.group_code})")

        # Display current Dimension Weights
        st.markdown(f"### Dimension Weights")
        st.markdown(f"These weights represent the relative importance of each dimension when evaluating companies within the **{current_sector.group_name}** sector. They must sum to 1.0.")
        weights_df = pd.DataFrame(
            {'Dimension': list(current_sector.dimension_weights.keys()),
             'Weight': [float(w) for w in current_sector.dimension_weights.values()]}
        )
        st.dataframe(weights_df, use_container_width=True)

        # Weight Distribution Visualization
        st.markdown(f"Visualizing the current weight distribution:")
        fig = px.pie(weights_df, values='Weight', names='Dimension', title='Current Dimension Weight Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Display current Calibrations
        st.markdown(f"### Calibration Parameters")
        st.markdown(f"Calibration parameters are sector-specific numerical values used in various models and calculations. These fine-tune our analysis for **{current_sector.group_name}**.")
        calibrations_df = pd.DataFrame(
            {'Parameter': list(current_sector.calibrations.keys()),
             'Value': [float(v) for v in current_sector.calibrations.values()]}
        )
        st.dataframe(calibrations_df, use_container_width=True)

        st.markdown(f"---")
        st.markdown(f"## Propose Configuration Changes")
        st.markdown(f"Use the forms below to propose new dimension weights and calibration parameters. Note that dimension weights are constrained to sum to $$1.0$$.")
        st.markdown(r"$$ \sum_{i=1}^{N} w_i = 1.0 $$")
        st.markdown(r"where $w_i$ represents the weight of dimension $i$, and $N$ is the total number of dimensions.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### Edit Dimension Weights")
            for dim, current_weight in st.session_state.editing_weights.items():
                st.session_state.editing_weights[dim] = st.number_input(
                    f"Weight for {dim}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_weight,
                    step=0.01,
                    format="%.3f",
                    key=f"weight_edit_{dim}"
                )
            st.session_state.weights_rationale = st.text_area(
                "Rationale for weight changes:",
                value=st.session_state.weights_rationale,
                key="weights_rationale_text"
            )

        with col2:
            st.markdown(f"### Edit Calibrations")
            for param, current_value in st.session_state.editing_calibrations.items():
                st.session_state.editing_calibrations[param] = st.number_input(
                    f"Value for {param}:",
                    value=current_value,
                    step=0.01,
                    format="%.4f", # Calibrations might have more precision
                    key=f"calibration_edit_{param}"
                )
            st.session_state.calibrations_rationale = st.text_area(
                "Rationale for calibration changes:",
                value=st.session_state.calibrations_rationale,
                key="calibrations_rationale_text"
            )

        st.markdown("---")
        if st.button("Propose & Simulate Update"):
            # Client-side validation for weights sum
            proposed_weights_decimal = {k: Decimal(str(v)) for k, v in st.session_state.editing_weights.items()}
            temp_config_for_validation = SectorConfig(
                focus_group_id=current_sector.focus_group_id,
                group_name=current_sector.group_name,
                group_code=current_sector.group_code,
                dimension_weights=proposed_weights_decimal,
                calibrations=current_sector.calibrations # Calibrations are not validated this way
            )

            if not temp_config_for_validation.validate_weights_sum():
                total_sum = sum(proposed_weights_decimal.values())
                st.session_state.show_weight_validation_error = True
                st.error(f"Error: Dimension weights must sum to 1.0. Current sum: {total_sum:.3f}")
            else:
                st.session_state.show_weight_validation_error = False
                st.success(f"Proposed changes for **{current_sector.group_name}** are valid and ready for backend processing.")
                st.info(f"In a real-world scenario, these changes (new weights: {st.session_state.editing_weights}, new calibrations: {st.session_state.editing_calibrations}) "
                        f"with rationale (Weights: '{st.session_state.weights_rationale}', Calibrations: '{st.session_state.calibrations_rationale}') "
                        f"would be sent to a FastAPI backend API endpoint (e.g., `/api/v1/sector/{current_sector.focus_group_id}/update_config`).")

                # Function Invocation: Simulate cache invalidation
                st.markdown(f"As part of the backend processing, the cached configuration for **{current_sector.group_name}** would be invalidated to ensure all downstream services fetch the freshest data.")
                asyncio.run(sector_service.invalidate_cache(current_sector.focus_group_id))
                st.success(f"Cache invalidated for sector: `{current_sector.focus_group_id}`. A subsequent request will fetch data directly from the database.")

                # Re-fetch the configuration to demonstrate cache invalidation
                # (Since we didn't actually update the DB, it will show original data,
                # but the *process* of cache invalidation is demonstrated.)
                st.session_state.selected_sector_config = asyncio.run(sector_service.get_config(st.session_state.selected_sector_id))
                st.warning(f"Note: Since no direct database write occurred in this simulation, the displayed values will revert to their original state. However, the cache invalidation step has been successfully demonstrated.")
                st.experimental_rerun() # Rerun to refresh display
    else:
        st.info("Please select a sector to manage its configuration.")
```

#### 4. Page: Organization Browser (Conceptual)
This page highlights the gap in the provided `source.py` for fetching organization data while demonstrating the user's need.

**Layout**:
*   Main column for title and explanation.

**Markdown Content**:
```python
st.markdown(f"# Organization Browser (Conceptual)")
st.markdown(f"This section is designed to allow users (Software Developers and Data Engineers) to browse organizations, filter them by sector, and view their sector-specific attributes. This is crucial for verifying how configuration parameters impact actual entity data.")
st.markdown(f"The lab's unified data architecture includes an `organizations` table with a foreign key to `focus_groups`, and `7 sector-specific attribute tables` (e.g., `org_attributes_manufacturing`, `org_attributes_financial_services`). A `vw_organizations_full` view is also defined, joining organizations with their respective sector attributes.")

st.markdown(f"**Current Limitation:**")
st.markdown(f"While the database schema (as seen in `source.py` DDL) supports these rich organization views, the provided `source.py` module for this Streamlit application does not expose public functions to directly query `organizations` or `vw_organizations_full`.")
st.markdown(f"To fully implement this feature, the `source.py` file (or a related data access layer) would need to include functions such as:")
st.markdown(f"- `get_all_organizations() -> List[Dict]`")
st.markdown(f"- `get_organizations_by_sector(sector_id: str) -> List[Dict]`")
st.markdown(f"- `get_organization_details(org_id: str) -> Dict`")
st.markdown(f"These functions would leverage the `db.fetch_all()` method, similar to how `SectorConfigService` loads its data from the database.")

st.info(f"For this blueprint, we are unable to display live organization data due to the constraint of strictly using functions available in the provided `source.py`. This highlights an important aspect of API design: ensuring all required data access patterns are exposed through the backend service.")
```
