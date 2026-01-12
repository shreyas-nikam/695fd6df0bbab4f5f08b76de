
# Streamlit Application Specification: Unified Data Architecture & Caching Lab

## 1. Application Overview

This Streamlit application serves as an interactive lab environment for Software Developers and Data Engineers. It guides learners through building a robust, configuration-driven data architecture, focusing on database schema design, data seeding, and implementing a configuration service with caching using a Jupyter Notebook-like experience. The app simulates a real-world workflow, allowing users to apply theoretical concepts in a practical setting.

**High-level Story Flow:**

1.  **Welcome & Objectives:** The user starts on a home page detailing the lab's purpose and key learning objectives.
2.  **Schema Design:** The user learns about designing a flexible database schema for focus group configurations, organizations, and sector-specific attributes. They interact with buttons to simulate the creation of these schemas in a PostgreSQL environment.
3.  **Data Seeding:** The user then proceeds to seed the designed tables with initial configuration data, including dimension weights and sector-specific calibrations.
4.  **Configuration Service:** The core concept of a configuration service is introduced, demonstrating how to retrieve sector-specific settings dynamically. Users can select a sector and visualize its unique dimension weights and calibration parameters.
5.  **Caching Layer:** The application then explores enhancing the configuration service with a Redis caching layer to improve performance. Users can trigger cache invalidation and observe its effects (simulated).
6.  **Unified Organization View:** Finally, users interact with a unified view of organizations, where sector-specific attributes are dynamically joined based on the configuration-driven architecture. They can query and filter organizations based on their assigned sectors.

Throughout the lab, the application provides detailed markdown explanations, code snippets (as if from a Jupyter Notebook), and interactive widgets to drive the learning process.

## 2. Code Requirements

### Import Statement

The following import statement will be used at the beginning of `app.py`:

```python
from source.py import *
import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio # Required to run async functions from source.py
```

### `st.session_state` Design

`st.session_state` will be used to preserve the application's state across user interactions and page navigations.

**Initialization:**

At the start of `app.py`, before any Streamlit calls, the following session state variables will be initialized:

```python
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'initial_data_seeded' not in st.session_state:
    st.session_state.initial_data_seeded = False
if 'weights_seeded' not in st.session_state:
    st.session_state.weights_seeded = False
if 'calibrations_seeded' not in st.session_state:
    st.session_state.calibrations_seeded = False
if 'organizations_seeded' not in st.session_state:
    st.session_state.organizations_seeded = False
if 'all_focus_groups' not in st.session_state:
    st.session_state.all_focus_groups = []
if 'selected_sector_id' not in st.session_state:
    st.session_state.selected_sector_id = None
if 'sector_configs_cache' not in st.session_state:
    st.session_state.sector_configs_cache = {} # To store fetched SectorConfig objects by ID
if 'unified_org_filter_sector_id' not in st.session_state:
    st.session_state.unified_org_filter_sector_id = None
```

**Updates:**

*   `st.session_state.current_page`: Updated by the `st.sidebar.selectbox` widget.
*   `st.session_state.db_initialized`: Set to `True` after `setup_database_schema()` is successfully called.
*   `st.session_state.initial_data_seeded`: Set to `True` after `seed_initial_data()` is successfully called.
*   `st.session_state.weights_seeded`: Set to `True` after `seed_dimension_weights_for_all_sectors()` is successfully called.
*   `st.session_state.calibrations_seeded`: Set to `True` after `seed_calibrations_for_all_sectors()` is successfully called.
*   `st.session_state.organizations_seeded`: Set to `True` after `insert_sample_organizations()` is successfully called.
*   `st.session_state.all_focus_groups`: Populated once `seed_initial_data()` is complete, by calling `get_all_focus_groups()`. This will be an array of dicts `[{'focus_group_id': '...', 'group_name': '...'}]`.
*   `st.session_state.selected_sector_id`: Updated by the `st.selectbox` widget on the "Configuration Service" page.
*   `st.session_state.sector_configs_cache`: Updated when `get_sector_config_from_service_sync()` is called; stores the returned `SectorConfig` object.
*   `st.session_state.unified_org_filter_sector_id`: Updated by the `st.selectbox` widget on the "Unified Organization View" page.

**Reads:**

All session state variables are read to conditionally render UI elements, determine workflow progress, and pass arguments to functions from `source.py`.

### UI Interactions and `source.py` Function Calls

The application will use a sidebar for navigation. Each "page" will be rendered conditionally based on `st.session_state.current_page`.

---

#### Sidebar Navigation

```python
with st.sidebar:
    st.header("Lab Navigation")
    page_options = [
        "Home",
        "2.1-2.5: Schema Design & Attributes",
        "2.2-2.3: Data Seeding",
        "2.6: Sector Configuration Service",
        "2.7: Redis Caching Layer",
        "2.8: Unified Organization View"
    ]
    st.session_state.current_page = st.selectbox(
        "Go to section:",
        page_options,
        index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    )
```

---

#### Page: Home

**Purpose**: Introduce the lab, its objectives, and the tools involved.

**Markdown Content**:

```python
st.title("Week 2: Unified Data Architecture & Caching Lab")

st.markdown(f"")
st.markdown(f"Welcome to Week 2 of our journey into building a robust Private Equity (PE) Intelligence Platform!")
st.markdown(f"")
st.markdown(f"In this lab, you'll transition from foundational setup to designing a truly configuration-driven data architecture.")
st.markdown(f"This approach is crucial for managing the complexity of diverse PE sectors without resorting to schema proliferation.")
st.markdown(f"")
st.subheader("Key Objectives")
st.markdown(f"")
st.markdown(f"- **Remember**: List the 7 PE sectors and their configuration parameters.")
st.markdown(f"- **Understand**: Explain why configuration-driven architecture avoids schema proliferation.")
st.markdown(f"- **Apply**: Implement focus group configuration loading with caching.")
st.markdown(f"- **Analyze**: Compare sector attribute tables vs JSONB approaches.")
st.markdown(f"- **Evaluate**: Assess dimension weight configurations for different sectors.")
st.markdown(f"")
st.subheader("Tools Introduced")
st.markdown(f"")
st.markdown(f"This week, we'll be working with powerful tools to achieve our architectural goals:")
st.markdown(f"")
st.markdown(f"- **PostgreSQL / Snowflake**: Our primary database, supporting both development and production environments.")
st.markdown(f"- **SQLAlchemy 2.0**: An ORM layer for advanced database interactions.")
st.markdown(f"- **Alembic**: For version-controlled schema migrations, ensuring smooth database evolution.")
st.markdown(f"- **Redis**: A fast in-memory data store for caching, essential for high-performance configuration lookups.")
st.markdown(f"- **structlog**: For structured logging, enhancing observability of our services.")
st.markdown(f"")
st.subheader("Key Concepts")
st.markdown(f"")
st.markdown(f"The central theme for this week is **One Schema, Many Configurations**. This means:")
st.markdown(f"")
st.markdown(f"- We avoid creating separate schemas for each PE sector.")
st.markdown(f"- Differentiation between sectors is achieved through data rows in configuration tables, not schema variations.")
st.markdown(f"- Focus Group Configuration Tables store weights and calibrations as data rows.")
st.markdown(f"- Queryable Sector Attribute Tables use typed columns instead of less flexible JSONB approaches.")
st.markdown(f"- Configuration Caching ensures that frequently accessed configurations are loaded once and used everywhere, reducing database load.")
```

**Function Calls**: None.

---

#### Page: 2.1-2.5: Schema Design & Attributes

**Purpose**: Guide the user through creating the database schema for configuration, organizations, and sector-specific attributes.

**Markdown Content & UI**:

```python
st.title("Task 2.1-2.5: Database Schema Design")
st.markdown(f"")
st.markdown(f"This section focuses on designing a flexible and extensible data architecture.")
st.markdown(f"We'll define tables that allow for configuration-driven differentiation across PE sectors, avoiding the 'schema per sector' anti-pattern.")
st.markdown(f"")

st.subheader("Design Principle: One Schema, Many Configurations")
st.markdown(f"")
st.markdown(f"A core principle of our architecture is that all 7 PE sectors share identical base schemas. Sector-specific differentiation is achieved through configuration tables and dedicated attribute tables, rather than varying the base schema.")
st.markdown(f"")
st.markdown(f"This approach minimizes `N×M` joins, prevents `NULL` proliferation in central tables, and allows for robust querying of sector-specific attributes using typed columns.")
st.markdown(f"")

st.subheader("Task 2.1: Design the Focus Group Configuration Schema")
st.markdown(f"")
st.markdown(f"We'll start by defining the `focus_groups` table to store our primary sectors, along with `dimensions` and `focus_group_dimension_weights`, and `focus_group_calibrations` to hold sector-specific configuration parameters.")
st.markdown(f"")
st.markdown(f"Here's the PostgreSQL DDL for these tables:")
st.markdown(f"```sql")
st.markdown(f"CREATE TABLE focus_groups (\n    focus_group_id VARCHAR(50) PRIMARY KEY,\n    platform VARCHAR(20) NOT NULL CHECK (platform IN ('pe_org_air', 'individual_air')),\n    group_name VARCHAR(100) NOT NULL,\n    group_code VARCHAR(30) NOT NULL,\n    group_description TEXT,\n    display_order INTEGER NOT NULL,\n    icon_name VARCHAR(50),\n    color_hex VARCHAR(7),\n    is_active BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (platform, group_code)\n);")
st.markdown(f"")
st.markdown(f"CREATE TABLE dimensions (\n    dimension_id VARCHAR(50) PRIMARY KEY,\n    platform VARCHAR(20) NOT NULL,\n    dimension_name VARCHAR(100) NOT NULL,\n    dimension_code VARCHAR(50) NOT NULL,\n    description TEXT,\n    min_score DECIMAL(5,2) DEFAULT 0,\n    max_score DECIMAL(5,2) DEFAULT 100,\n    display_order INTEGER NOT NULL,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (platform, dimension_code)\n);")
st.markdown(f"")
st.markdown(f"CREATE TABLE focus_group_dimension_weights (\n    weight_id SERIAL PRIMARY KEY,\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    dimension_id VARCHAR(50) NOT NULL REFERENCES dimensions(dimension_id),\n    weight DECIMAL(4,3) NOT NULL CHECK (weight >= 0 AND weight <= 1),\n    weight_rationale TEXT,\n    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,\n    effective_to DATE,\n    is_current BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (focus_group_id, dimension_id, effective_from)\n);")
st.markdown(f"")
st.markdown(f"CREATE TABLE focus_group_calibrations (\n    calibration_id SERIAL PRIMARY KEY,\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    parameter_name VARCHAR(100) NOT NULL,\n    parameter_value DECIMAL(10,4) NOT NULL,\n    parameter_type VARCHAR(20) DEFAULT 'numeric',\n    description TEXT,\n    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,\n    effective_to DATE,\n    is_current BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (focus_group_id, parameter_name, effective_from)\n);")
st.markdown(f"```")

st.subheader("Task 2.4: Design the Organizations Table with Sector Reference")
st.markdown(f"")
st.markdown(f"The `organizations` table will store core information about each company. Critically, it includes a `focus_group_id` as a foreign key to link each organization to its primary PE sector. This is the cornerstone of our configuration-driven approach.")
st.markdown(f"")
st.markdown(f"```sql")
st.markdown(f"CREATE TABLE organizations (\n    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n    legal_name VARCHAR(255) NOT NULL,\n    display_name VARCHAR(255),\n    ticker_symbol VARCHAR(10),\n    cik_number VARCHAR(20),\n    duns_number VARCHAR(20),\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    primary_sic_code VARCHAR(10),\n    primary_naics_code VARCHAR(10),\n    employee_count INTEGER,\n    annual_revenue_usd DECIMAL(15,2),\n    founding_year INTEGER,\n    headquarters_country VARCHAR(3),\n    headquarters_state VARCHAR(50),\n    headquarters_city VARCHAR(100),\n    website_url VARCHAR(500),\n    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    created_by VARCHAR(100),\n    CONSTRAINT chk_org_pe_platform CHECK (focus_group_id LIKE 'pe_%')\n);")
st.markdown(f"```")

st.subheader("Task 2.5: Design the Sector-Specific Attribute Tables")
st.markdown(f"")
st.markdown(f"Instead of using JSONB columns or adding many nullable columns to the main `organizations` table, we create separate, strongly-typed attribute tables for each sector. This keeps our data structured and queryable.")
st.markdown(f"")
st.markdown(f"For instance, here are examples for Manufacturing and Financial Services:")
st.markdown(f"```sql")
st.markdown(f"CREATE TABLE org_attributes_manufacturing (\n    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),\n    ot_systems VARCHAR(100)[],\n    it_ot_integration VARCHAR(20),\n    scada_vendor VARCHAR(100),\n    mes_system VARCHAR(100),\n    plant_count INTEGER,\n    automation_level VARCHAR(20),\n    iot_platforms VARCHAR(100)[],\n    digital_twin_status VARCHAR(20),\n    edge_computing BOOLEAN DEFAULT FALSE,\n    supply_chain_visibility VARCHAR(20),\n    demand_forecasting_ai BOOLEAN DEFAULT FALSE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);")
st.markdown(f"")
st.markdown(f"CREATE TABLE org_attributes_financial_services (\n    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),\n    regulatory_bodies VARCHAR(50)[],\n    charter_type VARCHAR(50),\n    model_risk_framework VARCHAR(50),\n    mrm_team_size INTEGER,\n    model_inventory_count INTEGER,\n    algo_trading BOOLEAN DEFAULT FALSE,\n    fraud_detection_ai BOOLEAN DEFAULT FALSE,\n    credit_ai BOOLEAN DEFAULT FALSE,\n    aml_ai BOOLEAN DEFAULT FALSE,\n    aum_billions DECIMAL(12,2),\n    total_assets_billions DECIMAL(12,2),\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);")
st.markdown(f"```")
st.markdown(f"Similar tables will be created for Healthcare, Technology, Retail, Energy, and Professional Services.")
st.markdown(f"")

st.markdown(f"---")
st.subheader("Action: Initialize Database Schema")
st.markdown(f"Click the button below to create all necessary tables for our configuration-driven data architecture.")
if not st.session_state.db_initialized:
    if st.button("Initialize Database Schema"):
        with st.spinner("Creating tables..."):
            success = setup_database_schema()
            if success:
                st.session_state.db_initialized = True
                st.success("Database schema initialized successfully!")
                st.balloons()
            else:
                st.error("Failed to initialize database schema.")
else:
    st.info("Database schema is already initialized.")
```

**Function Calls**:
*   `setup_database_schema()`: Called when the "Initialize Database Schema" button is clicked. Updates `st.session_state.db_initialized`.

---

#### Page: 2.2-2.3: Data Seeding

**Purpose**: Demonstrate how to populate the configuration tables with sector-specific data.

**Markdown Content & UI**:

```python
st.title("Task 2.2-2.3: Seed Sector Configuration Data")
st.markdown(f"")
st.markdown(f"With our schema in place, it's time to populate our configuration tables. This is where we define the unique characteristics for each PE sector as data rows, rather than schema changes.")
st.markdown(f"")

if not st.session_state.db_initialized:
    st.warning("Please initialize the database schema first in the 'Schema Design & Attributes' section.")
else:
    st.subheader("Action: Seed Initial Focus Groups and Dimensions")
    st.markdown(f"First, we'll populate the base `focus_groups` and `dimensions` tables.")
    if not st.session_state.initial_data_seeded:
        if st.button("Seed Initial Focus Groups & Dimensions"):
            with st.spinner("Seeding initial data..."):
                success = seed_initial_data()
                if success:
                    st.session_state.initial_data_seeded = True
                    st.session_state.all_focus_groups = get_all_focus_groups() # Populate for future selects
                    st.success("Initial focus groups and dimensions seeded!")
                else:
                    st.error("Failed to seed initial data.")
    else:
        st.info("Initial focus groups and dimensions already seeded.")
        st.markdown(f"")
        st.markdown(f"**Available Focus Groups:**")
        if st.session_state.all_focus_groups:
            st.dataframe(pd.DataFrame(st.session_state.all_focus_groups))

    if st.session_state.initial_data_seeded:
        st.subheader("Task 2.2: Seed Sector Dimension Weights")
        st.markdown(f"")
        st.markdown(f"Dimension weights are critical for our scoring models, indicating the relative importance of different AI/data dimensions for each sector. These are stored in `focus_group_dimension_weights`.")
        st.markdown(f"")
        st.markdown(f"Example `INSERT` statement for Manufacturing:")
        st.markdown(f"```sql")
        st.markdown(f"INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES\n    ('pe_manufacturing', 'pe_dim_data_infra', 0.22, 'OT/IT integration critical'),\n    ('pe_manufacturing', 'pe_dim_governance', 0.12, 'Less regulatory than finance/health'),\n    -- ... more weights for manufacturing ...\n    ('pe_financial_services', 'pe_dim_data_infra', 0.16, 'Mature infrastructure'),\n    -- ... and so on for all 7 sectors and 7 dimensions ...")
        st.markdown(f"```")
        
        st.subheader("Task 2.3: Seed Sector Calibrations")
        st.markdown(f"")
        st.markdown(f"Sector calibrations hold specific numeric or categorical parameters unique to each sector, like 'H&R Baseline' or 'EBITDA Multiplier'. These are stored in `focus_group_calibrations`.")
        st.markdown(f"")
        st.markdown(f"Example `INSERT` statement for Manufacturing:")
        st.markdown(f"```sql")
        st.markdown(f"INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES\n    ('pe_manufacturing', 'h_r_baseline', 72, 'numeric', 'Systematic opportunity baseline'),\n    ('pe_manufacturing', 'ebitda_multiplier', 0.90, 'numeric', 'Conservative EBITDA attribution'),\n    -- ... more calibrations for manufacturing ...\n    ('pe_financial_services', 'h_r_baseline', 82, 'numeric', 'Higher due to data maturity'),\n    -- ... and so on for all 7 sectors ...")
        st.markdown(f"```")

        st.markdown(f"---")
        st.subheader("Action: Seed All Dimension Weights and Calibrations")
        st.markdown(f"Click the button below to seed all dimension weights and calibration parameters for all 7 PE sectors.")
        if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
            if st.button("Seed All Weights and Calibrations"):
                with st.spinner("Seeding dimension weights..."):
                    weights_success = seed_dimension_weights_for_all_sectors()
                    if weights_success:
                        st.session_state.weights_seeded = True
                        st.success("Dimension weights seeded successfully!")
                    else:
                        st.error("Failed to seed dimension weights.")

                with st.spinner("Seeding calibrations..."):
                    calib_success = seed_calibrations_for_all_sectors()
                    if calib_success:
                        st.session_state.calibrations_seeded = True
                        st.success("Calibrations seeded successfully!")
                    else:
                        st.error("Failed to seed calibrations.")
                if weights_success and calib_success:
                    st.balloons()
        else:
            st.info("All dimension weights and calibrations are already seeded.")
```

**Function Calls**:
*   `seed_initial_data()`: Called when "Seed Initial Focus Groups & Dimensions" is clicked. Updates `st.session_state.initial_data_seeded`.
*   `get_all_focus_groups()`: Called after `seed_initial_data()` to populate `st.session_state.all_focus_groups`.
*   `seed_dimension_weights_for_all_sectors()`: Called when "Seed All Weights and Calibrations" is clicked. Updates `st.session_state.weights_seeded`.
*   `seed_calibrations_for_all_sectors()`: Called when "Seed All Weights and Calibrations" is clicked. Updates `st.session_state.calibrations_seeded`.

---

#### Page: 2.6: Sector Configuration Service

**Purpose**: Demonstrate the `SectorConfigService` for dynamic retrieval of sector configurations.

**Markdown Content & UI**:

```python
st.title("Task 2.6: Build the Sector Configuration Service")
st.markdown(f"")
st.markdown(f"Now that our configuration data is seeded, we need a service to efficiently retrieve these settings for specific sectors. This service encapsulates the logic for loading all relevant configuration parameters (weights, calibrations) into a single, easy-to-use object.")
st.markdown(f"")
st.markdown(f"The `SectorConfig` dataclass and `SectorConfigService` class are designed for this purpose. The `SectorConfig` object provides convenient properties and methods to access sector-specific parameters.")
st.markdown(f"")
st.markdown(f"Here's a simplified representation of the `SectorConfig` dataclass:")
st.markdown(f"```python")
st.markdown(f"@dataclass\nclass SectorConfig:\n    focus_group_id: str\n    group_name: str\n    group_code: str\n    dimension_weights: Dict[str, Decimal] = field(default_factory=dict)\n    calibrations: Dict[str, Decimal] = field(default_factory=dict)\n\n    @property\n    def h_r_baseline(self) -> Decimal:\n        # Retrieves H^R baseline from calibrations\n        return self.calibrations.get('h_r_baseline', Decimal('75'))\n\n    def get_dimension_weight(self, dimension_code: str) -> Decimal:\n        # Retrieves weight for a specific dimension\n        return self.dimension_weights.get(dimension_code, Decimal('0'))\n\n    def validate_weights_sum(self) -> bool:\n        # Verifies if dimension weights sum to 1.0\n        total = sum(self.dimension_weights.values())\n        return abs(total - Decimal('1.0')) < Decimal('0.001')")
st.markdown(f"```")

st.markdown(f"")
st.markdown(f"The `SectorConfigService` provides methods like `get_config(focus_group_id)` to load a full configuration object for a given sector from the database.")
st.markdown(f"")

if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
    st.warning("Please seed dimension weights and calibrations first in the 'Data Seeding' section.")
else:
    st.subheader("Demonstration: Retrieve & Analyze Sector Configurations")
    st.markdown(f"Select a PE sector below to retrieve its configuration using the `SectorConfigService` and examine its parameters.")

    fg_names = {fg['focus_group_id']: fg['group_name'] for fg in st.session_state.all_focus_groups}
    selected_fg_name = st.selectbox(
        "Select a Sector:",
        options=list(fg_names.values()),
        index=0,
        key="config_service_sector_select"
    )
    st.session_state.selected_sector_id = next(fg_id for fg_id, name in fg_names.items() if name == selected_fg_name)

    if st.session_state.selected_sector_id:
        st.markdown(f"")
        st.markdown(f"Calling `sector_service.get_config('{st.session_state.selected_sector_id}')`...")

        # Cache the fetched config to avoid repeated calls within the same interaction
        if st.session_state.selected_sector_id not in st.session_state.sector_configs_cache:
            sector_config = get_sector_config_from_service_sync(st.session_state.selected_sector_id)
            if sector_config:
                st.session_state.sector_configs_cache[st.session_state.selected_sector_id] = sector_config
        else:
            sector_config = st.session_state.sector_configs_cache[st.session_state.selected_sector_id]

        if sector_config:
            st.success(f"Configuration loaded for **{sector_config.group_name}** ({sector_config.focus_group_id})!")
            st.markdown(f"")
            st.subheader(f"Parameters for {sector_config.group_name}:")
            st.markdown(f"")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**H^R Baseline:** `{sector_config.h_r_baseline}`")
                st.markdown(r"where $H^R$ represents the Human-Readiness baseline score for a sector.")
                st.markdown(f"**EBITDA Multiplier:** `{sector_config.ebitda_multiplier}`")
                st.markdown(r"where $EBITDA$ is Earnings Before Interest, Taxes, Depreciation, and Amortization, and the multiplier adjusts its attribution for the sector.")
            with col2:
                st.markdown(f"**Position Factor Delta:** `{sector_config.position_factor_delta}`")
                st.markdown(r"where $\delta$ is the adjustment factor for an organization's strategic position within the sector.")
                st.markdown(f"**Talent Concentration Threshold:** `{sector_config.talent_concentration_threshold}`")
                st.markdown(r"where the threshold identifies sectors with high concentration of specialized talent.")
            
            st.markdown(f"")
            st.subheader(f"Dimension Weights for {sector_config.group_name}:")
            st.markdown(f"These weights define the relative importance of each AI/data dimension for the selected sector. They should sum up to 1.0 (or very close).")
            st.markdown(f"")

            if sector_config.validate_weights_sum():
                st.success(f"Dimension weights sum to 1.0 (Validation successful).")
            else:
                st.warning(f"Dimension weights do NOT sum to 1.0 (Validation failed). Total: {sum(sector_config.dimension_weights.values())}")

            weights_df = get_dimension_weights_for_chart(sector_config)
            st.dataframe(weights_df, use_container_width=True)

            fig = px.bar(weights_df, x='Dimension', y='Weight',
                         title=f'Dimension Weights for {sector_config.group_name}',
                         labels={'Weight': 'Relative Importance'},
                         color='Weight', color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not retrieve sector configuration.")
```

**Function Calls**:
*   `get_sector_config_from_service_sync(st.session_state.selected_sector_id)`: Called when a sector is selected to fetch its configuration. The result is cached in `st.session_state.sector_configs_cache`.
*   `get_dimension_weights_for_chart(sector_config)`: Helper function to prepare data for `plotly.express` bar chart.

---

#### Page: 2.7: Redis Caching Layer

**Purpose**: Explain and demonstrate the Redis caching layer for the `SectorConfigService`.

**Markdown Content & UI**:

```python
st.title("Task 2.7: Build the Redis Caching Layer")
st.markdown(f"")
st.markdown(f"To ensure our configuration service is highly performant and responsive, especially under heavy load, we introduce a Redis caching layer. Redis is an in-memory data store that provides extremely fast key-value lookups.")
st.markdown(f"")
st.markdown(f"The `SectorConfigService` is enhanced to first check the cache before hitting the database. If the configuration is found in Redis, it's a **cache hit**; otherwise, it's a **cache miss**, and the data is loaded from the database and then stored in Redis for future requests.")
st.markdown(f"")
st.markdown(f"Here's how caching is integrated into the `get_config` method:")
st.markdown(f"```python")
st.markdown(f"class SectorConfigService:\n    CACHE_KEY_SECTOR = \"sector:{{focus_group_id}}\"\n    CACHE_KEY_ALL = \"sectors:all\"\n    CACHE_TTL = 3600 # 1 hour\n\n    async def get_config(self, focus_group_id: str) -> Optional[SectorConfig]:\n        cache_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)\n        cached = cache.get(cache_key)\n        if cached:\n            # Cache HIT\n            return SectorConfig._dict_to_config(cached)\n\n        # Cache MISS - load from database\n        config = await self._load_from_db(focus_group_id)\n        if config:\n            cache.set(cache_key, config._config_to_dict(), self.CACHE_TTL)\n        return config")
st.markdown(f"```")

st.markdown(f"")
st.markdown(f"Furthermore, a mechanism for **cache invalidation** is crucial. When configuration data changes in the database, the corresponding cached entries must be removed or updated to prevent serving stale data. The `invalidate_cache` method handles this.")
st.markdown(f"")

if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
    st.warning("Please seed dimension weights and calibrations first in the 'Data Seeding' section to demonstrate caching.")
else:
    st.subheader("Demonstration: Caching Behavior and Invalidation")
    st.markdown(f"")
    st.markdown(f"Select a sector and click 'Fetch Config (with cache)' multiple times to observe the caching. Then, try 'Invalidate Cache' and fetch again.")

    fg_names = {fg['focus_group_id']: fg['group_name'] for fg in st.session_state.all_focus_groups}
    all_fg_options = ["All Sectors (invalidate all)"] + list(fg_names.values())

    selected_invalidate_scope_name = st.selectbox(
        "Select scope for cache invalidation:",
        options=all_fg_options,
        index=0,
        key="cache_invalidate_scope_select"
    )
    
    invalidate_fg_id = None
    if selected_invalidate_scope_name != "All Sectors (invalidate all)":
        invalidate_fg_id = next(fg_id for fg_id, name in fg_names.items() if name == selected_invalidate_scope_name)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Invalidate Cache"):
            with st.spinner(f"Invalidating cache for {selected_invalidate_scope_name}..."):
                invalidate_sector_cache_service_sync(invalidate_fg_id)
                st.session_state.sector_configs_cache = {} # Clear local app cache as well
                st.success("Cache invalidation triggered!")
    with col2:
        # Re-fetch from DB and set cache
        selected_fetch_fg_name = st.selectbox(
            "Select a Sector to Fetch (observe cache hits/misses):",
            options=list(fg_names.values()),
            index=0,
            key="cache_fetch_sector_select"
        )
        fetch_fg_id = next(fg_id for fg_id, name in fg_names.items() if name == selected_fetch_fg_name)
        
        if st.button("Fetch Config (with cache)", key="fetch_config_button"):
            with st.spinner(f"Fetching config for {selected_fetch_fg_name}..."):
                fetched_config = get_sector_config_from_service_sync(fetch_fg_id)
                if fetched_config:
                    st.success(f"Successfully fetched config for {fetched_config.group_name}.")
                    st.json(fetched_config._config_to_dict()) # Display raw data from source.py function

```

**Function Calls**:
*   `invalidate_sector_cache_service_sync(invalidate_fg_id)`: Called when "Invalidate Cache" is clicked. Clears the selected cache entries.
*   `get_sector_config_from_service_sync(fetch_fg_id)`: Called when "Fetch Config (with cache)" is clicked. Demonstrates cache hit/miss behavior.

---

#### Page: 2.8: Unified Organization View

**Purpose**: Show how to query and display a unified view of organizations, including their sector-specific attributes, leveraging the architecture built.

**Markdown Content & UI**:

```python
st.title("Task 2.8: Create the Unified Organization View")
st.markdown(f"")
st.markdown(f"The ultimate goal of our configuration-driven architecture is to provide a unified, easily queryable view of organizations, enriching their core data with sector-specific attributes dynamically.")
st.markdown(f"")
st.markdown(f"This is achieved by creating a database VIEW (`vw_organizations_full`) that joins the main `organizations` table with the respective `org_attributes_` tables based on the `focus_group_id` reference.")
st.markdown(f"")
st.markdown(f"Here's a simplified representation of the SQL VIEW definition:")
st.markdown(f"```sql")
st.markdown(f"CREATE OR REPLACE VIEW vw_organizations_full AS\nSELECT\n    o.*,\n    fg.group_name AS sector_name,\n    fg.group_code AS sector_code,\n    -- Manufacturing specific attributes\n    mfg.plant_count, mfg.automation_level, mfg.digital_twin_status,\n    -- Financial Services specific attributes\n    fin.regulatory_bodies, fin.algo_trading, fin.aum_billions,\n    -- ... similar attributes for other sectors ...\nFROM organizations o\nJOIN focus_groups fg ON o.focus_group_id = fg.focus_group_id\nLEFT JOIN org_attributes_manufacturing mfg ON o.organization_id = mfg.organization_id\nLEFT JOIN org_attributes_financial_services fin ON o.organization_id = fin.organization_id\n-- ... left joins for all other org_attributes_ tables ...\nLEFT JOIN org_attributes_professional_services ps ON o.organization_id = ps.organization_id;")
st.markdown(f"```")
st.markdown(f"")
st.markdown(f"This view allows us to query all organizations and their relevant sector attributes in a single statement, without needing to know which specific attribute table to query beforehand. The `LEFT JOIN` ensures that even organizations without specific attributes in a given table are still included.")
st.markdown(f"")

if not st.session_state.db_initialized:
    st.warning("Please initialize the database schema first in the 'Schema Design & Attributes' section.")
elif not st.session_state.initial_data_seeded:
    st.warning("Please seed initial focus groups and dimensions first in the 'Data Seeding' section.")
else:
    st.subheader("Action: Insert Sample Organizations")
    st.markdown(f"Let's populate our `organizations` table with some sample data, ensuring they are linked to different PE sectors.")
    if not st.session_state.organizations_seeded:
        if st.button("Insert Sample Organizations"):
            with st.spinner("Inserting sample organizations and attributes..."):
                success = insert_sample_organizations(num_orgs=10)
                if success:
                    st.session_state.organizations_seeded = True
                    st.success("10 sample organizations inserted with sector-specific attributes!")
                    st.balloons()
                else:
                    st.error("Failed to insert sample organizations. Ensure base data is seeded.")
    else:
        st.info("Sample organizations already inserted.")

    if st.session_state.organizations_seeded:
        st.subheader("Demonstration: Query Unified Organization View")
        st.markdown(f"")
        st.markdown(f"Now, let's query the `vw_organizations_full` view. You can filter by a specific sector or view all organizations.")

        fg_names = {fg['focus_group_id']: fg['group_name'] for fg in st.session_state.all_focus_groups}
        filter_options = {"All Sectors": None}
        filter_options.update({fg['group_name']: fg['focus_group_id'] for fg in st.session_state.all_focus_groups})

        selected_filter_name = st.selectbox(
            "Filter Organizations by Sector:",
            options=list(filter_options.keys()),
            index=0,
            key="unified_view_filter_select"
        )
        st.session_state.unified_org_filter_sector_id = filter_options[selected_filter_name]

        if st.button("Fetch Unified Organization Data"):
            with st.spinner(f"Fetching data for {selected_filter_name}..."):
                org_data_df = fetch_unified_organization_data_sync(st.session_state.unified_org_filter_sector_id)
                if not org_data_df.empty:
                    st.dataframe(org_data_df, use_container_width=True)
                    st.success(f"Displayed {len(org_data_df)} organizations.")
                else:
                    st.info("No organizations found for the selected filter.")
```

**Function Calls**:
*   `insert_sample_organizations(num_orgs=10)`: Called when "Insert Sample Organizations" is clicked. Updates `st.session_state.organizations_seeded`.
*   `fetch_unified_organization_data_sync(st.session_state.unified_org_filter_sector_id)`: Called when "Fetch Unified Organization Data" is clicked. Displays the resulting DataFrame.

