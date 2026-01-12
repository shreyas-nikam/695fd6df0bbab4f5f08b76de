

import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Any, Optional
import time # For simulating delays

# --- Placeholder/Mock Implementations for 'source.py' content ---
# This section replaces 'from source import *'

# Mock data for focus groups
sample_focus_groups = [
    {'focus_group_id': 'pe_manufacturing', 'platform': 'pe_org_air', 'group_name': 'Manufacturing', 'group_code': 'MFG', 'display_order': 1, 'icon_name': 'industry', 'color_hex': '#FF5733', 'is_active': True},
    {'focus_group_id': 'pe_financial_services', 'platform': 'pe_org_air', 'group_name': 'Financial Services', 'group_code': 'FIN', 'display_order': 2, 'icon_name': 'bank', 'color_hex': '#3366FF', 'is_active': True},
    {'focus_group_id': 'pe_healthcare', 'platform': 'pe_org_air', 'group_name': 'Healthcare', 'group_code': 'HLT', 'display_order': 3, 'icon_name': 'health', 'color_hex': '#33FF57', 'is_active': True},
    {'focus_group_id': 'pe_technology', 'platform': 'pe_org_air', 'group_name': 'Technology', 'group_code': 'TECH', 'display_order': 4, 'icon_name': 'laptop', 'color_hex': '#FF33EC', 'is_active': True},
    {'focus_group_id': 'pe_retail', 'platform': 'pe_org_air', 'group_name': 'Retail', 'group_code': 'RET', 'display_order': 5, 'icon_name': 'shopping_cart', 'color_hex': '#FF8C33', 'is_active': True},
    {'focus_group_id': 'pe_energy', 'platform': 'pe_org_air', 'group_name': 'Energy', 'group_code': 'ENG', 'display_order': 6, 'icon_name': 'lightbulb', 'color_hex': '#33FFEE', 'is_active': True},
    {'focus_group_id': 'pe_professional_services', 'platform': 'pe_org_air', 'group_name': 'Professional Services', 'group_code': 'PRO', 'display_order': 7, 'icon_name': 'briefcase', 'color_hex': '#8C33FF', 'is_active': True},
]

# Mock data for dimensions (simplified)
sample_dimensions = [
    {'dimension_id': 'pe_dim_data_infra', 'platform': 'pe_org_air', 'dimension_name': 'Data Infrastructure', 'dimension_code': 'data_infra', 'display_order': 1},
    {'dimension_id': 'pe_dim_governance', 'platform': 'pe_org_air', 'dimension_name': 'Data Governance', 'dimension_code': 'governance', 'display_order': 2},
    {'dimension_id': 'pe_dim_tech_adoption', 'platform': 'pe_org_air', 'dimension_name': 'Technology Adoption', 'dimension_code': 'tech_adoption', 'display_order': 3},
    {'dimension_id': 'pe_dim_analytics_ml', 'platform': 'pe_org_air', 'dimension_name': 'Analytics & ML', 'dimension_code': 'analytics_ml', 'display_order': 4},
    {'dimension_id': 'pe_dim_talent', 'platform': 'pe_org_air', 'dimension_name': 'Talent & Culture', 'dimension_code': 'talent', 'display_order': 5},
    {'dimension_id': 'pe_dim_process_automation', 'platform': 'pe_org_air', 'dimension_name': 'Process Automation', 'dimension_code': 'process_automation', 'display_order': 6},
]

@dataclass
class SectorConfig:
    focus_group_id: str
    group_name: str
    group_code: str
    dimension_weights: Dict[str, Decimal] = field(default_factory=dict)
    calibrations: Dict[str, Decimal] = field(default_factory=dict)

    @property
    def h_r_baseline(self) -> Decimal:
        return self.calibrations.get('h_r_baseline', Decimal('75'))

    @property
    def ebitda_multiplier(self) -> Decimal:
        return self.calibrations.get('ebitda_multiplier', Decimal('0.85'))

    @property
    def position_factor_delta(self) -> Decimal:
        return self.calibrations.get('position_factor_delta', Decimal('0.05'))

    @property
    def talent_concentration_threshold(self) -> Decimal:
        return self.calibrations.get('talent_concentration_threshold', Decimal('0.7'))

    def get_dimension_weight(self, dimension_code: str) -> Decimal:
        return self.dimension_weights.get(dimension_code, Decimal('0'))

    def validate_weights_sum(self) -> bool:
        total = sum(self.dimension_weights.values())
        return abs(total - Decimal('1.0')) < Decimal('0.001')

    def _config_to_dict(self) -> Dict[str, Any]:
        return {
            "focus_group_id": self.focus_group_id,
            "group_name": self.group_name,
            "group_code": self.group_code,
            "dimension_weights": {k: str(v) for k, v in self.dimension_weights.items()},
            "calibrations": {k: str(v) for k, v in self.calibrations.items()},
        }

    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> 'SectorConfig':
        return SectorConfig(
            focus_group_id=data["focus_group_id"],
            group_name=data["group_name"],
            group_code=data["group_code"],
            dimension_weights={k: Decimal(v) for k, v in data["dimension_weights"].items()},
            calibrations={k: Decimal(v) for k, v in data["calibrations"].items()},
        )

# Mock Redis Cache
class MockRedisCache:
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            st.info(f"Cache HIT for key: {key}")
            return self._cache[key]
        st.warning(f"Cache MISS for key: {key}")
        return None

    def set(self, key: str, value: Dict[str, Any], ttl: int):
        self._cache[key] = value
        st.success(f"Cache SET for key: {key}")

    def delete(self, key: str):
        if key in self._cache:
            del self._cache[key]
            st.success(f"Cache DELETE for key: {key}")
        else:
            st.info(f"Cache DELETE: Key {key} not found.")

    def flushall(self):
        self._cache = {}
        st.success("Cache FLUSHALL executed.")

cache = MockRedisCache()

# Mock functions for database operations
def setup_database_schema() -> bool:
    time.sleep(1) # Simulate DB operation
    # In a real app, this would execute DDL statements.
    # For this mock, we just return success.
    return True

def seed_initial_data() -> bool:
    time.sleep(1) # Simulate data seeding
    # This would populate focus_groups and dimensions tables
    return True

def get_all_focus_groups() -> List[Dict[str, Any]]:
    # In a real app, this would query the focus_groups table
    return sample_focus_groups

def seed_dimension_weights_for_all_sectors() -> bool:
    time.sleep(1) # Simulate data seeding
    # This would populate focus_group_dimension_weights
    return True

def seed_calibrations_for_all_sectors() -> bool:
    time.sleep(1) # Simulate data seeding
    # This would populate focus_group_calibrations
    return True

def get_sector_config_from_service_sync(focus_group_id: str) -> Optional[SectorConfig]:
    # Simulate caching logic
    cache_key = f"sector:{focus_group_id}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return SectorConfig._dict_to_config(cached_data)

    time.sleep(1) # Simulate DB fetch delay
    # Mock data generation for config (similar to the logic in thought process)
    for fg in sample_focus_groups:
        if fg['focus_group_id'] == focus_group_id:
            if focus_group_id == 'pe_manufacturing':
                weights = {'data_infra': Decimal('0.22'), 'governance': Decimal('0.12'), 'tech_adoption': Decimal('0.20'), 'analytics_ml': Decimal('0.18'), 'talent': Decimal('0.15'), 'process_automation': Decimal('0.13')}
                calibrations = {'h_r_baseline': Decimal('72'), 'ebitda_multiplier': Decimal('0.90'), 'position_factor_delta': Decimal('0.05'), 'talent_concentration_threshold': Decimal('0.65')}
            elif focus_group_id == 'pe_financial_services':
                weights = {'data_infra': Decimal('0.16'), 'governance': Decimal('0.25'), 'tech_adoption': Decimal('0.15'), 'analytics_ml': Decimal('0.24'), 'talent': Decimal('0.12'), 'process_automation': Decimal('0.08')}
                calibrations = {'h_r_baseline': Decimal('82'), 'ebitda_multiplier': Decimal('0.80'), 'position_factor_delta': Decimal('0.07'), 'talent_concentration_threshold': Decimal('0.75')}
            else: # Default for other sectors
                weights = {'data_infra': Decimal('0.18'), 'governance': Decimal('0.15'), 'tech_adoption': Decimal('0.19'), 'analytics_ml': Decimal('0.17'), 'talent': Decimal('0.16'), 'process_automation': Decimal('0.15')}
                calibrations = {'h_r_baseline': Decimal('75'), 'ebitda_multiplier': Decimal('0.85'), 'position_factor_delta': Decimal('0.06'), 'talent_concentration_threshold': Decimal('0.70')}
            
            # Ensure sum to 1.0 (for validation demo)
            current_sum = sum(weights.values())
            if current_sum != Decimal('1.0'):
                adjustment_factor = Decimal('1.0') / current_sum
                weights = {k: round(v * adjustment_factor, 3) for k, v in weights.items()}
                # Re-check sum after rounding to ensure it's still close for validation
                if abs(sum(weights.values()) - Decimal('1.0')) > Decimal('0.001'):
                    # Distribute remaining diff to first element
                    diff = Decimal('1.0') - sum(weights.values())
                    if weights:
                        first_key = list(weights.keys())[0]
                        weights[first_key] += diff


            config = SectorConfig(
                focus_group_id=fg['focus_group_id'],
                group_name=fg['group_name'],
                group_code=fg['group_code'],
                dimension_weights=weights,
                calibrations=calibrations
            )
            cache.set(cache_key, config._config_to_dict(), 3600) # Cache for 1 hour
            return config
    return None

def invalidate_sector_cache_service_sync(focus_group_id: Optional[str]) -> None:
    if focus_group_id:
        cache.delete(f"sector:{focus_group_id}")
    else:
        cache.flushall()

def get_dimension_weights_for_chart(sector_config: SectorConfig) -> pd.DataFrame:
    # Map dimension codes to names for better display
    dim_code_to_name = {dim['dimension_code']: dim['dimension_name'] for dim in sample_dimensions}
    weights_data = [{'Dimension': dim_code_to_name.get(dim_code, dim_code), 'Weight': float(weight)}
                    for dim_code, weight in sector_config.dimension_weights.items()]
    return pd.DataFrame(weights_data)

def insert_sample_organizations(num_orgs: int) -> bool:
    time.sleep(1) # Simulate DB operation
    # This would insert data into organizations and org_attributes_* tables
    return True

def fetch_unified_organization_data_sync(sector_id: Optional[str]) -> pd.DataFrame:
    time.sleep(1) # Simulate DB fetch delay
    data = []
    # Simplified dimensions
    
    for i in range(1, 11): # 10 dummy organizations
        org_fg = sample_focus_groups[i % len(sample_focus_groups)]
        org_fg_id = org_fg['focus_group_id']
        org_fg_name = org_fg['group_name']
        
        if sector_id and org_fg_id != sector_id:
            continue

        row = {
            'organization_id': f'org-{i:03d}',
            'legal_name': f'Org {i} Corp',
            'display_name': f'Org {i}',
            'ticker_symbol': f'ORG{i}',
            'focus_group_id': org_fg_id,
            'sector_name': org_fg_name,
            'employee_count': 1000 + i * 100,
            'annual_revenue_usd': 100_000_000 + i * 10_000_000,
            'headquarters_country': 'USA',
            'headquarters_state': f'State {i % 5}',
            'headquarters_city': f'City {i}',
            'website_url': f'http://www.org{i}.com',
            'status': 'active',
            'created_at': pd.Timestamp.now(),
            'updated_at': pd.Timestamp.now(),
            'created_by': 'admin',
            # Attributes for Manufacturing
            'plant_count': 5 if org_fg_id == 'pe_manufacturing' else None,
            'automation_level': 'High' if org_fg_id == 'pe_manufacturing' else None,
            'digital_twin_status': 'Implemented' if org_fg_id == 'pe_manufacturing' else None,
            'ot_systems': ['SCADA', 'DCS'] if org_fg_id == 'pe_manufacturing' else None,
            'it_ot_integration': 'Partial' if org_fg_id == 'pe_manufacturing' else None,

            # Attributes for Financial Services
            'regulatory_bodies': ['SEC', 'FINRA'] if org_fg_id == 'pe_financial_services' else None,
            'charter_type': 'Commercial Bank' if org_fg_id == 'pe_financial_services' else None,
            'algo_trading': True if org_fg_id == 'pe_financial_services' else None,
            'aum_billions': 50.0 + i * 2.5 if org_fg_id == 'pe_financial_services' else None,
            'total_assets_billions': 100.0 + i * 5.0 if org_fg_id == 'pe_financial_services' else None,
        }
        data.append(row)
    
    if data:
        all_cols = list(data[0].keys())
        df = pd.DataFrame(data, columns=all_cols)
    else:
        df = pd.DataFrame(data)

    return df

# --- End of Placeholder/Mock Implementations ---


st.set_page_config(page_title="QuLab: Data Layer & Caching", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Data Layer & Caching")
st.divider()

# Initialize Session State
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
    st.session_state.all_focus_groups = get_all_focus_groups()
if 'selected_sector_id' not in st.session_state:
    st.session_state.selected_sector_id = None
# Removed 'sector_configs_cache' from session state initialization as it was bypassing mock Redis cache
if 'unified_org_filter_sector_id' not in st.session_state:
    st.session_state.unified_org_filter_sector_id = None

# Sidebar Navigation
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
    
    current_index = 0
    if st.session_state.current_page in page_options:
        current_index = page_options.index(st.session_state.current_page)
        
    st.session_state.current_page = st.selectbox(
        "Go to section:",
        page_options,
        index=current_index
    )

# Page: Home
if st.session_state.current_page == "Home":
    st.title("Week 2: Unified Data Architecture & Caching Lab")

    st.markdown("")
    st.markdown("Welcome to Week 2 of our journey into building a robust Private Equity (PE) Intelligence Platform!")
    st.markdown("")
    st.markdown("In this lab, you'll transition from foundational setup to designing a truly configuration-driven data architecture.")
    st.markdown("This approach is crucial for managing the complexity of diverse PE sectors without resorting to schema proliferation.")
    st.markdown("")
    st.subheader("Key Objectives")
    st.markdown("")
    st.markdown("- **Remember**: List the 7 PE sectors and their configuration parameters.")
    st.markdown("- **Understand**: Explain why configuration-driven architecture avoids schema proliferation.")
    st.markdown("- **Apply**: Implement focus group configuration loading with caching.")
    st.markdown("- **Analyze**: Compare sector attribute tables vs JSONB approaches.")
    st.markdown("- **Evaluate**: Assess dimension weight configurations for different sectors.")
    st.markdown("")
    st.subheader("Tools Introduced")
    st.markdown("")
    st.markdown("This week, we'll be working with powerful tools to achieve our architectural goals:")
    st.markdown("")
    st.markdown("- **PostgreSQL / Snowflake**: Our primary database, supporting both development and production environments.")
    st.markdown("- **SQLAlchemy 2.0**: An ORM layer for advanced database interactions.")
    st.markdown("- **Alembic**: For version-controlled schema migrations, ensuring smooth database evolution.")
    st.markdown("- **Redis**: A fast in-memory data store for caching, essential for high-performance configuration lookups.")
    st.markdown("- **structlog**: For structured logging, enhancing observability of our services.")
    st.markdown("")
    st.subheader("Key Concepts")
    st.markdown("")
    st.markdown("The central theme for this week is **One Schema, Many Configurations**. This means:")
    st.markdown("")
    st.markdown("- We avoid creating separate schemas for each PE sector.")
    st.markdown("- Differentiation between sectors is achieved through data rows in configuration tables, not schema variations.")
    st.markdown("- Focus Group Configuration Tables store weights and calibrations as data rows.")
    st.markdown("- Queryable Sector Attribute Tables use typed columns instead of less flexible JSONB approaches.")
    st.markdown("- Configuration Caching ensures that frequently accessed configurations are loaded once and used everywhere, reducing database load.")

# Page: 2.1-2.5: Schema Design & Attributes
elif st.session_state.current_page == "2.1-2.5: Schema Design & Attributes":
    st.title("Task 2.1-2.5: Database Schema Design")
    st.markdown("")
    st.markdown("This section focuses on designing a flexible and extensible data architecture.")
    st.markdown("We'll define tables that allow for configuration-driven differentiation across PE sectors, avoiding the 'schema per sector' anti-pattern.")
    st.markdown("")

    st.subheader("Design Principle: One Schema, Many Configurations")
    st.markdown("")
    st.markdown("A core principle of our architecture is that all 7 PE sectors share identical base schemas. Sector-specific differentiation is achieved through configuration tables and dedicated attribute tables, rather than varying the base schema.")
    st.markdown("")
    st.markdown("This approach minimizes `N×M` joins, prevents `NULL` proliferation in central tables, and allows for robust querying of sector-specific attributes using typed columns.")
    st.markdown("")

    st.subheader("Task 2.1: Design the Focus Group Configuration Schema")
    st.markdown("")
    st.markdown("We'll start by defining the `focus_groups` table to store our primary sectors, along with `dimensions` and `focus_group_dimension_weights`, and `focus_group_calibrations` to hold sector-specific configuration parameters.")
    st.markdown("")
    st.markdown("Here's the PostgreSQL DDL for these tables:")
    st.markdown("```sql")
    st.markdown("CREATE TABLE focus_groups (\n    focus_group_id VARCHAR(50) PRIMARY KEY,\n    platform VARCHAR(20) NOT NULL CHECK (platform IN ('pe_org_air', 'individual_air')),\n    group_name VARCHAR(100) NOT NULL,\n    group_code VARCHAR(30) NOT NULL,\n    group_description TEXT,\n    display_order INTEGER NOT NULL,\n    icon_name VARCHAR(50),\n    color_hex VARCHAR(7),\n    is_active BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (platform, group_code)\n);")
    st.markdown("")
    st.markdown("CREATE TABLE dimensions (\n    dimension_id VARCHAR(50) PRIMARY KEY,\n    platform VARCHAR(20) NOT NULL,\n    dimension_name VARCHAR(100) NOT NULL,\n    dimension_code VARCHAR(50) NOT NULL,\n    description TEXT,\n    min_score DECIMAL(5,2) DEFAULT 0,\n    max_score DECIMAL(5,2) DEFAULT 100,\n    display_order INTEGER NOT NULL,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (platform, dimension_code)\n);")
    st.markdown("")
    st.markdown("CREATE TABLE focus_group_dimension_weights (\n    weight_id SERIAL PRIMARY KEY,\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    dimension_id VARCHAR(50) NOT NULL REFERENCES dimensions(dimension_id),\n    weight DECIMAL(4,3) NOT NULL CHECK (weight >= 0 AND weight <= 1),\n    weight_rationale TEXT,\n    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,\n    effective_to DATE,\n    is_current BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (focus_group_id, dimension_id, effective_from)\n);")
    st.markdown("")
    st.markdown("CREATE TABLE focus_group_calibrations (\n    calibration_id SERIAL PRIMARY KEY,\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    parameter_name VARCHAR(100) NOT NULL,\n    parameter_value DECIMAL(10,4) NOT NULL,\n    parameter_type VARCHAR(20) DEFAULT 'numeric',\n    description TEXT,\n    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,\n    effective_to DATE,\n    is_current BOOLEAN DEFAULT TRUE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    UNIQUE (focus_group_id, parameter_name, effective_from)\n);")
    st.markdown("```")

    st.subheader("Task 2.4: Design the Organizations Table with Sector Reference")
    st.markdown("")
    st.markdown("The `organizations` table will store core information about each company. Critically, it includes a `focus_group_id` as a foreign key to link each organization to its primary PE sector. This is the cornerstone of our configuration-driven approach.")
    st.markdown("")
    st.markdown("```sql")
    st.markdown("CREATE TABLE organizations (\n    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n    legal_name VARCHAR(255) NOT NULL,\n    display_name VARCHAR(255),\n    ticker_symbol VARCHAR(10),\n    cik_number VARCHAR(20),\n    duns_number VARCHAR(20),\n    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),\n    primary_sic_code VARCHAR(10),\n    primary_naics_code VARCHAR(10),\n    employee_count INTEGER,\n    annual_revenue_usd DECIMAL(15,2),\n    founding_year INTEGER,\n    headquarters_country VARCHAR(3),\n    headquarters_state VARCHAR(50),\n    headquarters_city VARCHAR(100),\n    website_url VARCHAR(500),\n    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n    created_by VARCHAR(100),\n    CONSTRAINT chk_org_pe_platform CHECK (focus_group_id LIKE 'pe_%')\n);")
    st.markdown("```")

    st.subheader("Task 2.5: Design the Sector-Specific Attribute Tables")
    st.markdown("")
    st.markdown("Instead of using JSONB columns or adding many nullable columns to the main `organizations` table, we create separate, strongly-typed attribute tables for each sector. This keeps our data structured and queryable.")
    st.markdown("")
    st.markdown("For instance, here are examples for Manufacturing and Financial Services:")
    st.markdown("```sql")
    st.markdown("CREATE TABLE org_attributes_manufacturing (\n    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),\n    ot_systems VARCHAR(100)[],\n    it_ot_integration VARCHAR(20),\n    scada_vendor VARCHAR(100),\n    mes_system VARCHAR(100),\n    plant_count INTEGER,\n    automation_level VARCHAR(20),\n    iot_platforms VARCHAR(100)[],\n    digital_twin_status VARCHAR(20),\n    edge_computing BOOLEAN DEFAULT FALSE,\n    supply_chain_visibility VARCHAR(20),\n    demand_forecasting_ai BOOLEAN DEFAULT FALSE,\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);")
    st.markdown("")
    st.markdown("CREATE TABLE org_attributes_financial_services (\n    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),\n    regulatory_bodies VARCHAR(50)[],\n    charter_type VARCHAR(50),\n    model_risk_framework VARCHAR(50),\n    mrm_team_size INTEGER,\n    model_inventory_count INTEGER,\n    algo_trading BOOLEAN DEFAULT FALSE,\n    fraud_detection_ai BOOLEAN DEFAULT FALSE,\n    credit_ai BOOLEAN DEFAULT FALSE,\n    aml_ai BOOLEAN DEFAULT FALSE,\n    aum_billions DECIMAL(12,2),\n    total_assets_billions DECIMAL(12,2),\n    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);")
    st.markdown("```")
    st.markdown("Similar tables will be created for Healthcare, Technology, Retail, Energy, and Professional Services.")
    st.markdown("")

    st.markdown("---")
    st.subheader("Action: Initialize Database Schema")
    st.markdown("Click the button below to create all necessary tables for our configuration-driven data architecture.")
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

# Page: 2.2-2.3: Data Seeding
elif st.session_state.current_page == "2.2-2.3: Data Seeding":
    st.title("Task 2.2-2.3: Seed Sector Configuration Data")
    st.markdown("")
    st.markdown("With our schema in place, it's time to populate our configuration tables. This is where we define the unique characteristics for each PE sector as data rows, rather than schema changes.")
    st.markdown("")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database schema first in the 'Schema Design & Attributes' section.")
    else:
        st.subheader("Action: Seed Initial Focus Groups and Dimensions")
        st.markdown("First, we'll populate the base `focus_groups` and `dimensions` tables.")
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
            st.markdown("")
            st.markdown("**Available Focus Groups:**")
            if st.session_state.all_focus_groups:
                st.dataframe(pd.DataFrame(st.session_state.all_focus_groups))

        if st.session_state.initial_data_seeded:
            st.subheader("Task 2.2: Seed Sector Dimension Weights")
            st.markdown("")
            st.markdown("Dimension weights are critical for our scoring models, indicating the relative importance of different AI/data dimensions for each sector. These are stored in `focus_group_dimension_weights`.")
            st.markdown("")
            st.markdown("Example `INSERT` statement for Manufacturing:")
            st.markdown("```sql")
            st.markdown("INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES\n    ('pe_manufacturing', 'pe_dim_data_infra', 0.22, 'OT/IT integration critical'),\n    ('pe_manufacturing', 'pe_dim_governance', 0.12, 'Less regulatory than finance/health'),\n    -- ... more weights for manufacturing ...\n    ('pe_financial_services', 'pe_dim_data_infra', 0.16, 'Mature infrastructure'),\n    -- ... and so on for all 7 sectors and 7 dimensions ...")
            st.markdown("```")
            
            st.subheader("Task 2.3: Seed Sector Calibrations")
            st.markdown("")
            st.markdown("Sector calibrations hold specific numeric or categorical parameters unique to each sector, like 'H&R Baseline' or 'EBITDA Multiplier'. These are stored in `focus_group_calibrations`.")
            st.markdown("")
            st.markdown("Example `INSERT` statement for Manufacturing:")
            st.markdown("```sql")
            st.markdown("INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES\n    ('pe_manufacturing', 'h_r_baseline', 72, 'numeric', 'Systematic opportunity baseline'),\n    ('pe_manufacturing', 'ebitda_multiplier', 0.90, 'numeric', 'Conservative EBITDA attribution'),\n    -- ... more calibrations for manufacturing ...\n    ('pe_financial_services', 'h_r_baseline', 82, 'numeric', 'Higher due to data maturity'),\n    -- ... and so on for all 7 sectors ...")
            st.markdown("```")

            st.markdown("---")
            st.subheader("Action: Seed All Dimension Weights and Calibrations")
            st.markdown("Click the button below to seed all dimension weights and calibration parameters for all 7 PE sectors.")
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

# Page: 2.6: Sector Configuration Service
elif st.session_state.current_page == "2.6: Sector Configuration Service":
    st.title("Task 2.6: Build the Sector Configuration Service")
    st.markdown("")
    st.markdown("Now that our configuration data is seeded, we need a service to efficiently retrieve these settings for specific sectors. This service encapsulates the logic for loading all relevant configuration parameters (weights, calibrations) into a single, easy-to-use object.")
    st.markdown("")
    st.markdown("The `SectorConfig` dataclass and `SectorConfigService` class are designed for this purpose. The `SectorConfig` object provides convenient properties and methods to access sector-specific parameters.")
    st.markdown("")
    st.markdown("Here's a simplified representation of the `SectorConfig` dataclass:")
    st.markdown("```python")
    st.markdown("@dataclass\nclass SectorConfig:\n    focus_group_id: str\n    group_name: str\n    group_code: str\n    dimension_weights: Dict[str, Decimal] = field(default_factory=dict)\n    calibrations: Dict[str, Decimal] = field(default_factory=dict)\n\n    @property\n    def h_r_baseline(self) -> Decimal:\n        # Retrieves H^R baseline from calibrations\n        return self.calibrations.get('h_r_baseline', Decimal('75'))\n\n    def get_dimension_weight(self, dimension_code: str) -> Decimal:\n        # Retrieves weight for a specific dimension\n        return self.dimension_weights.get(dimension_code, Decimal('0'))\n\n    def validate_weights_sum(self) -> bool:\n        # Verifies if dimension weights sum to 1.0\n        total = sum(self.dimension_weights.values())\n        return abs(total - Decimal('1.0')) < Decimal('0.001')")
    st.markdown("```")

    st.markdown("")
    st.markdown("The `SectorConfigService` provides methods like `get_config(focus_group_id)` to load a full configuration object for a given sector from the database.")
    st.markdown("")

    if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
        st.warning("Please seed dimension weights and calibrations first in the 'Data Seeding' section.")
    else:
        st.subheader("Demonstration: Retrieve & Analyze Sector Configurations")
        st.markdown("Select a PE sector below to retrieve its configuration using the `SectorConfigService` and examine its parameters.")

        fg_names = {fg['focus_group_id']: fg['group_name'] for fg in st.session_state.all_focus_groups}
        selected_fg_name = st.selectbox(
            "Select a Sector:",
            options=list(fg_names.values()),
            index=0,
            key="config_service_sector_select"
        )
        st.session_state.selected_sector_id = next((fg_id for fg_id, name in fg_names.items() if name == selected_fg_name), None)

        if st.session_state.selected_sector_id:
            st.markdown("")
            st.markdown(f"Calling `get_sector_config_from_service_sync('{st.session_state.selected_sector_id}')`...")

            # FIX: Removed the local Streamlit session state cache (st.session_state.sector_configs_cache)
            # This ensures that get_sector_config_from_service_sync is called every time,
            # allowing the MockRedisCache within it to demonstrate cache hits/misses.
            sector_config = get_sector_config_from_service_sync(st.session_state.selected_sector_id)

            if sector_config:
                st.success(f"Configuration loaded for **{sector_config.group_name}** ({sector_config.focus_group_id})!")
                st.markdown("")
                st.subheader(f"Parameters for {sector_config.group_name}:")
                st.markdown("")

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
                
                st.markdown("")
                st.subheader(f"Dimension Weights for {sector_config.group_name}:")
                st.markdown(f"These weights define the relative importance of each AI/data dimension for the selected sector. They should sum up to 1.0 (or very close).")
                st.markdown("")

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

# Page: 2.7: Redis Caching Layer
elif st.session_state.current_page == "2.7: Redis Caching Layer":
    st.title("Task 2.7: Build the Redis Caching Layer")
    st.markdown("")
    st.markdown("To ensure our configuration service is highly performant and responsive, especially under heavy load, we introduce a Redis caching layer. Redis is an in-memory data store that provides extremely fast key-value lookups.")
    st.markdown("")
    st.markdown("The `SectorConfigService` is enhanced to first check the cache before hitting the database. If the configuration is found in Redis, it's a **cache hit**; otherwise, it's a **cache miss**, and the data is loaded from the database and then stored in Redis for future requests.")
    st.markdown("")
    st.markdown("Here's how caching is integrated into the `get_config` method:")
    st.markdown("```python")
    st.markdown("class SectorConfigService:\n    CACHE_KEY_SECTOR = \"sector:{{focus_group_id}}\"\n    CACHE_KEY_ALL = \"sectors:all\"\n    CACHE_TTL = 3600 # 1 hour\n\n    async def get_config(self, focus_group_id: str) -> Optional[SectorConfig]:\n        cache_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)\n        cached = cache.get(cache_key)\n        if cached:\n            # Cache HIT\n            return SectorConfig._dict_to_config(cached)\n\n        # Cache MISS - load from database\n        config = await self._load_from_db(focus_group_id)\n        if config:\n            cache.set(cache_key, config._config_to_dict(), self.CACHE_TTL)\n        return config")
    st.markdown("```")

    st.markdown("")
    st.markdown("Furthermore, a mechanism for **cache invalidation** is crucial. When configuration data changes in the database, the corresponding cached entries must be removed or updated to prevent serving stale data. The `invalidate_cache` method handles this.")
    st.markdown("")

    if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
        st.warning("Please seed dimension weights and calibrations first in the 'Data Seeding' section to demonstrate caching.")
    else:
        st.subheader("Demonstration: Caching Behavior and Invalidation")
        st.markdown("")
        st.markdown("Select a sector and click 'Fetch Config (with cache)' multiple times to observe the caching. Then, try 'Invalidate Cache' and fetch again.")

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
            invalidate_fg_id = next((fg_id for fg_id, name in fg_names.items() if name == selected_invalidate_scope_name), None)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Invalidate Cache"):
                with st.spinner(f"Invalidating cache for {selected_invalidate_scope_name}..."):
                    invalidate_sector_cache_service_sync(invalidate_fg_id)
                    # Removed `st.session_state.sector_configs_cache = {}` as it's no longer used.
                    st.success("Cache invalidation triggered!")
        with col2:
            # Re-fetch from DB and set cache
            selected_fetch_fg_name = st.selectbox(
                "Select a Sector to Fetch (observe cache hits/misses):",
                options=list(fg_names.values()),
                index=0,
                key="cache_fetch_sector_select"
            )
            fetch_fg_id = next((fg_id for fg_id, name in fg_names.items() if name == selected_fetch_fg_name), None)
            
            if st.button("Fetch Config (with cache)", key="fetch_config_button"):
                with st.spinner(f"Fetching config for {selected_fetch_fg_name}..."):
                    fetched_config = get_sector_config_from_service_sync(fetch_fg_id)
                    if fetched_config:
                        st.success(f"Successfully fetched config for {fetched_config.group_name}.")
                        st.json(fetched_config._config_to_dict()) # Display raw data from source.py function

# Page: 2.8: Unified Organization View
elif st.session_state.current_page == "2.8: Unified Organization View":
    st.title("Task 2.8: Create the Unified Organization View")
    st.markdown("")
    st.markdown("The ultimate goal of our configuration-driven architecture is to provide a unified, easily queryable view of organizations, enriching their core data with sector-specific attributes dynamically.")
    st.markdown("")
    st.markdown("This is achieved by creating a database VIEW (`vw_organizations_full`) that joins the main `organizations` table with the respective `org_attributes_` tables based on the `focus_group_id` reference.")
    st.markdown("")
    st.markdown("Here's a simplified representation of the SQL VIEW definition:")
    st.markdown("```sql")
    st.markdown("CREATE OR REPLACE VIEW vw_organizations_full AS\nSELECT\n    o.*,\n    fg.group_name AS sector_name,\n    fg.group_code AS sector_code,\n    -- Manufacturing specific attributes\n    mfg.plant_count, mfg.automation_level, mfg.digital_twin_status,\n    -- Financial Services specific attributes\n    fin.regulatory_bodies, fin.algo_trading, fin.aum_billions,\n    -- ... similar attributes for other sectors ...\nFROM organizations o\nJOIN focus_groups fg ON o.focus_group_id = fg.focus_group_id\nLEFT JOIN org_attributes_manufacturing mfg ON o.organization_id = mfg.organization_id\nLEFT JOIN org_attributes_financial_services fin ON o.organization_id = fin.organization_id\n-- ... left joins for all other org_attributes_ tables ...\nLEFT JOIN org_attributes_professional_services ps ON o.organization_id = ps.organization_id;")
    st.markdown("```")
    st.markdown("")
    st.markdown("This view allows us to query all organizations and their relevant sector attributes in a single statement, without needing to know which specific attribute table to query beforehand. The `LEFT JOIN` ensures that even organizations without specific attributes in a given table are still included.")
    st.markdown("")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database schema first in the 'Schema Design & Attributes' section.")
    elif not st.session_state.initial_data_seeded:
        st.warning("Please seed initial focus groups and dimensions first in the 'Data Seeding' section.")
    else:
        st.subheader("Action: Insert Sample Organizations")
        st.markdown("Let's populate our `organizations` table with some sample data, ensuring they are linked to different PE sectors.")
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
            st.markdown("")
            st.markdown("Now, let's query the `vw_organizations_full` view. You can filter by a specific sector or view all organizations.")

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



# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
