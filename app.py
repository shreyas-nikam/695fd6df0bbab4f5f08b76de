

import streamlit as st
import pandas as pd
import plotly.express as px
import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Any, Optional
import time  # For simulating delays

# --- Placeholder/Mock Implementations for 'source.py' content ---
# This section replaces 'from source import *'

# Mock data for focus groups
sample_focus_groups = [
    {'focus_group_id': 'pe_manufacturing', 'platform': 'pe_org_air', 'group_name': 'Manufacturing',
        'group_code': 'MFG', 'display_order': 1, 'icon_name': 'industry', 'color_hex': '#FF5733', 'is_active': True},
    {'focus_group_id': 'pe_financial_services', 'platform': 'pe_org_air', 'group_name': 'Financial Services',
        'group_code': 'FIN', 'display_order': 2, 'icon_name': 'bank', 'color_hex': '#3366FF', 'is_active': True},
    {'focus_group_id': 'pe_healthcare', 'platform': 'pe_org_air', 'group_name': 'Healthcare',
        'group_code': 'HC', 'display_order': 3, 'icon_name': 'health', 'color_hex': '#33FF57', 'is_active': True},
    {'focus_group_id': 'pe_technology', 'platform': 'pe_org_air', 'group_name': 'Technology',
        'group_code': 'TECH', 'display_order': 4, 'icon_name': 'laptop', 'color_hex': '#FF33EC', 'is_active': True},
    {'focus_group_id': 'pe_retail', 'platform': 'pe_org_air', 'group_name': 'Retail & Consumer', 'group_code': 'RTL',
        'display_order': 5, 'icon_name': 'shopping_cart', 'color_hex': '#FF8C33', 'is_active': True},
    {'focus_group_id': 'pe_energy', 'platform': 'pe_org_air', 'group_name': 'Energy & Utilities', 'group_code': 'ENR',
        'display_order': 6, 'icon_name': 'lightbulb', 'color_hex': '#33FFEE', 'is_active': True},
    {'focus_group_id': 'pe_professional_services', 'platform': 'pe_org_air', 'group_name': 'Professional Services',
        'group_code': 'PS', 'display_order': 7, 'icon_name': 'briefcase', 'color_hex': '#8C33FF', 'is_active': True},
]

# Mock data for dimensions (7 dimensions for PE Org-AI-R)
sample_dimensions = [
    {'dimension_id': 'pe_dim_data_infra', 'platform': 'pe_org_air',
        'dimension_name': 'Data Infrastructure', 'dimension_code': 'data_infrastructure', 'display_order': 1},
    {'dimension_id': 'pe_dim_governance', 'platform': 'pe_org_air',
        'dimension_name': 'AI Governance', 'dimension_code': 'ai_governance', 'display_order': 2},
    {'dimension_id': 'pe_dim_tech_stack', 'platform': 'pe_org_air',
        'dimension_name': 'Technology Stack', 'dimension_code': 'technology_stack', 'display_order': 3},
    {'dimension_id': 'pe_dim_talent', 'platform': 'pe_org_air',
        'dimension_name': 'Talent', 'dimension_code': 'talent', 'display_order': 4},
    {'dimension_id': 'pe_dim_leadership', 'platform': 'pe_org_air',
        'dimension_name': 'Leadership', 'dimension_code': 'leadership', 'display_order': 5},
    {'dimension_id': 'pe_dim_use_cases', 'platform': 'pe_org_air',
        'dimension_name': 'Use Case Portfolio', 'dimension_code': 'use_case_portfolio', 'display_order': 6},
    {'dimension_id': 'pe_dim_culture', 'platform': 'pe_org_air',
        'dimension_name': 'Culture', 'dimension_code': 'culture', 'display_order': 7},
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
            dimension_weights={k: Decimal(
                v) for k, v in data["dimension_weights"].items()},
            calibrations={k: Decimal(v)
                          for k, v in data["calibrations"].items()},
        )

# Mock Redis Cache


class MockRedisCache:
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        # Return cached value without displaying message
        # Messages will only show for SET and DELETE operations
        return self._cache.get(key, None)

    def set(self, key: str, value: Dict[str, Any], ttl: int):
        is_new = key not in self._cache
        self._cache[key] = value
        if is_new:
            st.success(f"✅ Cache SET (new entry): `{key}`")
        else:
            st.info(f"🔄 Cache SET (updated): `{key}`")

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
    time.sleep(1)  # Simulate DB operation
    # In a real app, this would execute DDL statements.
    # For this mock, we just return success.
    return True


def seed_initial_data() -> bool:
    time.sleep(1)  # Simulate data seeding
    # This would populate focus_groups and dimensions tables
    return True


def get_all_focus_groups() -> List[Dict[str, Any]]:
    # In a real app, this would query the focus_groups table
    return sample_focus_groups


def seed_dimension_weights_for_all_sectors() -> bool:
    time.sleep(1)  # Simulate data seeding
    # This would populate focus_group_dimension_weights
    return True


def seed_calibrations_for_all_sectors() -> bool:
    time.sleep(1)  # Simulate data seeding
    # This would populate focus_group_calibrations
    return True


def get_sector_config_from_service_sync(focus_group_id: str) -> Optional[SectorConfig]:
    # Simulate caching logic
    cache_key = f"sector:{focus_group_id}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return SectorConfig._dict_to_config(cached_data)

    time.sleep(1)  # Simulate DB fetch delay
    # Mock data generation for config (similar to the logic in thought process)
    for fg in sample_focus_groups:
        if fg['focus_group_id'] == focus_group_id:
            if focus_group_id == 'pe_manufacturing':
                weights = {'data_infra': Decimal('0.22'), 'governance': Decimal('0.12'), 'tech_adoption': Decimal(
                    '0.20'), 'analytics_ml': Decimal('0.18'), 'talent': Decimal('0.15'), 'process_automation': Decimal('0.13')}
                calibrations = {'h_r_baseline': Decimal('72'), 'ebitda_multiplier': Decimal(
                    '0.90'), 'position_factor_delta': Decimal('0.05'), 'talent_concentration_threshold': Decimal('0.65')}
            elif focus_group_id == 'pe_financial_services':
                weights = {'data_infra': Decimal('0.16'), 'governance': Decimal('0.25'), 'tech_adoption': Decimal(
                    '0.15'), 'analytics_ml': Decimal('0.24'), 'talent': Decimal('0.12'), 'process_automation': Decimal('0.08')}
                calibrations = {'h_r_baseline': Decimal('82'), 'ebitda_multiplier': Decimal(
                    '0.80'), 'position_factor_delta': Decimal('0.07'), 'talent_concentration_threshold': Decimal('0.75')}
            else:  # Default for other sectors
                weights = {'data_infra': Decimal('0.18'), 'governance': Decimal('0.15'), 'tech_adoption': Decimal(
                    '0.19'), 'analytics_ml': Decimal('0.17'), 'talent': Decimal('0.16'), 'process_automation': Decimal('0.15')}
                calibrations = {'h_r_baseline': Decimal('75'), 'ebitda_multiplier': Decimal(
                    '0.85'), 'position_factor_delta': Decimal('0.06'), 'talent_concentration_threshold': Decimal('0.70')}

            # Ensure sum to 1.0 (for validation demo)
            current_sum = sum(weights.values())
            if current_sum != Decimal('1.0'):
                adjustment_factor = Decimal('1.0') / current_sum
                weights = {k: round(v * adjustment_factor, 3)
                           for k, v in weights.items()}
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
            cache.set(cache_key, config._config_to_dict(),
                      3600)  # Cache for 1 hour
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
    time.sleep(1)  # Simulate DB operation
    # This would insert data into organizations and org_attributes_* tables
    return True


def fetch_unified_organization_data_sync(sector_id: Optional[str]) -> pd.DataFrame:
    time.sleep(1)  # Simulate DB fetch delay
    data = []

    # Generate 14 organizations (2 per sector) to ensure good distribution
    for i in range(1, 15):
        org_fg = sample_focus_groups[(i - 1) % len(sample_focus_groups)]
        org_fg_id = org_fg['focus_group_id']
        org_fg_name = org_fg['group_name']
        org_fg_code = org_fg['group_code']

        if sector_id and org_fg_id != sector_id:
            continue

        row = {
            'organization_id': f'org-{i:03d}',
            'legal_name': f'{org_fg_name} Company {i}',
            'display_name': f'{org_fg_code} Org {i}',
            'ticker_symbol': f'{org_fg_code}{i}',
            'focus_group_id': org_fg_id,
            'sector_name': org_fg_name,
            'sector_code': org_fg_code,
            'employee_count': 1000 + i * 150,
            'annual_revenue_usd': 50_000_000 + i * 15_000_000,
            'headquarters_country': 'USA',
            'headquarters_state': ['CA', 'NY', 'TX', 'IL', 'MA'][i % 5],
            'headquarters_city': ['San Francisco', 'New York', 'Houston', 'Chicago', 'Boston'][i % 5],
            'website_url': f'http://www.{org_fg_code.lower()}org{i}.com',
            'status': 'active',
            'created_at': pd.Timestamp.now(),
            'updated_at': pd.Timestamp.now(),
            'created_by': 'system_admin',

            # Manufacturing attributes
            'plant_count': (3 + i % 5) if org_fg_id == 'pe_manufacturing' else None,
            'automation_level': ['High', 'Medium', 'Advanced'][i % 3] if org_fg_id == 'pe_manufacturing' else None,
            'digital_twin_status': ['Implemented', 'Pilot', 'Planned'][i % 3] if org_fg_id == 'pe_manufacturing' else None,

            # Financial Services attributes
            'algo_trading': (i % 2 == 0) if org_fg_id == 'pe_financial_services' else None,
            'aum_billions': round(25.0 + i * 5.5, 2) if org_fg_id == 'pe_financial_services' else None,
            'total_assets_billions': round(75.0 + i * 12.5, 2) if org_fg_id == 'pe_financial_services' else None,

            # Healthcare attributes
            'hipaa_certified': True if org_fg_id == 'pe_healthcare' else None,
            'ehr_system': ['Epic', 'Cerner', 'Meditech'][i % 3] if org_fg_id == 'pe_healthcare' else None,
            'clinical_ai_deployed': (i % 2 == 1) if org_fg_id == 'pe_healthcare' else None,

            # Technology attributes
            'github_stars_total': (1000 + i * 500) if org_fg_id == 'pe_technology' else None,
            'ml_platform': ['TensorFlow', 'PyTorch', 'JAX'][i % 3] if org_fg_id == 'pe_technology' else None,
            'ai_product_features': (5 + i * 2) if org_fg_id == 'pe_technology' else None,

            # Retail attributes
            'cdp_vendor': ['Segment', 'mParticle', 'Tealium'][i % 3] if org_fg_id == 'pe_retail' else None,
            'personalization_ai': (i % 2 == 0) if org_fg_id == 'pe_retail' else None,
            'store_count': (50 + i * 25) if org_fg_id == 'pe_retail' else None,

            # Energy attributes
            'smart_grid_pct': round(30.0 + i * 5.0, 2) if org_fg_id == 'pe_energy' else None,
            'predictive_maintenance': (i % 2 == 1) if org_fg_id == 'pe_energy' else None,
            'renewable_pct': round(15.0 + i * 3.5, 2) if org_fg_id == 'pe_energy' else None,

            # Professional Services attributes
            'firm_type': ['Consulting', 'Legal', 'Accounting'][i % 3] if org_fg_id == 'pe_professional_services' else None,
            'client_ai_services': (i % 2 == 0) if org_fg_id == 'pe_professional_services' else None,
            'document_ai': (i % 2 == 1) if org_fg_id == 'pe_professional_services' else None,
        }
        data.append(row)

    if data:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()

    return df

# --- End of Placeholder/Mock Implementations ---


st.set_page_config(
    page_title="QuLab: Unified Data Architecture & Caching", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Unified Data Architecture & Caching Lab")
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
        "2.1: Schema Design & Attributes",
        "2.2: Data Seeding",
        "2.3: Configuration Service",
        "2.4: Redis Caching",
        "2.5: Organization View"
    ]

    current_index = 0
    if st.session_state.current_page in page_options:
        current_index = page_options.index(st.session_state.current_page)

    st.session_state.current_page = st.selectbox(
        "",
        page_options,
        index=current_index
    )

    # Show Key Objectives and Tools Introduced only on Home page
    st.markdown("---")
    st.markdown("""**🎯Key Objectives**
- **Remember**: List the 7 PE sectors and their configuration parameters.
- **Understand**: Explain why configuration-driven architecture avoids schema proliferation.
- **Apply**: Implement focus group configuration loading with caching.
- **Analyze**: Compare sector attribute tables vs JSONB approaches.
- **Evaluate**: Assess dimension weight configurations for different sectors.
---
**🛠️ Tools Introduced**
- **PostgreSQL / Snowflake**: Our primary database, supporting both development and production environments.
- **SQLAlchemy 2.0**: An ORM layer for advanced database interactions.
- **Alembic**: For version-controlled schema migrations, ensuring smooth database evolution.
- **Redis**: A fast in-memory data store for caching, essential for high-performance configuration lookups.
- **structlog**: For structured logging, enhancing observability of our services.
""")

# Page: Home
if st.session_state.current_page == "Home":
    st.markdown(
        "Welcome to Week 2 of our journey into building a robust Private Equity (PE) Intelligence Platform!")

    st.markdown(
        "In this lab, you'll transition from foundational setup to designing a truly configuration-driven data architecture. This approach is crucial for managing the complexity of diverse PE sectors without resorting to schema proliferation.")

    st.markdown("---")
    st.subheader("Key Concepts")

    st.markdown(
        """The central theme for this week is **One Schema, Many Configurations**. This means:
- We avoid creating separate schemas for each PE sector.
- Differentiation between sectors is achieved through data rows in configuration tables, not schema variations.
- Focus Group Configuration Tables store weights and calibrations as data rows.
- Queryable Sector Attribute Tables use typed columns instead of less flexible JSONB approaches.
- Configuration Caching ensures that frequently accessed configurations are loaded once and used everywhere, reducing database load.""")

    st.markdown("---")
    st.subheader("""Prerequisites
- Week 1 completed (FastAPI, Pydantic settings)
- SQL proficiency (JOINs, views)
- Understanding of database normalization
""")

# Page: 2.1: Schema Design & Attributes
elif st.session_state.current_page == "2.1: Schema Design & Attributes":
    st.title("Task 2.1: Database Schema Design")

    st.markdown(
        "This section focuses on designing a flexible and extensible data architecture.")
    st.markdown("We'll define tables that allow for configuration-driven differentiation across PE sectors, avoiding the 'schema per sector' anti-pattern.")

    st.subheader("Design Principle: One Schema, Many Configurations")

    st.markdown("A core principle of our architecture is that all 7 PE sectors share identical base schemas. Sector-specific differentiation is achieved through configuration tables and dedicated attribute tables, rather than varying the base schema.")

    st.markdown("This approach minimizes `N×M` joins, prevents `NULL` proliferation in central tables, and allows for robust querying of sector-specific attributes using typed columns.")

    st.subheader("Focus Group Configuration Schema")

    st.markdown("We'll start by defining the `focus_groups` table to store our primary sectors, along with `dimensions`, `focus_group_dimension_weights`, and `focus_group_calibrations` to hold sector-specific configuration parameters.")

    st.info("📁 **File:** `migrations/versions/002_focus_group_schema.sql`")
    st.markdown("**PostgreSQL DDL (Schema Definition Only):**")
    st.markdown("""
```sql
-- ============================================
-- FOCUS GROUPS MASTER TABLE
-- ============================================
CREATE TABLE focus_groups (
    focus_group_id VARCHAR(50) PRIMARY KEY,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('pe_org_air', 'individual_air')),
    group_name VARCHAR(100) NOT NULL,
    group_code VARCHAR(30) NOT NULL,
    group_description TEXT,
    display_order INTEGER NOT NULL,
    icon_name VARCHAR(50),
    color_hex VARCHAR(7),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (platform, group_code)
);

-- ============================================
-- DIMENSIONS TABLE (PE Org-AI-R: 7 Dimensions)
-- ============================================
CREATE TABLE dimensions (
    dimension_id VARCHAR(50) PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    dimension_name VARCHAR(100) NOT NULL,
    dimension_code VARCHAR(50) NOT NULL,
    description TEXT,
    min_score DECIMAL(5,2) DEFAULT 0,
    max_score DECIMAL(5,2) DEFAULT 100,
    display_order INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (platform, dimension_code)
);

-- ============================================
-- FOCUS GROUP DIMENSION WEIGHTS
-- ============================================
CREATE TABLE focus_group_dimension_weights (
    weight_id SERIAL PRIMARY KEY,
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
    dimension_id VARCHAR(50) NOT NULL REFERENCES dimensions(dimension_id),
    weight DECIMAL(4,3) NOT NULL CHECK (weight >= 0 AND weight <= 1),
    weight_rationale TEXT,
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (focus_group_id, dimension_id, effective_from)
);

CREATE INDEX idx_weights_current ON focus_group_dimension_weights(focus_group_id, is_current)
    WHERE is_current = TRUE;

-- ============================================
-- FOCUS GROUP CALIBRATIONS
-- ============================================
CREATE TABLE focus_group_calibrations (
    calibration_id SERIAL PRIMARY KEY,
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value DECIMAL(10,4) NOT NULL,
    parameter_type VARCHAR(20) DEFAULT 'numeric',
    description TEXT,
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (focus_group_id, parameter_name, effective_from)
);
```
""")

    st.subheader("Organizations Table with Sector Reference")

    st.markdown("The `organizations` table will store core information about each company. Critically, it includes a `focus_group_id` as a foreign key to link each organization to its primary PE sector. This is the cornerstone of our configuration-driven approach.")

    st.info("📁 **File:** `migrations/versions/002d_organization_schema.sql`")
    st.markdown("""
```sql
CREATE TABLE organizations (
    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    legal_name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    ticker_symbol VARCHAR(10),
    cik_number VARCHAR(20),
    duns_number VARCHAR(20),
    
    -- Sector Assignment
    focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
    
    -- Industry Classification
    primary_sic_code VARCHAR(10),
    primary_naics_code VARCHAR(10),
    
    -- Firmographics
    employee_count INTEGER,
    annual_revenue_usd DECIMAL(15,2),
    founding_year INTEGER,
    headquarters_country VARCHAR(3),
    headquarters_state VARCHAR(50),
    headquarters_city VARCHAR(100),
    website_url VARCHAR(500),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    
    CONSTRAINT chk_org_pe_platform CHECK (focus_group_id LIKE 'pe_%')
);

CREATE INDEX idx_org_focus_group ON organizations(focus_group_id);
CREATE INDEX idx_org_ticker ON organizations(ticker_symbol) WHERE ticker_symbol IS NOT NULL;
```
""")

    st.subheader("Sector-Specific Attribute Tables")

    st.markdown("Instead of using JSONB columns or adding many nullable columns to the main `organizations` table, we create separate, strongly-typed attribute tables for each sector. This keeps our data structured and queryable.")

    st.info("📁 **File:** `migrations/versions/002e_sector_attributes.sql`")
    st.markdown("**All 7 Sector Attribute Tables:**")
    st.markdown("""
```sql
-- Manufacturing Sector Attributes
CREATE TABLE org_attributes_manufacturing (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    ot_systems VARCHAR(100)[],
    it_ot_integration VARCHAR(20),
    scada_vendor VARCHAR(100),
    mes_system VARCHAR(100),
    plant_count INTEGER,
    automation_level VARCHAR(20),
    iot_platforms VARCHAR(100)[],
    digital_twin_status VARCHAR(20),
    edge_computing BOOLEAN DEFAULT FALSE,
    supply_chain_visibility VARCHAR(20),
    demand_forecasting_ai BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial Services Sector Attributes
CREATE TABLE org_attributes_financial_services (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    regulatory_bodies VARCHAR(50)[],
    charter_type VARCHAR(50),
    model_risk_framework VARCHAR(50),
    mrm_team_size INTEGER,
    model_inventory_count INTEGER,
    algo_trading BOOLEAN DEFAULT FALSE,
    fraud_detection_ai BOOLEAN DEFAULT FALSE,
    credit_ai BOOLEAN DEFAULT FALSE,
    aml_ai BOOLEAN DEFAULT FALSE,
    aum_billions DECIMAL(12,2),
    total_assets_billions DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Healthcare Sector Attributes
CREATE TABLE org_attributes_healthcare (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    hipaa_certified BOOLEAN DEFAULT FALSE,
    hitrust_certified BOOLEAN DEFAULT FALSE,
    fda_clearances VARCHAR(100)[],
    fda_clearance_count INTEGER DEFAULT 0,
    ehr_system VARCHAR(100),
    ehr_integration_level VARCHAR(20),
    fhir_enabled BOOLEAN DEFAULT FALSE,
    clinical_ai_deployed BOOLEAN DEFAULT FALSE,
    imaging_ai BOOLEAN DEFAULT FALSE,
    org_type VARCHAR(50),
    bed_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Technology Sector Attributes
CREATE TABLE org_attributes_technology (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    tech_category VARCHAR(50),
    primary_language VARCHAR(50),
    cloud_native BOOLEAN DEFAULT TRUE,
    github_org VARCHAR(100),
    github_stars_total INTEGER,
    open_source_projects INTEGER,
    ml_platform VARCHAR(100),
    llm_integration BOOLEAN DEFAULT FALSE,
    ai_product_features INTEGER,
    gpu_infrastructure BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retail Sector Attributes
CREATE TABLE org_attributes_retail (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    retail_type VARCHAR(50),
    store_count INTEGER,
    ecommerce_pct DECIMAL(5,2),
    cdp_vendor VARCHAR(100),
    loyalty_program BOOLEAN DEFAULT FALSE,
    loyalty_members INTEGER,
    personalization_ai BOOLEAN DEFAULT FALSE,
    recommendation_engine VARCHAR(100),
    demand_forecasting BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Energy Sector Attributes
CREATE TABLE org_attributes_energy (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    energy_type VARCHAR(50),
    regulated BOOLEAN DEFAULT FALSE,
    scada_systems VARCHAR(100)[],
    ami_deployed BOOLEAN DEFAULT FALSE,
    smart_grid_pct DECIMAL(5,2),
    generation_capacity_mw DECIMAL(12,2),
    grid_optimization_ai BOOLEAN DEFAULT FALSE,
    predictive_maintenance BOOLEAN DEFAULT FALSE,
    renewable_pct DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Professional Services Sector Attributes
CREATE TABLE org_attributes_professional_services (
    organization_id UUID PRIMARY KEY REFERENCES organizations(organization_id),
    firm_type VARCHAR(50),
    partnership_model VARCHAR(50),
    partner_count INTEGER,
    professional_staff INTEGER,
    km_system VARCHAR(100),
    document_ai BOOLEAN DEFAULT FALSE,
    knowledge_graph BOOLEAN DEFAULT FALSE,
    client_ai_services BOOLEAN DEFAULT FALSE,
    internal_ai_tools BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
""")

    st.markdown("---")
    st.subheader("Action: Initialize Database Schema")
    st.markdown(
        "Click the button below to create all necessary tables for our configuration-driven data architecture.")
    if not st.session_state.db_initialized:
        if st.button("Initialize Database Schema"):
            with st.spinner("Creating tables..."):
                success = setup_database_schema()
                if success:
                    st.session_state.db_initialized = True
                    st.success("Database schema initialized successfully!")
                else:
                    st.error("Failed to initialize database schema.")
    else:
        st.info("Database schema is already initialized.")

# Page: 2.2: Data Seeding
elif st.session_state.current_page == "2.2: Data Seeding":
    st.title("Task 2.2: Seed Sector Configuration Data")

    st.markdown("With our schema in place, it's time to populate our configuration tables. This is where we define the unique characteristics for each PE sector as data rows, rather than schema changes.")

    if not st.session_state.db_initialized:
        st.warning(
            "Please initialize the database schema first in the '2.1: Schema Design & Attributes' section.")
    else:
        st.subheader("Step 1: Seed Initial Focus Groups and Dimensions")
        st.markdown(
            "First, we'll populate the base `focus_groups` and `dimensions` tables.")
        st.markdown("""
```sql
-- Seed PE Org-AI-R Sectors
INSERT INTO focus_groups (focus_group_id, platform, group_name, group_code, display_order) VALUES
    ('pe_manufacturing', 'pe_org_air', 'Manufacturing', 'MFG', 1),
    ('pe_financial_services', 'pe_org_air', 'Financial Services', 'FIN', 2),
    ('pe_healthcare', 'pe_org_air', 'Healthcare', 'HC', 3),
    ('pe_technology', 'pe_org_air', 'Technology', 'TECH', 4),
    ('pe_retail', 'pe_org_air', 'Retail & Consumer', 'RTL', 5),
    ('pe_energy', 'pe_org_air', 'Energy & Utilities', 'ENR', 6),
    ('pe_professional_services', 'pe_org_air', 'Professional Services', 'PS', 7);

-- Seed Dimensions
INSERT INTO dimensions (dimension_id, platform, dimension_name, dimension_code, display_order) VALUES
    ('pe_dim_data_infra', 'pe_org_air', 'Data Infrastructure', 'data_infrastructure', 1),
    ('pe_dim_governance', 'pe_org_air', 'AI Governance', 'ai_governance', 2),
    ('pe_dim_tech_stack', 'pe_org_air', 'Technology Stack', 'technology_stack', 3),
    ('pe_dim_talent', 'pe_org_air', 'Talent', 'talent', 4),
    ('pe_dim_leadership', 'pe_org_air', 'Leadership', 'leadership', 5),
    ('pe_dim_use_cases', 'pe_org_air', 'Use Case Portfolio', 'use_case_portfolio', 6),
    ('pe_dim_culture', 'pe_org_air', 'Culture', 'culture', 7);
    
```
        """)

        if not st.session_state.initial_data_seeded:
            if st.button("Seed Initial Focus Groups & Dimensions"):
                with st.spinner("Seeding initial data..."):
                    success = seed_initial_data()
                    if success:
                        st.session_state.initial_data_seeded = True
                        # Populate for future selects
                        st.session_state.all_focus_groups = get_all_focus_groups()
                        st.success(
                            "Initial focus groups and dimensions seeded!")
                    else:
                        st.error("Failed to seed initial data.")
        else:
            st.info("Initial focus groups and dimensions already seeded.")

            st.markdown("**Available Focus Groups:**")
            if st.session_state.all_focus_groups:
                st.dataframe(pd.DataFrame(st.session_state.all_focus_groups))

        if st.session_state.initial_data_seeded:
            st.markdown("---")
            st.subheader("Step 2: Seed Sector Dimension Weights")

            st.markdown("Dimension weights are critical for our scoring models, indicating the relative importance of different AI/data dimensions for each sector. These are stored in `focus_group_dimension_weights`.")

            st.info(
                "📁 **File:** `migrations/versions/002b_seed_dimension_weights.sql`")
            st.markdown("**Full INSERT statements for all 7 sectors:**")
            st.markdown("""
```sql                
-- Manufacturing (emphasis: data infra, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_manufacturing', 'pe_dim_data_infra', 0.22, 'OT/IT integration critical'),
('pe_manufacturing', 'pe_dim_governance', 0.12, 'Less regulatory than finance/health'),
('pe_manufacturing', 'pe_dim_tech_stack', 0.18, 'Edge computing, IoT platforms'),
('pe_manufacturing', 'pe_dim_talent', 0.15, 'AI + manufacturing expertise scarce'),
('pe_manufacturing', 'pe_dim_leadership', 0.12, 'Traditional leadership acceptable'),
('pe_manufacturing', 'pe_dim_use_cases', 0.14, 'Clear ROI in operations'),
('pe_manufacturing', 'pe_dim_culture', 0.07, 'Safety culture > innovation');

-- Financial Services (emphasis: governance, talent)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_financial_services', 'pe_dim_data_infra', 0.16, 'Mature infrastructure'),
('pe_financial_services', 'pe_dim_governance', 0.22, 'Regulatory imperative'),
('pe_financial_services', 'pe_dim_tech_stack', 0.14, 'Standard cloud stacks'),
('pe_financial_services', 'pe_dim_talent', 0.18, 'Quant + ML talent critical'),
('pe_financial_services', 'pe_dim_leadership', 0.12, 'C-suite AI awareness high'),
('pe_financial_services', 'pe_dim_use_cases', 0.10, 'Well-understood use cases'),
('pe_financial_services', 'pe_dim_culture', 0.08, 'Risk-averse by design');

-- Healthcare (emphasis: governance, data, leadership)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_healthcare', 'pe_dim_data_infra', 0.20, 'EHR integration critical'),
('pe_healthcare', 'pe_dim_governance', 0.20, 'FDA/HIPAA compliance'),
('pe_healthcare', 'pe_dim_tech_stack', 0.14, 'EHR-centric ecosystems'),
('pe_healthcare', 'pe_dim_talent', 0.15, 'Clinical + AI dual expertise'),
('pe_healthcare', 'pe_dim_leadership', 0.15, 'Physician champions matter'),
('pe_healthcare', 'pe_dim_use_cases', 0.10, 'Long validation cycles'),
('pe_healthcare', 'pe_dim_culture', 0.06, 'Evidence-based culture exists');

-- Technology (emphasis: talent, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_technology', 'pe_dim_data_infra', 0.15, 'Assumed competent'),
('pe_technology', 'pe_dim_governance', 0.12, 'Less regulated'),
('pe_technology', 'pe_dim_tech_stack', 0.18, 'Core differentiator'),
('pe_technology', 'pe_dim_talent', 0.22, 'Talent is everything'),
('pe_technology', 'pe_dim_leadership', 0.13, 'Tech-savvy by default'),
('pe_technology', 'pe_dim_use_cases', 0.15, 'Product innovation'),
('pe_technology', 'pe_dim_culture', 0.05, 'Innovation assumed');

-- Retail & Consumer (emphasis: data, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_retail', 'pe_dim_data_infra', 0.20, 'Customer data platforms'),
('pe_retail', 'pe_dim_governance', 0.12, 'Privacy focus, less regulated'),
('pe_retail', 'pe_dim_tech_stack', 0.15, 'Standard cloud + CDP'),
('pe_retail', 'pe_dim_talent', 0.15, 'Data science accessible'),
('pe_retail', 'pe_dim_leadership', 0.13, 'Digital transformation focus'),
('pe_retail', 'pe_dim_use_cases', 0.18, 'Clear revenue impact'),
('pe_retail', 'pe_dim_culture', 0.07, 'Customer-centric exists');

-- Energy & Utilities (emphasis: data, tech stack, use cases)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_energy', 'pe_dim_data_infra', 0.22, 'SCADA/OT data critical'),
('pe_energy', 'pe_dim_governance', 0.15, 'Regulatory + safety'),
('pe_energy', 'pe_dim_tech_stack', 0.18, 'Grid tech, edge computing'),
('pe_energy', 'pe_dim_talent', 0.12, 'Talent scarcity'),
('pe_energy', 'pe_dim_leadership', 0.13, 'Traditional but evolving'),
('pe_energy', 'pe_dim_use_cases', 0.15, 'Clear operational value'),
('pe_energy', 'pe_dim_culture', 0.05, 'Safety culture paramount');

-- Professional Services (emphasis: talent, leadership)
INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight,
weight_rationale) VALUES
('pe_professional_services', 'pe_dim_data_infra', 0.14, 'Document-centric'),
('pe_professional_services', 'pe_dim_governance', 0.15, 'Client confidentiality'),
('pe_professional_services', 'pe_dim_tech_stack', 0.12, 'Standard productivity'),
('pe_professional_services', 'pe_dim_talent', 0.22, 'People are the product'),
('pe_professional_services', 'pe_dim_leadership', 0.17, 'Partner adoption critical'),
('pe_professional_services', 'pe_dim_use_cases', 0.12, 'Client + internal'),
('pe_professional_services', 'pe_dim_culture', 0.08, 'Innovation varies');
""")

            st.subheader("Step 3: Seed Sector Calibrations")

            st.markdown("Sector calibrations hold specific numeric or categorical parameters unique to each sector, like 'H&R Baseline' or 'EBITDA Multiplier'. These are stored in `focus_group_calibrations`.")

            st.info("📁 **File:** `migrations/versions/002c_seed_calibrations.sql`")
            st.markdown("**Full INSERT statements for all 7 sectors:**")
            st.markdown("""```sql
-- migrations/versions/002c_seed_calibrations.sql
INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value,
parameter_type, description) VALUES
-- Manufacturing
('pe_manufacturing', 'h_r_baseline', 72, 'numeric', 'Systematic opportunity baseline'),
('pe_manufacturing', 'ebitda_multiplier', 0.90, 'numeric', 'Conservative EBITDA attribution'),
('pe_manufacturing', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to talent scarcity'),
('pe_manufacturing', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Financial Services
('pe_financial_services', 'h_r_baseline', 82, 'numeric', 'Higher due to data maturity'),
('pe_financial_services', 'ebitda_multiplier', 1.10, 'numeric', 'Higher AI leverage'),
('pe_financial_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_financial_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
('pe_financial_services', 'governance_minimum', 60, 'threshold', 'Min governance for approval'),
-- Healthcare
('pe_healthcare', 'h_r_baseline', 78, 'numeric', 'Moderate with growth potential'),
('pe_healthcare', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
('pe_healthcare', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_healthcare', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
('pe_healthcare', 'governance_minimum', 65, 'threshold', 'Higher governance requirement'),
-- Technology
('pe_technology', 'h_r_baseline', 85, 'numeric', 'Highest - AI native'),
('pe_technology', 'ebitda_multiplier', 1.15, 'numeric', 'Strong AI leverage'),
('pe_technology', 'talent_concentration_threshold', 0.30, 'threshold', 'Higher talent expected'),
('pe_technology', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Retail
('pe_retail', 'h_r_baseline', 75, 'numeric', 'Growing AI adoption'),
('pe_retail', 'ebitda_multiplier', 1.05, 'numeric', 'Clear personalization ROI'),
('pe_retail', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_retail', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Energy
('pe_energy', 'h_r_baseline', 68, 'numeric', 'Lower but high potential'),
('pe_energy', 'ebitda_multiplier', 0.85, 'numeric', 'Longer payback periods'),
('pe_energy', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to scarcity'),
('pe_energy', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
-- Professional Services
('pe_professional_services', 'h_r_baseline', 76, 'numeric', 'Knowledge work automation'),
('pe_professional_services', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
('pe_professional_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
('pe_professional_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment');
```                         
                        
                        """)

            st.markdown("---")
            st.subheader("Action: Seed All Dimension Weights and Calibrations")
            st.markdown(
                "Click the button below to seed all dimension weights and calibration parameters for all 7 PE sectors.")
            if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
                if st.button("Seed All Weights and Calibrations"):
                    with st.spinner("Seeding dimension weights..."):
                        weights_success = seed_dimension_weights_for_all_sectors()
                        if weights_success:
                            st.session_state.weights_seeded = True
                            st.success(
                                "Dimension weights seeded successfully!")
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
                        st.success(
                            "All dimension weights and calibrations seeded!")

                    # Display all seeded data in organized tables
                    st.markdown("---")
                    st.subheader("📊 Seeded Data Overview")

                    # Tab layout for better organization
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["Focus Groups", "Dimensions", "Dimension Weights", "Calibrations"])

                    with tab1:
                        st.markdown("**All PE Org-AI-R Sectors:**")
                        fg_df = pd.DataFrame(st.session_state.all_focus_groups)
                        st.dataframe(fg_df, use_container_width=True,
                                     hide_index=True)

                    with tab2:
                        st.markdown("**All 7 Dimensions for PE Org-AI-R:**")
                        dim_df = pd.DataFrame(sample_dimensions)
                        st.dataframe(dim_df, use_container_width=True,
                                     hide_index=True)

                    with tab3:
                        st.markdown("**Dimension Weights for All 7 Sectors:**")
                        st.caption(
                            "Shows the relative importance of each dimension per sector (should sum to 1.0)")

                        # Create comprehensive weights data
                        weights_data = []
                        for fg in st.session_state.all_focus_groups:
                            fg_id = fg['focus_group_id']
                            fg_name = fg['group_name']

                            # Mock weight data based on the SQL we showed
                            if fg_id == 'pe_manufacturing':
                                weights = {'data_infrastructure': 0.22, 'ai_governance': 0.12, 'technology_stack': 0.18,
                                           'talent': 0.15, 'leadership': 0.12, 'use_case_portfolio': 0.14, 'culture': 0.07}
                            elif fg_id == 'pe_financial_services':
                                weights = {'data_infrastructure': 0.16, 'ai_governance': 0.22, 'technology_stack': 0.14,
                                           'talent': 0.18, 'leadership': 0.12, 'use_case_portfolio': 0.10, 'culture': 0.08}
                            elif fg_id == 'pe_healthcare':
                                weights = {'data_infrastructure': 0.20, 'ai_governance': 0.20, 'technology_stack': 0.14,
                                           'talent': 0.15, 'leadership': 0.15, 'use_case_portfolio': 0.10, 'culture': 0.06}
                            elif fg_id == 'pe_technology':
                                weights = {'data_infrastructure': 0.15, 'ai_governance': 0.12, 'technology_stack': 0.18,
                                           'talent': 0.22, 'leadership': 0.13, 'use_case_portfolio': 0.15, 'culture': 0.05}
                            elif fg_id == 'pe_retail':
                                weights = {'data_infrastructure': 0.20, 'ai_governance': 0.12, 'technology_stack': 0.15,
                                           'talent': 0.15, 'leadership': 0.13, 'use_case_portfolio': 0.18, 'culture': 0.07}
                            elif fg_id == 'pe_energy':
                                weights = {'data_infrastructure': 0.22, 'ai_governance': 0.15, 'technology_stack': 0.18,
                                           'talent': 0.12, 'leadership': 0.13, 'use_case_portfolio': 0.15, 'culture': 0.05}
                            else:  # pe_professional_services
                                weights = {'data_infrastructure': 0.14, 'ai_governance': 0.15, 'technology_stack': 0.12,
                                           'talent': 0.22, 'leadership': 0.17, 'use_case_portfolio': 0.12, 'culture': 0.08}

                            for dim_code, weight in weights.items():
                                weights_data.append({
                                    'Sector': fg_name,
                                    'Dimension': dim_code.replace('_', ' ').title(),
                                    'Weight': weight
                                })

                        weights_df = pd.DataFrame(weights_data)

                        # Pivot table for better visualization
                        pivot_weights = weights_df.pivot(
                            index='Sector', columns='Dimension', values='Weight')
                        st.dataframe(pivot_weights, use_container_width=True)

                        # Show weight sums for validation
                        st.markdown("**Weight Sums (Validation):**")
                        weight_sums = pivot_weights.sum(axis=1)
                        sum_df = pd.DataFrame(
                            {'Sector': weight_sums.index, 'Total Weight': weight_sums.values})
                        sum_df['Valid'] = sum_df['Total Weight'].apply(
                            lambda x: '✅' if abs(x - 1.0) < 0.001 else '❌')
                        st.dataframe(sum_df, use_container_width=True,
                                     hide_index=True)

                    with tab4:
                        st.markdown(
                            "**Calibration Parameters for All 7 Sectors:**")
                        st.caption(
                            "Sector-specific numeric parameters for H^R calculations and business metrics")

                        # Create comprehensive calibrations data
                        calib_data = []
                        calib_params = {
                            'pe_manufacturing': {'h_r_baseline': 72, 'ebitda_multiplier': 0.90,
                                                 'talent_concentration_threshold': 0.20, 'position_factor_delta': 0.15},
                            'pe_financial_services': {'h_r_baseline': 82, 'ebitda_multiplier': 1.10,
                                                      'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15,
                                                      'governance_minimum': 60},
                            'pe_healthcare': {'h_r_baseline': 78, 'ebitda_multiplier': 1.00,
                                              'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15,
                                              'governance_minimum': 65},
                            'pe_technology': {'h_r_baseline': 85, 'ebitda_multiplier': 1.15,
                                              'talent_concentration_threshold': 0.30, 'position_factor_delta': 0.15},
                            'pe_retail': {'h_r_baseline': 75, 'ebitda_multiplier': 1.05,
                                          'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15},
                            'pe_energy': {'h_r_baseline': 68, 'ebitda_multiplier': 0.85,
                                          'talent_concentration_threshold': 0.20, 'position_factor_delta': 0.15},
                            'pe_professional_services': {'h_r_baseline': 76, 'ebitda_multiplier': 1.00,
                                                         'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15}
                        }

                        for fg in st.session_state.all_focus_groups:
                            fg_id = fg['focus_group_id']
                            fg_name = fg['group_name']
                            params = calib_params.get(fg_id, {})

                            for param_name, param_value in params.items():
                                calib_data.append({
                                    'Sector': fg_name,
                                    'Parameter': param_name.replace('_', ' ').title(),
                                    'Value': param_value
                                })

                        calib_df = pd.DataFrame(calib_data)

                        # Pivot for better view
                        pivot_calib = calib_df.pivot(
                            index='Sector', columns='Parameter', values='Value')
                        st.dataframe(pivot_calib, use_container_width=True)
            else:
                st.info("All dimension weights and calibrations are already seeded.")
                # Display all seeded data in organized tables
                st.markdown("---")
                st.subheader("📊 Seeded Data Overview")

                # Tab layout for better organization
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["Focus Groups", "Dimensions", "Dimension Weights", "Calibrations"])

                with tab1:
                    st.markdown("**All PE Org-AI-R Sectors:**")
                    fg_df = pd.DataFrame(st.session_state.all_focus_groups)
                    st.dataframe(fg_df, use_container_width=True,
                                 hide_index=True)

                with tab2:
                    st.markdown("**All 7 Dimensions for PE Org-AI-R:**")
                    dim_df = pd.DataFrame(sample_dimensions)
                    st.dataframe(dim_df, use_container_width=True,
                                 hide_index=True)

                with tab3:
                    st.markdown("**Dimension Weights for All 7 Sectors:**")
                    st.caption(
                        "Shows the relative importance of each dimension per sector (should sum to 1.0)")

                    # Create comprehensive weights data
                    weights_data = []
                    for fg in st.session_state.all_focus_groups:
                        fg_id = fg['focus_group_id']
                        fg_name = fg['group_name']

                        # Mock weight data based on the SQL we showed
                        if fg_id == 'pe_manufacturing':
                            weights = {'data_infrastructure': 0.22, 'ai_governance': 0.12, 'technology_stack': 0.18,
                                       'talent': 0.15, 'leadership': 0.12, 'use_case_portfolio': 0.14, 'culture': 0.07}
                        elif fg_id == 'pe_financial_services':
                            weights = {'data_infrastructure': 0.16, 'ai_governance': 0.22, 'technology_stack': 0.14,
                                       'talent': 0.18, 'leadership': 0.12, 'use_case_portfolio': 0.10, 'culture': 0.08}
                        elif fg_id == 'pe_healthcare':
                            weights = {'data_infrastructure': 0.20, 'ai_governance': 0.20, 'technology_stack': 0.14,
                                       'talent': 0.15, 'leadership': 0.15, 'use_case_portfolio': 0.10, 'culture': 0.06}
                        elif fg_id == 'pe_technology':
                            weights = {'data_infrastructure': 0.15, 'ai_governance': 0.12, 'technology_stack': 0.18,
                                       'talent': 0.22, 'leadership': 0.13, 'use_case_portfolio': 0.15, 'culture': 0.05}
                        elif fg_id == 'pe_retail':
                            weights = {'data_infrastructure': 0.20, 'ai_governance': 0.12, 'technology_stack': 0.15,
                                       'talent': 0.15, 'leadership': 0.13, 'use_case_portfolio': 0.18, 'culture': 0.07}
                        elif fg_id == 'pe_energy':
                            weights = {'data_infrastructure': 0.22, 'ai_governance': 0.15, 'technology_stack': 0.18,
                                       'talent': 0.12, 'leadership': 0.13, 'use_case_portfolio': 0.15, 'culture': 0.05}
                        else:  # pe_professional_services
                            weights = {'data_infrastructure': 0.14, 'ai_governance': 0.15, 'technology_stack': 0.12,
                                       'talent': 0.22, 'leadership': 0.17, 'use_case_portfolio': 0.12, 'culture': 0.08}

                        for dim_code, weight in weights.items():
                            weights_data.append({
                                'Sector': fg_name,
                                'Dimension': dim_code.replace('_', ' ').title(),
                                'Weight': weight
                            })

                    weights_df = pd.DataFrame(weights_data)

                    # Pivot table for better visualization
                    pivot_weights = weights_df.pivot(
                        index='Sector', columns='Dimension', values='Weight')
                    st.dataframe(pivot_weights, use_container_width=True)

                    # Show weight sums for validation
                    st.markdown("**Weight Sums (Validation):**")
                    weight_sums = pivot_weights.sum(axis=1)
                    sum_df = pd.DataFrame(
                        {'Sector': weight_sums.index, 'Total Weight': weight_sums.values})
                    sum_df['Valid'] = sum_df['Total Weight'].apply(
                        lambda x: '✅' if abs(x - 1.0) < 0.001 else '❌')
                    st.dataframe(sum_df, use_container_width=True,
                                 hide_index=True)

                with tab4:
                    st.markdown(
                        "**Calibration Parameters for All 7 Sectors:**")
                    st.caption(
                        "Sector-specific numeric parameters for H^R calculations and business metrics")

                    # Create comprehensive calibrations data
                    calib_data = []
                    calib_params = {
                        'pe_manufacturing': {'h_r_baseline': 72, 'ebitda_multiplier': 0.90,
                                             'talent_concentration_threshold': 0.20, 'position_factor_delta': 0.15},
                        'pe_financial_services': {'h_r_baseline': 82, 'ebitda_multiplier': 1.10,
                                                  'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15,
                                                  'governance_minimum': 60},
                        'pe_healthcare': {'h_r_baseline': 78, 'ebitda_multiplier': 1.00,
                                          'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15,
                                          'governance_minimum': 65},
                        'pe_technology': {'h_r_baseline': 85, 'ebitda_multiplier': 1.15,
                                          'talent_concentration_threshold': 0.30, 'position_factor_delta': 0.15},
                        'pe_retail': {'h_r_baseline': 75, 'ebitda_multiplier': 1.05,
                                      'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15},
                        'pe_energy': {'h_r_baseline': 68, 'ebitda_multiplier': 0.85,
                                      'talent_concentration_threshold': 0.20, 'position_factor_delta': 0.15},
                        'pe_professional_services': {'h_r_baseline': 76, 'ebitda_multiplier': 1.00,
                                                     'talent_concentration_threshold': 0.25, 'position_factor_delta': 0.15}
                    }

                    for fg in st.session_state.all_focus_groups:
                        fg_id = fg['focus_group_id']
                        fg_name = fg['group_name']
                        params = calib_params.get(fg_id, {})

                        for param_name, param_value in params.items():
                            calib_data.append({
                                'Sector': fg_name,
                                'Parameter': param_name.replace('_', ' ').title(),
                                'Value': param_value
                            })

                    calib_df = pd.DataFrame(calib_data)

                    # Pivot for better view
                    pivot_calib = calib_df.pivot(
                        index='Sector', columns='Parameter', values='Value')
                    st.dataframe(pivot_calib, use_container_width=True)


# Page: 2.3: Configuration Service
elif st.session_state.current_page == "2.3: Configuration Service":
    st.title("Task 2.3: Build the Sector Configuration Service")

    st.markdown("Now that our configuration data is seeded, we need a service to efficiently retrieve these settings for specific sectors. This service encapsulates the logic for loading all relevant configuration parameters (weights, calibrations) into a single, easy-to-use object.")

    st.markdown("The `SectorConfig` dataclass and `SectorConfigService` class are designed for this purpose. The `SectorConfig` object provides convenient properties and methods to access sector-specific parameters.")

    st.info("📁 **File:** `src/pe_orgair/services/sector_config.py`")
    st.markdown("**Complete Sector Configuration Service Implementation:**")
    st.code("""# src/pe_orgair/services/sector_config.py
\"\"\"Sector configuration service with caching.\"\"\"
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from decimal import Decimal
import structlog

from pe_orgair.db.snowflake import db
from pe_orgair.infrastructure.cache import cache

logger = structlog.get_logger()

@dataclass
class SectorConfig:
    \"\"\"Complete configuration for a PE sector.\"\"\"
    focus_group_id: str
    group_name: str
    group_code: str
    dimension_weights: Dict[str, Decimal] = field(default_factory=dict)
    calibrations: Dict[str, Decimal] = field(default_factory=dict)

    @property
    def h_r_baseline(self) -> Decimal:
        \"\"\"Get H^R baseline for this sector.\"\"\"
        return self.calibrations.get('h_r_baseline', Decimal('75'))

    @property
    def ebitda_multiplier(self) -> Decimal:
        \"\"\"Get EBITDA multiplier for this sector.\"\"\"
        return self.calibrations.get('ebitda_multiplier', Decimal('1.0'))

    @property
    def position_factor_delta(self) -> Decimal:
        \"\"\"Get position factor delta (δ) for H^R calculation.\"\"\"
        return self.calibrations.get('position_factor_delta', Decimal('0.15'))

    @property
    def talent_concentration_threshold(self) -> Decimal:
        \"\"\"Get talent concentration threshold.\"\"\"
        return self.calibrations.get('talent_concentration_threshold', Decimal('0.25'))

    def get_dimension_weight(self, dimension_code: str) -> Decimal:
        \"\"\"Get weight for a specific dimension.\"\"\"
        return self.dimension_weights.get(dimension_code, Decimal('0'))

    def validate_weights_sum(self) -> bool:
        \"\"\"Verify dimension weights sum to 1.0.\"\"\"
        total = sum(self.dimension_weights.values())
        return abs(total - Decimal('1.0')) < Decimal('0.001')


class SectorConfigService:
    \"\"\"Service for loading and caching sector configurations.\"\"\"
    
    CACHE_KEY_SECTOR = "sector:{focus_group_id}"
    CACHE_KEY_ALL = "sectors:all"
    CACHE_TTL = 3600  # 1 hour

    async def get_config(self, focus_group_id: str) -> Optional[SectorConfig]:
        \"\"\"Get configuration for a single sector.\"\"\"
        cache_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)
        
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            return self._dict_to_config(cached)
        
        # Load from database
        config = await self._load_from_db(focus_group_id)
        if config:
            cache.set(cache_key, self._config_to_dict(config), self.CACHE_TTL)
        return config

    async def get_all_configs(self) -> List[SectorConfig]:
        \"\"\"Get all PE sector configurations.\"\"\"
        cache_key = self.CACHE_KEY_ALL
        cached = cache.get(cache_key)
        if cached:
            return [self._dict_to_config(c) for c in cached]
        
        configs = await self._load_all_from_db()
        cache.set(cache_key, [self._config_to_dict(c) for c in configs], self.CACHE_TTL)
        return configs

    async def _load_from_db(self, focus_group_id: str) -> Optional[SectorConfig]:
        \"\"\"Load single configuration from database.\"\"\"
        # Get base focus group
        fg_query = \"\"\"
            SELECT focus_group_id, group_name, group_code
            FROM focus_groups
            WHERE focus_group_id = %(focus_group_id)s
            AND platform = 'pe_org_air'
            AND is_active = TRUE
        \"\"\"
        fg_row = db.fetch_one(fg_query, {'focus_group_id': focus_group_id})
        if not fg_row:
            return None
        
        # Get dimension weights
        weights_query = \"\"\"
            SELECT d.dimension_code, w.weight
            FROM focus_group_dimension_weights w
            JOIN dimensions d ON w.dimension_id = d.dimension_id
            WHERE w.focus_group_id = %(focus_group_id)s AND w.is_current = TRUE
            ORDER BY d.display_order
        \"\"\"
        weights_rows = db.fetch_all(weights_query, {'focus_group_id': focus_group_id})
        dimension_weights = {
            row['dimension_code']: Decimal(str(row['weight']))
            for row in weights_rows
        }
        
        # Get calibrations
        calib_query = \"\"\"
            SELECT parameter_name, parameter_value
            FROM focus_group_calibrations
            WHERE focus_group_id = %(focus_group_id)s AND is_current = TRUE
        \"\"\"
        calib_rows = db.fetch_all(calib_query, {'focus_group_id': focus_group_id})
        calibrations = {
            row['parameter_name']: Decimal(str(row['parameter_value']))
            for row in calib_rows
        }
        
        config = SectorConfig(
            focus_group_id=fg_row['focus_group_id'],
            group_name=fg_row['group_name'],
            group_code=fg_row['group_code'],
            dimension_weights=dimension_weights,
            calibrations=calibrations,
        )
        
        if not config.validate_weights_sum():
            logger.warning("invalid_weights_sum", focus_group_id=focus_group_id)
        
        return config

    async def _load_all_from_db(self) -> List[SectorConfig]:
        \"\"\"Load all sector configurations from database.\"\"\"
        fg_query = \"\"\"
            SELECT focus_group_id
            FROM focus_groups
            WHERE platform = 'pe_org_air' AND is_active = TRUE
            ORDER BY display_order
        \"\"\"
        fg_rows = db.fetch_all(fg_query)
        
        configs = []
        for row in fg_rows:
            config = await self._load_from_db(row['focus_group_id'])
            if config:
                configs.append(config)
        return configs

    def _config_to_dict(self, config: SectorConfig) -> dict:
        \"\"\"Convert config to dict for caching.\"\"\"
        return {
            'focus_group_id': config.focus_group_id,
            'group_name': config.group_name,
            'group_code': config.group_code,
            'dimension_weights': {k: str(v) for k, v in config.dimension_weights.items()},
            'calibrations': {k: str(v) for k, v in config.calibrations.items()},
        }

    def _dict_to_config(self, data: dict) -> SectorConfig:
        \"\"\"Convert cached dict to config.\"\"\"
        return SectorConfig(
            focus_group_id=data['focus_group_id'],
            group_name=data['group_name'],
            group_code=data['group_code'],
            dimension_weights={k: Decimal(v) for k, v in data['dimension_weights'].items()},
            calibrations={k: Decimal(v) for k, v in data['calibrations'].items()},
        )

    def invalidate_cache(self, focus_group_id: Optional[str] = None) -> None:
        \"\"\"Invalidate cached configurations.\"\"\"
        if focus_group_id:
            cache.delete(self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id))
        cache.invalidate_pattern("sectors:*")
        logger.info("sector_cache_invalidated", focus_group_id=focus_group_id)


# Singleton instance
sector_service = SectorConfigService()
""", language="python")

    st.markdown("""
    **Key Features of the Sector Configuration Service:**
    - ✅ **SectorConfig Dataclass**: Encapsulates all sector-specific parameters
    - ✅ **Convenient Properties**: Access calibrations via `h_r_baseline`, `ebitda_multiplier`, etc.
    - ✅ **Database Integration**: Loads configuration from multiple tables (focus_groups, weights, calibrations)
    - ✅ **Validation**: Checks that dimension weights sum to 1.0
    - ✅ **Cache Integration**: Prepares data for Redis caching (covered in next section)
    - ✅ **Singleton Pattern**: Single service instance for the application
    """)

    if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
        st.warning(
            "Please seed dimension weights and calibrations first in the '2.2: Data Seeding' section.")
    else:
        st.subheader("Demonstration: Retrieve & Analyze Sector Configurations")
        st.markdown(
            "Select a PE sector below to retrieve its configuration using the `SectorConfigService` and examine its parameters.")

        fg_names = {fg['focus_group_id']: fg['group_name']
                    for fg in st.session_state.all_focus_groups}
        selected_fg_name = st.selectbox(
            "Select a Sector:",
            options=list(fg_names.values()),
            index=0,
            key="config_service_sector_select"
        )
        st.session_state.selected_sector_id = next(
            (fg_id for fg_id, name in fg_names.items() if name == selected_fg_name), None)

        if st.session_state.selected_sector_id:

            st.markdown(
                f"Calling `get_sector_config_from_service_sync('{st.session_state.selected_sector_id}')`...")

            # FIX: Removed the local Streamlit session state cache (st.session_state.sector_configs_cache)
            # This ensures that get_sector_config_from_service_sync is called every time,
            # allowing the MockRedisCache within it to demonstrate cache hits/misses.
            sector_config = get_sector_config_from_service_sync(
                st.session_state.selected_sector_id)

            if sector_config:
                st.success(
                    f"Configuration loaded for **{sector_config.group_name}** ({sector_config.focus_group_id})!")

                st.subheader(f"Parameters for {sector_config.group_name}:")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"**H^R Baseline:** `{sector_config.h_r_baseline}`")
                    st.markdown(
                        r"where $H^R$ represents the Human-Readiness baseline score for a sector.")
                    st.markdown(
                        f"**EBITDA Multiplier:** `{sector_config.ebitda_multiplier}`")
                    st.markdown(
                        r"where $EBITDA$ is Earnings Before Interest, Taxes, Depreciation, and Amortization, and the multiplier adjusts its attribution for the sector.")
                with col2:
                    st.markdown(
                        f"**Position Factor Delta:** `{sector_config.position_factor_delta}`")
                    st.markdown(
                        r"where $\delta$ is the adjustment factor for an organization's strategic position within the sector.")
                    st.markdown(
                        f"**Talent Concentration Threshold:** `{sector_config.talent_concentration_threshold}`")
                    st.markdown(
                        r"where the threshold identifies sectors with high concentration of specialized talent.")

                st.subheader(
                    f"Dimension Weights for {sector_config.group_name}:")
                st.markdown(
                    f"These weights define the relative importance of each AI/data dimension for the selected sector. They should sum up to 1.0 (or very close).")

                if sector_config.validate_weights_sum():
                    st.success(
                        f"Dimension weights sum to 1.0 (Validation successful).")
                else:
                    st.warning(
                        f"Dimension weights do NOT sum to 1.0 (Validation failed). Total: {sum(sector_config.dimension_weights.values())}")

                weights_df = get_dimension_weights_for_chart(sector_config)
                st.dataframe(weights_df, use_container_width=True)

                fig = px.bar(weights_df, x='Dimension', y='Weight',
                             title=f'Dimension Weights for {sector_config.group_name}',
                             labels={'Weight': 'Relative Importance'},
                             color='Weight', color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not retrieve sector configuration.")

# Page: 2.4: Redis Caching
elif st.session_state.current_page == "2.4: Redis Caching":
    st.title("Task 2.4: Build the Redis Caching Layer")

    st.markdown("To ensure our configuration service is highly performant and responsive, especially under heavy load, we introduce a Redis caching layer. Redis is an in-memory data store that provides extremely fast key-value lookups.")

    st.markdown("The `SectorConfigService` is enhanced to first check the cache before hitting the database. If the configuration is found in Redis, it's a **cache hit**; otherwise, it's a **cache miss**, and the data is loaded from the database and then stored in Redis for future requests.")

    st.info("📁 **File:** `src/pe_orgair/infrastructure/cache.py`")
    st.markdown("**Complete Redis Cache Manager Implementation:**")
    st.code("""# src/pe_orgair/infrastructure/cache.py
\"\"\"Redis caching with TTL and pub/sub.\"\"\"
from typing import Any, Optional, TypeVar, Callable
from functools import wraps
import json
import redis
import structlog

from pe_orgair.config.settings import settings

logger = structlog.get_logger()
T = TypeVar("T")


class CacheManager:
    \"\"\"Redis cache manager with TTL support.\"\"\"
    
    def __init__(self):
        self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self._pubsub = self._client.pubsub()
    
    def get(self, key: str) -> Optional[Any]:
        \"\"\"Get cached value.\"\"\"
        value = self._client.get(key)
        if value:
            logger.debug("cache_hit", key=key)
            return json.loads(value)
        logger.debug("cache_miss", key=key)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        \"\"\"Set cached value with TTL.\"\"\"
        self._client.setex(key, ttl, json.dumps(value))
        logger.debug("cache_set", key=key, ttl=ttl)
    
    def delete(self, key: str) -> None:
        \"\"\"Delete cached value.\"\"\"
        self._client.delete(key)
        logger.debug("cache_delete", key=key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        \"\"\"Invalidate all keys matching pattern.\"\"\"
        keys = self._client.keys(pattern)
        if keys:
            count = self._client.delete(*keys)
            logger.info("cache_invalidated", pattern=pattern, count=count)
            return count
        return 0
    
    def publish(self, channel: str, message: dict) -> None:
        \"\"\"Publish message to channel.\"\"\"
        self._client.publish(channel, json.dumps(message))
        logger.debug("pubsub_published", channel=channel)


cache = CacheManager()


def cached(key_template: str, ttl: int = 3600):
    \"\"\"Decorator for caching function results.\"\"\"
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_template.format(*args, **kwargs)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
""", language="python")

    st.markdown("""
    **Key Features of the Redis Cache Manager:**
    - ✅ **TTL Support**: Automatic expiration of cached values
    - ✅ **JSON Serialization**: Handles complex data structures
    - ✅ **Pattern-based Invalidation**: Clear multiple related cache entries at once
    - ✅ **Pub/Sub Support**: Real-time cache invalidation across services
    - ✅ **Decorator Support**: `@cached` decorator for easy function result caching
    - ✅ **Structured Logging**: Debug and audit cache operations
    """)

    st.markdown("**Integration with SectorConfigService:**")
    st.markdown(
        "The cache is used in the `get_config` method to check for cached configurations before hitting the database:")
    st.code("""# In SectorConfigService.get_config():
cache_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)

# Check cache first
cached = cache.get(cache_key)
if cached:
    # Cache HIT - return immediately
    return self._dict_to_config(cached)

# Cache MISS - load from database
config = await self._load_from_db(focus_group_id)
if config:
    # Store in cache for future requests
    cache.set(cache_key, self._config_to_dict(config), self.CACHE_TTL)
return config
""", language="python")

    st.markdown("Furthermore, a mechanism for **cache invalidation** is crucial. When configuration data changes in the database, the corresponding cached entries must be removed or updated to prevent serving stale data. The `invalidate_cache` method handles this.")

    if not st.session_state.weights_seeded or not st.session_state.calibrations_seeded:
        st.warning(
            "Please seed dimension weights and calibrations first in the '2.2: Data Seeding' section to demonstrate caching.")
    else:
        st.subheader("Demonstration: Caching Behavior and Invalidation")

        st.markdown("Select a sector and click 'Fetch Config (with cache)' multiple times to observe the caching. Then, try 'Invalidate Cache' and fetch again.")

        fg_names = {fg['focus_group_id']: fg['group_name']
                    for fg in st.session_state.all_focus_groups}
        all_fg_options = [
            "All Sectors (invalidate all)"] + list(fg_names.values())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🗑️ Cache Invalidation**")
            selected_invalidate_scope_name = st.selectbox(
                "Select scope for cache invalidation:",
                options=all_fg_options,
                index=0,
                key="cache_invalidate_scope_select"
            )

            invalidate_fg_id = None
            if selected_invalidate_scope_name != "All Sectors (invalidate all)":
                invalidate_fg_id = next((fg_id for fg_id, name in fg_names.items(
                ) if name == selected_invalidate_scope_name), None)

            if st.button("Invalidate Cache", use_container_width=True):
                with st.spinner(f"Invalidating cache for {selected_invalidate_scope_name}..."):
                    invalidate_sector_cache_service_sync(invalidate_fg_id)
                    # Removed `st.session_state.sector_configs_cache = {}` as it's no longer used.
                    st.success("Cache invalidation triggered!")

        with col2:
            st.markdown("**📥 Fetch Configuration**")
            selected_fetch_fg_name = st.selectbox(
                "Select a Sector to Fetch (observe cache hits/misses):",
                options=list(fg_names.values()),
                index=0,
                key="cache_fetch_sector_select"
            )

            fetch_fg_id = next((fg_id for fg_id, name in fg_names.items(
            ) if name == selected_fetch_fg_name), None)

            if st.button("Fetch Config (with cache)", key="fetch_config_button", use_container_width=True):
                with st.spinner(f"Fetching config for {selected_fetch_fg_name}..."):
                    fetched_config = get_sector_config_from_service_sync(
                        fetch_fg_id)
                    if fetched_config:
                        st.success(
                            f"Successfully fetched config for {fetched_config.group_name}.")
                        st.session_state.fetched_config = fetched_config

        # Display configuration outside columns for better layout
        if hasattr(st.session_state, 'fetched_config') and st.session_state.fetched_config:
            fetched_config = st.session_state.fetched_config

            # Display configuration in organized sections
            st.subheader("Configuration Details")

            with st.expander("📊 Dimension Weights", expanded=True):
                weights_display = [{'Dimension': k, 'Weight': float(v)}
                                   for k, v in fetched_config.dimension_weights.items()]
                st.dataframe(pd.DataFrame(weights_display),
                             use_container_width=True, hide_index=True)

            with st.expander("⚙️ Calibration Parameters", expanded=True):
                calib_display = [{'Parameter': k, 'Value': float(v)}
                                 for k, v in fetched_config.calibrations.items()]
                st.dataframe(pd.DataFrame(calib_display),
                             use_container_width=True, hide_index=True)


# Page: 2.5: Organization View
elif st.session_state.current_page == "2.5: Organization View":
    st.title("Task 2.5: Create the Unified Organization View")

    st.markdown("The ultimate goal of our configuration-driven architecture is to provide a unified, easily queryable view of organizations, enriching their core data with sector-specific attributes dynamically.")

    st.markdown("This is achieved by creating a database VIEW (`vw_organizations_full`) that joins the main `organizations` table with the respective `org_attributes_` tables based on the `focus_group_id` reference.")

    st.subheader("📋 SQL VIEW Definition")
    st.markdown(
        "Below is the complete SQL VIEW definition that creates the unified organization view:")

    st.info("📁 **File:** `migrations/versions/003_organization_views.sql`")
    sql_view_definition = """-- Views for querying organizations with sector attributes
CREATE OR REPLACE VIEW vw_organizations_full AS
SELECT
    o.*,
    fg.group_name AS sector_name,
    fg.group_code AS sector_code,
    -- Manufacturing
    mfg.plant_count, mfg.automation_level, mfg.digital_twin_status,
    -- Financial Services
    fin.regulatory_bodies, fin.algo_trading, fin.aum_billions,
    -- Healthcare
    hc.hipaa_certified, hc.ehr_system, hc.clinical_ai_deployed,
    -- Technology
    tech.github_stars_total, tech.ml_platform, tech.ai_product_features,
    -- Retail
    rtl.cdp_vendor, rtl.personalization_ai, rtl.store_count,
    -- Energy
    enr.smart_grid_pct, enr.predictive_maintenance, enr.renewable_pct,
    -- Professional Services
    ps.firm_type, ps.client_ai_services, ps.document_ai
FROM organizations o
JOIN focus_groups fg ON o.focus_group_id = fg.focus_group_id
LEFT JOIN org_attributes_manufacturing mfg ON o.organization_id = mfg.organization_id
LEFT JOIN org_attributes_financial_services fin ON o.organization_id = fin.organization_id
LEFT JOIN org_attributes_healthcare hc ON o.organization_id = hc.organization_id
LEFT JOIN org_attributes_technology tech ON o.organization_id = tech.organization_id
LEFT JOIN org_attributes_retail rtl ON o.organization_id = rtl.organization_id
LEFT JOIN org_attributes_energy enr ON o.organization_id = enr.organization_id
LEFT JOIN org_attributes_professional_services ps ON o.organization_id = ps.organization_id;"""

    st.code(sql_view_definition, language="sql")

    st.markdown("""
    **Key Features of this View:**
    - ✅ Joins all organization core data from the `organizations` table
    - ✅ Includes sector metadata (name, code) from `focus_groups`
    - ✅ Uses `LEFT JOIN` to include sector-specific attributes for all 7 sectors
    - ✅ Provides a single query interface for all organization data
    - ✅ NULL values appear for sectors that don't match specific attribute tables
    """)

    st.markdown("This view allows us to query all organizations and their relevant sector attributes in a single statement, without needing to know which specific attribute table to query beforehand. The `LEFT JOIN` ensures that even organizations without specific attributes in a given table are still included.")

    if not st.session_state.db_initialized:
        st.warning(
            "Please initialize the database schema first in the '2.1: Schema Design & Attributes' section.")
    elif not st.session_state.initial_data_seeded:
        st.warning(
            "Please seed initial focus groups and dimensions first in the '2.2: Data Seeding' section.")
    else:
        st.subheader("Action: Insert Sample Organizations")
        st.markdown(
            "Let's populate our `organizations` table with some sample data, ensuring they are linked to different PE sectors.")
        if not st.session_state.organizations_seeded:
            if st.button("Insert Sample Organizations"):
                with st.spinner("Inserting sample organizations and attributes..."):
                    success = insert_sample_organizations(num_orgs=10)
                    if success:
                        st.session_state.organizations_seeded = True
                        st.success(
                            "10 sample organizations inserted with sector-specific attributes!")
                    else:
                        st.error(
                            "Failed to insert sample organizations. Ensure base data is seeded.")
        else:
            st.info("Sample organizations already inserted.")

        if st.session_state.organizations_seeded:
            st.subheader("Demonstration: Query Unified Organization View")

            st.markdown(
                "Now, let's query the `vw_organizations_full` view. You can filter by a specific sector or view all organizations.")

            fg_names = {fg['focus_group_id']: fg['group_name']
                        for fg in st.session_state.all_focus_groups}
            filter_options = {"All Sectors": None}
            filter_options.update({fg['group_name']: fg['focus_group_id']
                                  for fg in st.session_state.all_focus_groups})

            selected_filter_name = st.selectbox(
                "Filter Organizations by Sector:",
                options=list(filter_options.keys()),
                index=0,
                key="unified_view_filter_select"
            )
            st.session_state.unified_org_filter_sector_id = filter_options[selected_filter_name]

            if st.button("Fetch Unified Organization Data"):
                with st.spinner(f"Fetching data for {selected_filter_name}..."):
                    org_data_df = fetch_unified_organization_data_sync(
                        st.session_state.unified_org_filter_sector_id)
                    if not org_data_df.empty:
                        st.success(
                            f"✅ Successfully retrieved {len(org_data_df)} organizations from `vw_organizations_full`")

                        # Show some metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Organizations", len(org_data_df))
                        with col2:
                            total_revenue = org_data_df['annual_revenue_usd'].sum(
                            ) / 1_000_000_000
                            st.metric("Total Revenue",
                                      f"${total_revenue:.2f}B")
                        with col3:
                            total_employees = org_data_df['employee_count'].sum(
                            )
                            st.metric("Total Employees",
                                      f"{total_employees:,}")

                        st.subheader("📊 Query Results")
                        st.dataframe(
                            org_data_df, use_container_width=True, height=400)

                        # Show example SQL query used
                        st.subheader("🔍 Example Query Used")
                        if st.session_state.unified_org_filter_sector_id:
                            example_query = f"""SELECT * FROM vw_organizations_full 
WHERE focus_group_id = '{st.session_state.unified_org_filter_sector_id}';"""
                        else:
                            example_query = """SELECT * FROM vw_organizations_full;"""
                        st.code(example_query, language="sql")
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
