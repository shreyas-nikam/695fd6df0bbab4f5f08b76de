
from streamlit.testing.v1 import AppTest
import pytest
import pandas as pd
from decimal import Decimal

# Sample data for focus groups, used to simulate state after seeding
SAMPLE_FOCUS_GROUPS = [
    {'focus_group_id': 'pe_manufacturing', 'group_name': 'Manufacturing', 'group_code': 'MFG', 'platform': 'pe_org_air'},
    {'focus_group_id': 'pe_financial_services', 'group_name': 'Financial Services', 'group_code': 'FIN', 'platform': 'pe_org_air'},
    {'focus_group_id': 'pe_technology', 'group_name': 'Technology', 'group_code': 'TECH', 'platform': 'pe_org_air'}
]

# Helper to run the app with initial session state values set
def run_app_with_state(page: str = 'Home', db_initialized: bool = False, initial_data_seeded: bool = False,
                       weights_seeded: bool = False, calibrations_seeded: bool = False,
                       organizations_seeded: bool = False,
                       all_focus_groups: list = None,
                       sector_configs_cache: dict = None):
    at = AppTest.from_file("app.py")
    at.session_state.current_page = page
    at.session_state.db_initialized = db_initialized
    at.session_state.initial_data_seeded = initial_data_seeded
    at.session_state.weights_seeded = weights_seeded
    at.session_state.calibrations_seeded = calibrations_seeded
    at.session_state.organizations_seeded = organizations_seeded
    if all_focus_groups is not None:
        at.session_state.all_focus_groups = all_focus_groups
    else:
        # Ensure it's always a list, even if empty
        at.session_state.all_focus_groups = []
    if sector_configs_cache is not None:
        at.session_state.sector_configs_cache = sector_configs_cache
    else:
        at.session_state.sector_configs_cache = {}
    return at.run()

# A more comprehensive helper for session state setup assuming everything is seeded
def get_fully_seeded_at(current_page: str = 'Home'):
    at = AppTest.from_file("app.py")
    at.session_state.current_page = current_page
    at.session_state.db_initialized = True
    at.session_state.initial_data_seeded = True
    at.session_state.weights_seeded = True
    at.session_state.calibrations_seeded = True
    at.session_state.organizations_seeded = True
    at.session_state.all_focus_groups = SAMPLE_FOCUS_GROUPS
    at.session_state.sector_configs_cache = {} # Empty cache for fresh fetch tests
    return at.run()


def test_home_page_renders_correctly():
    at = run_app_with_state('Home')
    assert at.title[0].value == "QuLab: Data Layer & Caching"
    assert at.title[1].value == "Week 2: Unified Data Architecture & Caching Lab"
    assert "Welcome to Week 2 of our journey" in at.markdown[0].value
    assert "Key Objectives" in at.subheader[0].value
    assert "Tools Introduced" in at.subheader[1].value
    assert "Key Concepts" in at.subheader[2].value


def test_schema_design_page_renders_and_initializes_db():
    at = run_app_with_state('2.1-2.5: Schema Design & Attributes', db_initialized=False)
    assert at.title[1].value == "Task 2.1-2.5: Database Schema Design"
    assert "CREATE TABLE focus_groups" in at.markdown[6].value
    assert at.button[0].label == "Initialize Database Schema"
    
    # Simulate button click
    at.button[0].click().run()
    assert at.session_state.db_initialized is True
    assert at.success[0].value == "Database schema initialized successfully!"
    assert at.balloons[0].value is True
    assert at.info[0].value == "Database schema is already initialized." # App should now show this


def test_data_seeding_page_warning_if_db_not_initialized():
    at = run_app_with_state('2.2-2.3: Data Seeding', db_initialized=False)
    assert at.warning[0].value == "Please initialize the database schema first in the 'Schema Design & Attributes' section."


def test_data_seeding_initial_focus_groups_and_dimensions():
    at = run_app_with_state('2.2-2.3: Data Seeding', db_initialized=True, initial_data_seeded=False)
    assert at.button[0].label == "Seed Initial Focus Groups & Dimensions"

    # Simulate button click
    at.button[0].click().run()
    assert at.session_state.initial_data_seeded is True
    assert at.success[0].value == "Initial focus groups and dimensions seeded!"
    assert len(at.session_state.all_focus_groups) == len(SAMPLE_FOCUS_GROUPS) # Assuming `get_all_focus_groups` returns SAMPLE_FOCUS_GROUPS
    
    # Rerunning with seeded state should show info and dataframe
    at = run_app_with_state('2.2-2.3: Data Seeding', db_initialized=True, initial_data_seeded=True, all_focus_groups=SAMPLE_FOCUS_GROUPS)
    assert at.info[0].value == "Initial focus groups and dimensions already seeded."
    assert at.dataframe[0].value.equals(pd.DataFrame(SAMPLE_FOCUS_GROUPS))


def test_data_seeding_weights_and_calibrations():
    at = run_app_with_state('2.2-2.3: Data Seeding', db_initialized=True, initial_data_seeded=True, all_focus_groups=SAMPLE_FOCUS_GROUPS)
    assert at.button[1].label == "Seed All Weights and Calibrations"

    # Simulate button click
    at.button[1].click().run()
    assert at.session_state.weights_seeded is True
    assert at.session_state.calibrations_seeded is True
    assert at.success[0].value == "Dimension weights seeded successfully!"
    assert at.success[1].value == "Calibrations seeded successfully!"
    assert at.balloons[0].value is True

    # Rerunning with all seeded state should show info
    at = run_app_with_state('2.2-2.3: Data Seeding', db_initialized=True, initial_data_seeded=True,
                            weights_seeded=True, calibrations_seeded=True, all_focus_groups=SAMPLE_FOCUS_GROUPS)
    assert at.info[1].value == "All dimension weights and calibrations are already seeded."


def test_sector_config_service_page_warning_if_not_seeded():
    at = run_app_with_state('2.6: Sector Configuration Service', weights_seeded=False, calibrations_seeded=False)
    assert at.warning[0].value == "Please seed dimension weights and calibrations first in the 'Data Seeding' section."


def test_sector_config_service_displays_config_details():
    at = get_fully_seeded_at('2.6: Sector Configuration Service')

    # Select "Manufacturing" from the selectbox
    at.selectbox[0].set_value("Manufacturing").run()
    
    assert at.session_state.selected_sector_id == 'pe_manufacturing'
    assert at.success[0].value.startswith("Configuration loaded for **Manufacturing**")
    
    # Verify displayed parameters (assuming source.py mock returns these values for Manufacturing)
    # The actual values depend on the `get_sector_config_from_service_sync` implementation in `source.py`
    # Here, we assert based on the values provided in the app description as examples
    assert "H^R Baseline: `72`" in at.markdown[11].value
    assert "EBITDA Multiplier: `0.90`" in at.markdown[12].value
    assert "Position Factor Delta: `0.05`" in at.markdown[13].value
    assert "Talent Concentration Threshold: `0.85`" in at.markdown[14].value

    assert "Dimension Weights for Manufacturing:" in at.subheader[2].value
    assert at.success[1].value == "Dimension weights sum to 1.0 (Validation successful)." # Assumes the mocked or actual data sums to 1.0
    assert at.dataframe[0].value is not None
    assert at.plotly_chart[0].value is not None


def test_redis_caching_layer_invalidates_and_fetches():
    at = get_fully_seeded_at('2.7: Redis Caching Layer')

    # Invalidate all sectors cache
    at.selectbox[0].set_value("All Sectors (invalidate all)").run()
    at.button[0].click().run() # Click "Invalidate Cache"
    assert at.success[0].value == "Cache invalidation triggered!"
    assert at.session_state.sector_configs_cache == {}

    # Fetch config for "Financial Services"
    at.selectbox[1].set_value("Financial Services").run()
    at.button[1].click().run() # Click "Fetch Config (with cache)"
    assert at.success[1].value == "Successfully fetched config for Financial Services."
    assert at.json[0].value is not None # Check if JSON output is present


def test_unified_organization_view_warnings():
    at = run_app_with_state('2.8: Unified Organization View', db_initialized=False)
    assert at.warning[0].value == "Please initialize the database schema first in the 'Schema Design & Attributes' section."

    at = run_app_with_state('2.8: Unified Organization View', db_initialized=True, initial_data_seeded=False)
    assert at.warning[0].value == "Please seed initial focus groups and dimensions first in the 'Data Seeding' section."


def test_unified_organization_view_insert_and_fetch_organizations():
    at = get_fully_seeded_at('2.8: Unified Organization View')
    at.session_state.organizations_seeded = False # Reset for this specific test case
    
    # Insert sample organizations
    at.button[0].click().run() # Click "Insert Sample Organizations"
    assert at.session_state.organizations_seeded is True
    assert at.success[0].value == "10 sample organizations inserted with sector-specific attributes!"
    assert at.balloons[0].value is True

    # Check info message when already seeded
    at = get_fully_seeded_at('2.8: Unified Organization View')
    assert at.info[0].value == "Sample organizations already inserted."

    # Fetch all organizations
    at.selectbox[0].set_value("All Sectors").run()
    at.button[1].click().run() # Click "Fetch Unified Organization Data"
    assert at.dataframe[0].value is not None
    assert at.success[1].value.startswith("Displayed")
    assert len(at.dataframe[0].value) >= 10 # Assuming at least 10 sample orgs are returned by mock
    # The exact number depends on the `insert_sample_organizations` and `fetch_unified_organization_data_sync` implementations in source.py

    # Fetch organizations filtered by "Manufacturing"
    at.selectbox[0].set_value("Manufacturing").run()
    at.button[1].click().run()
    assert at.dataframe[0].value is not None
    assert at.success[1].value.startswith("Displayed")
    # The number of organizations will depend on the `insert_sample_organizations` mock distributing them
    # and `fetch_unified_organization_data_sync` filtering them. We expect it to be a subset.
    assert "Manufacturing" in at.dataframe[0].value['sector_name'].unique()


def test_sidebar_navigation_between_pages():
    at = AppTest.from_file("app.py").run()
    
    # Navigate to "2.1-2.5: Schema Design & Attributes"
    at.selectbox[0].set_value("2.1-2.5: Schema Design & Attributes").run()
    assert at.session_state.current_page == "2.1-2.5: Schema Design & Attributes"
    assert at.title[1].value == "Task 2.1-2.5: Database Schema Design"

    # Navigate to "2.7: Redis Caching Layer"
    at.selectbox[0].set_value("2.7: Redis Caching Layer").run()
    assert at.session_state.current_page == "2.7: Redis Caching Layer"
    assert at.title[1].value == "Task 2.7: Build the Redis Caching Layer"

    # Navigate to "2.8: Unified Organization View"
    at.selectbox[0].set_value("2.8: Unified Organization View").run()
    assert at.session_state.current_page == "2.8: Unified Organization View"
    assert at.title[1].value == "Task 2.8: Create the Unified Organization View"

