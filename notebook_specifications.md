
# Building a Configuration-Driven Data Platform for PE Sector-Specific Insights

## Introduction: Empowering PE Org-AI-R with Flexible Data Architecture

As a **Software Developer** at PE Org-AI-R, our mission is to build a robust platform that provides unparalleled insights into private equity investments across diverse sectors like Manufacturing, Healthcare, and Financial Services. Each of these 7 sectors has unique evaluation criteria, investment parameters, and data attributes. The challenge is to manage these sector-specific behaviors without creating fragmented database schemas or embedding complex, hardcoded logic that becomes a maintenance nightmare.

This notebook will guide you through the process of designing and implementing a **configuration-driven data architecture**. This approach centralizes and standardizes how sector-specific logic is managed, ensuring flexibility, scalability, and maintainability. Instead of modifying code or schema for every new sector requirement, we'll define these behaviors through data, allowing for dynamic adjustments and future expansion.

By the end of this lab, you will have built the foundational components for a data platform that truly adapts to the unique needs of each private equity sector.

---

## 1. Setup: Installing Libraries & Initializing Mock Infrastructure

Before diving into schema design and data seeding, we need to set up our environment by installing the necessary Python libraries and preparing a simulated database and caching layer. For this demonstration, we'll use a `MockDatabaseClient` to simulate PostgreSQL interactions and a `MockRedisClient` to emulate Redis caching, allowing us to focus on the core logic without requiring external service setup.

```python
!pip install pandas matplotlib seaborn psycopg2-binary redis
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from decimal import Decimal
import json
import uuid
import datetime

# --- Mock Infrastructure Classes ---

# Mock Database Client for PostgreSQL simulation
class MockDatabaseClient:
    """Simulates PostgreSQL database operations for demonstration purposes."""
    def __init__(self):
        self.tables = {}
        self.sequences = {} # For SERIAL/AUTOINCREMENT
        self.foreign_keys = {}
        self.unique_constraints = {}

    def _get_next_sequence_val(self, seq_name):
        if seq_name not in self.sequences:
            self.sequences[seq_name] = 1
        else:
            self.sequences[seq_name] += 1
        return self.sequences[seq_name]

    def execute_ddl(self, sql_statement: str):
        """Simulates executing DDL statements like CREATE TABLE."""
        # Simplified DDL parsing for demonstration
        table_name = None
        if "CREATE TABLE" in sql_statement:
            table_name = sql_statement.split("CREATE TABLE ")[1].split(" ")[0].strip()
            self.tables[table_name] = [] # Initialize table as an empty list of dicts
            print(f"Mock DB: Table '{table_name}' created (simulated).")
        elif "CREATE OR REPLACE VIEW" in sql_statement:
            view_name = sql_statement.split("CREATE OR REPLACE VIEW ")[1].split(" ")[0].strip()
            # Views are just metadata for now, not actual data storage in mock DB
            print(f"Mock DB: View '{view_name}' created (simulated).")
        else:
            print(f"Mock DB: DDL executed (simulated, statement not fully parsed): {sql_statement[:50]}...")

    def fetch_one(self, query: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulates fetching a single row."""
        # This is a very basic simulation; a real ORM would parse this
        # For this lab, assume the query is simple enough to match a specific record
        table_name = query.split("FROM ")[1].split(" ")[0].strip()
        if table_name not in self.tables:
            return None

        # Simple WHERE clause matching based on params
        for row in self.tables[table_name]:
            match = True
            for k, v in params.items():
                if row.get(k) != v:
                    match = False
                    break
            if match:
                return row
        return None

    def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Simulates fetching multiple rows."""
        table_name = query.split("FROM ")[1].split(" ")[0].strip() # Get table name after FROM
        if table_name not in self.tables:
            return []

        results = []
        for row in self.tables[table_name]:
            # Apply simple WHERE clause filtering if params are provided
            if params:
                match = True
                for k, v in params.items():
                    if row.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Apply simple SELECT clause filtering if needed, otherwise return full row
            selected_cols = []
            if "SELECT" in query:
                select_clause = query.split("SELECT ")[1].split(" FROM")[0]
                selected_cols = [col.strip() for col in select_clause.split(',')]
                
                # Handle aliases like 'd.dimension_code'
                processed_row = {}
                for col in selected_cols:
                    if '.' in col:
                        # Assuming alias is 'd' or 'w' as in the provided queries
                        real_col_name = col.split('.')[-1]
                        if real_col_name in row:
                            processed_row[real_col_name] = row[real_col_name]
                    elif col in row:
                        processed_row[col] = row[col]
                    else:
                        processed_row[col] = None # Or raise error, depending on strictness
                results.append(processed_row)
            else:
                results.append(row)
        
        # Simulate ORDER BY if present (very basic)
        if "ORDER BY" in query:
            order_by_col = query.split("ORDER BY ")[1].strip().split(' ')[0]
            if '.' in order_by_col: # Handle d.display_order
                order_by_col = order_by_col.split('.')[-1]
            if order_by_col in results[0]: # Check if column exists in results
                results.sort(key=lambda x: x.get(order_by_col))

        return results


    def insert_rows(self, table_name: str, rows: List[Dict[str, Any]]):
        """Simulates inserting multiple rows into a table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' does not exist.")
        
        for row in rows:
            processed_row = row.copy()
            # Handle SERIAL/AUTOINCREMENT for primary keys if column is like 'weight_id' or 'calibration_id'
            if 'weight_id' in processed_row and processed_row['weight_id'] is None:
                processed_row['weight_id'] = self._get_next_sequence_val(f"{table_name}_weight_id_seq")
            if 'calibration_id' in processed_row and processed_row['calibration_id'] is None:
                processed_row['calibration_id'] = self._get_next_sequence_val(f"{table_name}_calibration_id_seq")
            if 'organization_id' in processed_row and processed_row['organization_id'] is None:
                processed_row['organization_id'] = str(uuid.uuid4())
            
            # Apply default values for TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            if 'created_at' in processed_row and processed_row['created_at'] is None:
                processed_row['created_at'] = datetime.datetime.now().isoformat()
            if 'updated_at' in processed_row and processed_row['updated_at'] is None:
                processed_row['updated_at'] = datetime.datetime.now().isoformat()
            if 'effective_from' in processed_row and processed_row['effective_from'] is None:
                processed_row['effective_from'] = datetime.date.today().isoformat()
            if 'is_current' in processed_row and processed_row['is_current'] is None:
                processed_row['is_current'] = True

            self.tables[table_name].append(processed_row)
        print(f"Mock DB: Inserted {len(rows)} rows into '{table_name}'.")

    def get_table_data(self, table_name: str) -> List[Dict[str, Any]]:
        """Retrieves all data from a simulated table."""
        return self.tables.get(table_name, [])

    def clear(self):
        """Clears all tables and sequences for fresh start."""
        self.tables = {}
        self.sequences = {}
        self.foreign_keys = {}
        self.unique_constraints = {}
        print("Mock DB: All tables and sequences cleared.")

db = MockDatabaseClient()

# Mock Redis Client for caching simulation
class MockRedisClient:
    """Simulates Redis caching operations."""
    def __init__(self):
        self.cache = {}
        self.pubsub_channels = {}

    def get(self, key: str) -> Optional[str]:
        """Retrieves a value from cache."""
        val = self.cache.get(key)
        if val:
            print(f"Mock Redis: Cache HIT for '{key}'")
            return json.dumps(val) # Return as JSON string to mimic real Redis
        print(f"Mock Redis: Cache MISS for '{key}'")
        return None

    def setex(self, key: str, ttl: int, value: str) -> None:
        """Sets a value in cache with a TTL."""
        self.cache[key] = json.loads(value) # Store as Python object
        print(f"Mock Redis: Set cache for '{key}' with TTL {ttl}s")

    def delete(self, key: str) -> None:
        """Deletes a key from cache."""
        if key in self.cache:
            del self.cache[key]
            print(f"Mock Redis: Deleted '{key}' from cache.")
        else:
            print(f"Mock Redis: Key '{key}' not found for deletion.")

    def keys(self, pattern: str) -> List[str]:
        """Finds keys matching a pattern."""
        # Simple pattern matching, e.g., 'prefix:*'
        prefix = pattern.replace('*', '')
        return [k for k in self.cache if k.startswith(prefix)]

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidates all keys matching a pattern."""
        keys_to_delete = self.keys(pattern)
        count = 0
        for key in keys_to_delete:
            self.delete(key)
            count += 1
        print(f"Mock Redis: Invalidated {count} keys matching pattern '{pattern}'.")
        return count

    def publish(self, channel: str, message: str) -> None:
        """Publishes a message to a channel."""
        self.pubsub_channels.setdefault(channel, []).append(json.loads(message))
        print(f"Mock Redis: Published message to channel '{channel}'")

    def clear(self):
        """Clears the cache and pubsub channels."""
        self.cache = {}
        self.pubsub_channels = {}
        print("Mock Redis: Cache and pub/sub channels cleared.")

cache = MockRedisClient()

# Mock Logger for structured logging
class MockLogger:
    def info(self, message: str, **kwargs):
        print(f"INFO: {message} {kwargs}")
    def warning(self, message: str, **kwargs):
        print(f"WARNING: {message} {kwargs}")
    def exception(self, message: str, **kwargs):
        print(f"EXCEPTION: {message} {kwargs}")

logger = MockLogger()

# Initialize Decimal context
import decimal
decimal.getcontext().prec = 10 # Set precision for Decimal operations

print("\nRequired libraries installed and mock infrastructure initialized.")
```

---

## 2. Task 2.1: Designing the Core Configuration Schema

### Story + Context + Real-World Relevance

Our first step in building a configuration-driven data platform is to define the core schema for managing sector-specific configurations. As a Data Engineer, I know that having a flexible and normalized schema is critical to prevent "schema proliferation" – where a new schema is created for every sector, leading to unmanageable complexity. Instead, we'll implement a "One Schema, Many Configurations" approach. This means we design generic tables that can hold configuration data (weights and calibrations) for *all* sectors as rows, rather than columns or separate tables per sector.

We need three main configuration tables:
1.  `focus_groups`: Defines the sectors themselves (e.g., Manufacturing, Healthcare).
2.  `dimensions`: Defines the generic evaluation dimensions (e.g., Data Infrastructure, AI Governance) that apply across sectors.
3.  `focus_group_dimension_weights`: Stores the importance (weight) of each dimension for a specific sector.
4.  `focus_group_calibrations`: Stores numeric parameters specific to each sector.

This design allows us to manage all sector configurations as data, making the system highly adaptable. For example, if the weighting of 'AI Governance' needs to change for the 'Financial Services' sector, it's a simple data update, not a code deployment.

```python
def create_focus_group_schema():
    """
    Creates the DDL for the core focus group configuration tables in PostgreSQL.
    This demonstrates the "One Schema, Many Configurations" principle.
    """
    ddl_statements = [
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
        CREATE INDEX idx_weights_current ON focus_group_dimension_weights(focus_group_id, is_current)
        WHERE is_current = TRUE;
        """,
        """
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
        """
    ]
    for ddl in ddl_statements:
        db.execute_ddl(ddl)

db.clear() # Ensure a clean slate for schema creation
create_focus_group_schema()
print("\nCore configuration schema (focus groups, dimensions, weights, calibrations) defined.")
```

### Explanation of Execution

The code above simulates the execution of PostgreSQL Data Definition Language (DDL) statements to create the foundational tables. The `focus_groups` table defines the 7 PE sectors. The `dimensions` table lists the generic criteria used to evaluate companies. `focus_group_dimension_weights` and `focus_group_calibrations` are the crucial "configuration tables" where sector-specific behaviors are stored as data. This setup is a direct application of the "One Schema, Many Configurations" principle, preventing future schema complexity for the PE Org-AI-R platform. The index `idx_weights_current` helps in quickly retrieving the active weights for a sector, which is important for performance.

---

## 3. Task 2.2: Seeding Sector Dimension Weights

### Story + Context + Real-World Relevance

With the schema prepared, the next step is to populate it with the actual configuration data. As a Data Engineer, seeding this initial data is crucial to make the platform operational. We'll start by defining the generic dimensions and then their specific weights for each of the 7 PE sectors. These weights directly influence how each sector evaluates potential investments, reflecting their strategic priorities. For example, 'Data Infrastructure' might be weighted higher in Manufacturing due to OT/IT integration, while 'Talent' is key in Technology.

The "weight" of a dimension for a particular sector, $w_{sd}$, represents its relative importance. We expect that for any given sector $s$, the sum of weights across all dimensions $D$ should be close to 1.0 (or 100%), i.e., $\sum_{d \in D} w_{sd} \approx 1$. Deviations from this sum could indicate an issue with the configuration.

```python
def seed_initial_data():
    """
    Seeds initial data for focus groups and dimensions, then populates dimension weights for sectors.
    """
    # Seed PE Org-AI-R Sectors
    focus_groups_data = [
        {'focus_group_id': 'pe_manufacturing', 'platform': 'pe_org_air', 'group_name': 'Manufacturing', 'group_code': 'MFG', 'display_order': 1, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_financial_services', 'platform': 'pe_org_air', 'group_name': 'Financial Services', 'group_code': 'FIN', 'display_order': 2, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_healthcare', 'platform': 'pe_org_air', 'group_name': 'Healthcare', 'group_code': 'HC', 'display_order': 3, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_technology', 'platform': 'pe_org_air', 'group_name': 'Technology', 'group_code': 'TECH', 'display_order': 4, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_retail', 'platform': 'pe_org_air', 'group_name': 'Retail & Consumer', 'group_code': 'RTL', 'display_order': 5, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_energy', 'platform': 'pe_org_air', 'group_name': 'Energy & Utilities', 'group_code': 'ENR', 'display_order': 6, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
        {'focus_group_id': 'pe_professional_services', 'platform': 'pe_org_air', 'group_name': 'Professional Services', 'group_code': 'PS', 'display_order': 7, 'group_description': None, 'icon_name': None, 'color_hex': None, 'is_active': True, 'created_at': None, 'updated_at': None},
    ]
    db.insert_rows('focus_groups', focus_groups_data)

    # Seed Dimensions
    dimensions_data = [
        {'dimension_id': 'pe_dim_data_infra', 'platform': 'pe_org_air', 'dimension_name': 'Data Infrastructure', 'dimension_code': 'data_infrastructure', 'display_order': 1, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_governance', 'platform': 'pe_org_air', 'dimension_name': 'AI Governance', 'dimension_code': 'ai_governance', 'display_order': 2, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_tech_stack', 'platform': 'pe_org_air', 'dimension_name': 'Technology Stack', 'dimension_code': 'technology_stack', 'display_order': 3, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_talent', 'platform': 'pe_org_air', 'dimension_name': 'Talent', 'dimension_code': 'talent', 'display_order': 4, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_leadership', 'platform': 'pe_org_air', 'dimension_name': 'Leadership', 'dimension_code': 'leadership', 'display_order': 5, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_use_cases', 'platform': 'pe_org_air', 'dimension_name': 'Use Case Portfolio', 'dimension_code': 'use_case_portfolio', 'display_order': 6, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
        {'dimension_id': 'pe_dim_culture', 'platform': 'pe_org_air', 'dimension_name': 'Culture', 'dimension_code': 'culture', 'display_order': 7, 'description': None, 'min_score': 0, 'max_score': 100, 'created_at': None},
    ]
    db.insert_rows('dimensions', dimensions_data)

    # Seed Dimension Weights for all sectors
    dimension_weights_data = [
        # Manufacturing
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.22'), 'weight_rationale': 'OT/IT integration critical', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.12'), 'weight_rationale': 'Less regulatory than finance/health', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.18'), 'weight_rationale': 'Edge computing, IoT platforms', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.15'), 'weight_rationale': 'AI + manufacturing expertise scarce', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.12'), 'weight_rationale': 'Traditional leadership acceptable', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.14'), 'weight_rationale': 'Clear ROI in operations', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_manufacturing', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.07'), 'weight_rationale': 'Safety culture > innovation', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Financial Services
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.16'), 'weight_rationale': 'Mature infrastructure', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.22'), 'weight_rationale': 'Regulatory imperative', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.14'), 'weight_rationale': 'Standard cloud stacks', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.18'), 'weight_rationale': 'Quant + ML talent critical', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.12'), 'weight_rationale': 'C-suite AI awareness high', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.10'), 'weight_rationale': 'Well-understood use cases', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_financial_services', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.08'), 'weight_rationale': 'Risk-averse by design', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Healthcare
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.20'), 'weight_rationale': 'EHR integration critical', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.20'), 'weight_rationale': 'FDA/HIPAA compliance', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.14'), 'weight_rationale': 'EHR-centric ecosystems', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.15'), 'weight_rationale': 'Clinical + AI dual expertise', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.15'), 'weight_rationale': 'Physician champions matter', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.10'), 'weight_rationale': 'Long validation cycles', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_healthcare', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.06'), 'weight_rationale': 'Evidence-based culture exists', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Technology
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.15'), 'weight_rationale': 'Assumed competent', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.12'), 'weight_rationale': 'Less regulated', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.18'), 'weight_rationale': 'Core differentiator', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.22'), 'weight_rationale': 'Talent is everything', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.13'), 'weight_rationale': 'Tech-savvy by default', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.15'), 'weight_rationale': 'Product innovation', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_technology', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.05'), 'weight_rationale': 'Innovation assumed', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Retail & Consumer
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.20'), 'weight_rationale': 'Customer data platforms', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.12'), 'weight_rationale': 'Privacy focus, less regulated', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.15'), 'weight_rationale': 'Standard cloud + CDP', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.15'), 'weight_rationale': 'Data science accessible', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.13'), 'weight_rationale': 'Digital transformation focus', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.18'), 'weight_rationale': 'Clear revenue impact', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_retail', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.07'), 'weight_rationale': 'Customer-centric exists', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Energy & Utilities
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.22'), 'weight_rationale': 'SCADA/OT data critical', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.15'), 'weight_rationale': 'Regulatory + safety', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.18'), 'weight_rationale': 'Grid tech, edge computing', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.12'), 'weight_rationale': 'Talent scarcity', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.13'), 'weight_rationale': 'Traditional but evolving', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.15'), 'weight_rationale': 'Clear operational value', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_energy', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.05'), 'weight_rationale': 'Safety culture paramount', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Professional Services
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_data_infra', 'weight': Decimal('0.14'), 'weight_rationale': 'Document-centric', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_governance', 'weight': Decimal('0.15'), 'weight_rationale': 'Client confidentiality', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_tech_stack', 'weight': Decimal('0.12'), 'weight_rationale': 'Standard productivity', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_talent', 'weight': Decimal('0.22'), 'weight_rationale': 'People are the product', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_leadership', 'weight': Decimal('0.17'), 'weight_rationale': 'Partner adoption critical', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_use_cases', 'weight': Decimal('0.12'), 'weight_rationale': 'Client + internal', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'weight_id': None, 'focus_group_id': 'pe_professional_services', 'dimension_id': 'pe_dim_culture', 'weight': Decimal('0.08'), 'weight_rationale': 'Innovation varies', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
    ]
    db.insert_rows('focus_group_dimension_weights', dimension_weights_data)

seed_initial_data()

# Fetch and display dimension weights for one sector (e.g., Manufacturing)
fg_weights_query = """
SELECT
    fg.group_name AS sector_name,
    d.dimension_name,
    w.weight,
    w.weight_rationale
FROM focus_group_dimension_weights w
JOIN focus_groups fg ON w.focus_group_id = fg.focus_group_id
JOIN dimensions d ON w.dimension_id = d.dimension_id
WHERE fg.focus_group_id = %(focus_group_id)s
AND w.is_current = TRUE
ORDER BY d.display_order;
"""
manufacturing_weights = db.fetch_all(fg_weights_query, {'focus_group_id': 'pe_manufacturing'})
df_manufacturing_weights = pd.DataFrame(manufacturing_weights)
print("\nManufacturing Sector Dimension Weights:")
display(df_manufacturing_weights)

# Visualize weights across all dimensions for Manufacturing
plt.figure(figsize=(10, 6))
sns.barplot(x='dimension_name', y='weight', data=df_manufacturing_weights, palette='viridis')
plt.title('Dimension Weights for Manufacturing Sector')
plt.xlabel('Dimension Name')
plt.ylabel('Weight')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Get all dimension weights to check the sum and for cross-sector comparison
all_weights_query = """
SELECT
    fg.group_name AS sector_name,
    d.dimension_code,
    w.weight
FROM focus_group_dimension_weights w
JOIN focus_groups fg ON w.focus_group_id = fg.focus_group_id
JOIN dimensions d ON w.dimension_id = d.dimension_id
WHERE w.is_current = TRUE;
"""
all_weights = db.fetch_all(all_weights_query)
df_all_weights = pd.DataFrame(all_weights)

# Calculate sum of weights for each sector
weight_sums = df_all_weights.groupby('sector_name')['weight'].sum().reset_index()
weight_sums.rename(columns={'weight': 'total_weight'}, inplace=True)
print("\nTotal Dimension Weight per Sector:")
display(weight_sums)

# Visualize dimension weights across sectors
pivot_df = df_all_weights.pivot(index='sector_name', columns='dimension_code', values='weight')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('Dimension Weights Across PE Sectors')
plt.xlabel('Dimension')
plt.ylabel('Sector')
plt.tight_layout()
plt.show()
```

### Explanation of Execution

The tables `focus_groups` and `dimensions` are populated first, creating the master lists of sectors and evaluation criteria. Then, the `focus_group_dimension_weights` table is filled with data from the attachment. Each row explicitly links a `focus_group_id` (sector) to a `dimension_id` and assigns a `weight`. This effectively parametrizes the evaluation model for each sector.

The output shows a DataFrame of the manufacturing sector's dimension weights and a bar chart visualizing their relative importance. This helps us confirm the configuration is correctly loaded and provides immediate insight into the manufacturing sector's priorities. The total weight per sector is also displayed, reinforcing the constraint that weights should sum to 1.0. Finally, a heatmap provides a powerful comparative view of dimension weights across all seven PE sectors, clearly illustrating how priorities differ—for instance, 'AI Governance' is more critical in Financial Services, while 'Data Infrastructure' is prominent in Energy.

---

## 4. Task 2.3: Seeding Sector Calibrations

### Story + Context + Real-World Relevance

Beyond dimension weights, each sector often has specific numeric or threshold parameters that calibrate how insights are generated or interpreted. As a Data Engineer, I must seed these `focus_group_calibrations` into the database. These calibrations are crucial for nuanced analysis within each sector. For example, a 'h_r_baseline' might represent a target performance score for a sector, while an 'ebitda_multiplier' could adjust financial projections based on sector-specific risk or growth factors. Storing these as data rows allows for dynamic updates without altering code, aligning with our configuration-driven philosophy.

```python
def seed_sector_calibrations():
    """
    Seeds sector-specific calibration parameters.
    """
    calibrations_data = [
        # Manufacturing
        {'calibration_id': None, 'focus_group_id': 'pe_manufacturing', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('72'), 'parameter_type': 'numeric', 'description': 'Systematic opportunity baseline', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_manufacturing', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('0.90'), 'parameter_type': 'numeric', 'description': 'Conservative EBITDA attribution', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_manufacturing', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.20'), 'parameter_type': 'threshold', 'description': 'Lower due to talent scarcity', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_manufacturing', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Financial Services
        {'calibration_id': None, 'focus_group_id': 'pe_financial_services', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('82'), 'parameter_type': 'numeric', 'description': 'Higher due to data maturity', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_financial_services', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('1.10'), 'parameter_type': 'numeric', 'description': 'Higher AI leverage', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_financial_services', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.25'), 'parameter_type': 'threshold', 'description': 'Standard threshold', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_financial_services', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_financial_services', 'parameter_name': 'governance_minimum', 'parameter_value': Decimal('60'), 'parameter_type': 'threshold', 'description': 'Min governance for approval', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Healthcare
        {'calibration_id': None, 'focus_group_id': 'pe_healthcare', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('78'), 'parameter_type': 'numeric', 'description': 'Moderate with growth potential', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_healthcare', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('1.00'), 'parameter_type': 'numeric', 'description': 'Standard attribution', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_healthcare', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.25'), 'parameter_type': 'threshold', 'description': 'Standard threshold', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_healthcare', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_healthcare', 'parameter_name': 'governance_minimum', 'parameter_value': Decimal('65'), 'parameter_type': 'threshold', 'description': 'Higher governance requirement', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Technology
        {'calibration_id': None, 'focus_group_id': 'pe_technology', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('85'), 'parameter_type': 'numeric', 'description': 'Highest - AI native', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_technology', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('1.15'), 'parameter_type': 'numeric', 'description': 'Strong AI leverage', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_technology', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.30'), 'parameter_type': 'threshold', 'description': 'Higher talent expected', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_technology', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Retail
        {'calibration_id': None, 'focus_group_id': 'pe_retail', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('75'), 'parameter_type': 'numeric', 'description': 'Growing AI adoption', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_retail', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('1.05'), 'parameter_type': 'numeric', 'description': 'Clear personalization ROI', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_retail', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.25'), 'parameter_type': 'threshold', 'description': 'Standard threshold', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_retail', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Energy
        {'calibration_id': None, 'focus_group_id': 'pe_energy', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('68'), 'parameter_type': 'numeric', 'description': 'Lower but high potential', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_energy', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('0.85'), 'parameter_type': 'numeric', 'description': 'Longer payback periods', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_energy', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.20'), 'parameter_type': 'threshold', 'description': 'Lower due to scarcity', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_energy', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        # Professional Services
        {'calibration_id': None, 'focus_group_id': 'pe_professional_services', 'parameter_name': 'h_r_baseline', 'parameter_value': Decimal('76'), 'parameter_type': 'numeric', 'description': 'Knowledge work automation', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_professional_services', 'parameter_name': 'ebitda_multiplier', 'parameter_value': Decimal('1.00'), 'parameter_type': 'numeric', 'description': 'Standard attribution', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_professional_services', 'parameter_name': 'talent_concentration_threshold', 'parameter_value': Decimal('0.25'), 'parameter_type': 'threshold', 'description': 'Standard threshold', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
        {'calibration_id': None, 'focus_group_id': 'pe_professional_services', 'parameter_name': 'position_factor_delta', 'parameter_value': Decimal('0.15'), 'parameter_type': 'numeric', 'description': 'H^R position adjustment', 'effective_from': None, 'effective_to': None, 'is_current': None, 'created_at': None},
    ]
    db.insert_rows('focus_group_calibrations', calibrations_data)

seed_sector_calibrations()

# Fetch and display calibration parameters for a specific sector (e.g., Financial Services)
financial_services_calibrations_query = """
SELECT
    fg.group_name AS sector_name,
    c.parameter_name,
    c.parameter_value,
    c.parameter_type,
    c.description
FROM focus_group_calibrations c
JOIN focus_groups fg ON c.focus_group_id = fg.focus_group_id
WHERE fg.focus_group_id = %(focus_group_id)s
AND c.is_current = TRUE
ORDER BY c.parameter_name;
"""
financial_calibrations = db.fetch_all(financial_services_calibrations_query, {'focus_group_id': 'pe_financial_services'})
df_financial_calibrations = pd.DataFrame(financial_calibrations)
print("\nFinancial Services Sector Calibration Parameters:")
display(df_financial_calibrations)
```

### Explanation of Execution

The `focus_group_calibrations` table is populated with a variety of parameters specific to each sector. For instance, the 'Financial Services' sector has a 'governance_minimum' threshold, while 'Manufacturing' has a 'conservative EBITDA attribution' factor. These values, stored as `DECIMAL(10,4)`, represent specific numerical adjustments or thresholds. The displayed DataFrame for 'Financial Services' shows these parameters in a clear, tabular format, validating that the calibration data has been correctly ingested and is accessible. This allows the PE Org-AI-R platform to apply these precise adjustments in its sector-specific analysis.

---

## 5. Task 2.4: Establishing the Organization Structure

### Story + Context + Real-World Relevance

The core entities our platform analyzes are organizations. As a Data Engineer, designing the `organizations` table is central. A critical requirement is to link each organization to a specific PE sector (`focus_group_id`). This link is the backbone of our configuration-driven approach, enabling us to apply the correct sector-specific weights and calibrations when evaluating an organization. The table also includes common firmographic data that is relevant across all sectors, promoting a single, unified view of core organizational data.

```python
def create_organizations_schema():
    """
    Creates the DDL for the organizations table with a foreign key to focus_groups.
    """
    ddl_statement = """
    CREATE TABLE organizations (
        organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        legal_name VARCHAR(255) NOT NULL,
        display_name VARCHAR(255),
        ticker_symbol VARCHAR(10),
        cik_number VARCHAR(20),
        duns_number VARCHAR(20),
        focus_group_id VARCHAR(50) NOT NULL REFERENCES focus_groups(focus_group_id),
        primary_sic_code VARCHAR(10),
        primary_naics_code VARCHAR(10),
        employee_count INTEGER,
        annual_revenue_usd DECIMAL(15,2),
        founding_year INTEGER,
        headquarters_country VARCHAR(3),
        headquarters_state VARCHAR(50),
        headquarters_city VARCHAR(100),
        website_url VARCHAR(500),
        status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by VARCHAR(100),
        CONSTRAINT chk_org_pe_platform CHECK (focus_group_id LIKE 'pe_%')
    );
    """
    db.execute_ddl(ddl_statement)

    # Indexes for performance
    db.execute_ddl("CREATE INDEX idx_org_focus_group ON organizations(focus_group_id);")
    db.execute_ddl("CREATE INDEX idx_org_ticker ON organizations(ticker_symbol) WHERE ticker_symbol IS NOT NULL;")

create_organizations_schema()

# Seed some dummy organizations
organizations_data = [
    {'organization_id': None, 'legal_name': 'Global Manufacturing Corp', 'display_name': 'Global MFG', 'focus_group_id': 'pe_manufacturing', 'annual_revenue_usd': Decimal('500000000.00'), 'employee_count': 1500, 'created_at': None},
    {'organization_id': None, 'legal_name': 'Future FinTech Solutions', 'display_name': 'Future FinTech', 'focus_group_id': 'pe_financial_services', 'annual_revenue_usd': Decimal('120000000.00'), 'employee_count': 800, 'created_at': None},
    {'organization_id': None, 'legal_name': 'Health Innovations Ltd', 'display_name': 'Health Innov', 'focus_group_id': 'pe_healthcare', 'annual_revenue_usd': Decimal('300000000.00'), 'employee_count': 2000, 'created_at': None},
    {'organization_id': None, 'legal_name': 'Quantum AI Systems', 'display_name': 'Quantum AI', 'focus_group_id': 'pe_technology', 'annual_revenue_usd': Decimal('80000000.00'), 'employee_count': 400, 'created_at': None},
]
db.insert_rows('organizations', organizations_data)

# Display some organizations
df_organizations = pd.DataFrame(db.get_table_data('organizations'))
print("\nSample Organizations:")
display(df_organizations[['organization_id', 'legal_name', 'focus_group_id', 'annual_revenue_usd']])
```

### Explanation of Execution

The `organizations` table is created with a `focus_group_id` column acting as a foreign key to the `focus_groups` table. This is paramount for our "One Schema, Many Configurations" strategy, as it explicitly links each organization to its relevant sector and thus to its specific configuration data (weights and calibrations). The `UUID` primary key ensures unique identifiers for organizations, and the `CHECK` constraint on `focus_group_id` helps enforce data integrity by ensuring only PE-related focus groups are assigned. Seeding a few dummy organizations demonstrates how data would be stored, and the displayed DataFrame confirms the successful creation and population of this critical table.

---

## 6. Task 2.5: Defining Sector-Specific Attributes

### Story + Context + Real-World Relevance

While the `organizations` table holds common attributes, each sector demands unique, granular data points. As a Data Engineer, the decision to use "Queryable Sector Attribute Tables" instead of a generic JSONB column is a strategic one. JSONB columns can become unwieldy for querying and indexing specific attributes across many organizations. By creating separate, typed tables for each sector's attributes, we ensure:
1.  **Strong Typing**: Each attribute has a defined data type, preventing data quality issues.
2.  **Queryability**: Specific attributes can be easily filtered, aggregated, and indexed without complex JSON parsing.
3.  **Performance**: Database optimizations (e.g., indexes) can be applied directly to individual columns.

This task involves creating dedicated attribute tables for each of the 7 PE sectors, each linked to the main `organizations` table via `organization_id`. For example, 'Manufacturing' needs fields like `ot_systems` and `scada_vendor`, while 'Financial Services' requires `regulatory_bodies` and `model_risk_framework`.

```python
def create_sector_attribute_schemas():
    """
    Creates the DDL for sector-specific attribute tables.
    """
    ddl_statements = [
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """,
        """
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
        """
    ]
    for ddl in ddl_statements:
        db.execute_ddl(ddl)

create_sector_attribute_schemas()

# Seed some dummy attribute data
orgs_in_db = db.get_table_data('organizations')
org_id_mfg = next((org['organization_id'] for org in orgs_in_db if org['focus_group_id'] == 'pe_manufacturing'), None)
org_id_fin = next((org['organization_id'] for org in orgs_in_db if org['focus_group_id'] == 'pe_financial_services'), None)
org_id_hc = next((org['organization_id'] for org in orgs_in_db if org['focus_group_id'] == 'pe_healthcare'), None)

if org_id_mfg:
    db.insert_rows('org_attributes_manufacturing', [{
        'organization_id': org_id_mfg,
        'ot_systems': ['SCADA', 'DCS'],
        'it_ot_integration': 'High',
        'plant_count': 5,
        'edge_computing': True,
        'created_at': None
    }])
if org_id_fin:
    db.insert_rows('org_attributes_financial_services', [{
        'organization_id': org_id_fin,
        'regulatory_bodies': ['SEC', 'FINRA'],
        'charter_type': 'Commercial Bank',
        'fraud_detection_ai': True,
        'aum_billions': Decimal('150.75'),
        'created_at': None
    }])
if org_id_hc:
    db.insert_rows('org_attributes_healthcare', [{
        'organization_id': org_id_hc,
        'hipaa_certified': True,
        'ehr_system': 'Epic',
        'bed_count': 300,
        'clinical_ai_deployed': True,
        'created_at': None
    }])

print("\nSample sector-specific attribute data seeded for Manufacturing, Financial Services, and Healthcare.")
```

### Explanation of Execution

This section creates seven distinct attribute tables, one for each PE sector. Each table is designed with typed columns (`VARCHAR`, `INTEGER`, `BOOLEAN`, `DECIMAL`, `VARCHAR[]` for arrays) specific to the needs of that sector, and crucially, they all link back to the `organizations` table using `organization_id` as a primary and foreign key. This "Queryable Sector Attribute Tables" approach offers significant advantages over flexible JSONB fields, particularly for data integrity, query performance, and straightforward data governance. Seeding sample data for a few organizations demonstrates how these attributes are stored, ready for deeper, sector-specific analysis.

---

## 7. Task 2.6: Building the Configuration Service Logic

### Story + Context + Real-World Relevance

Now that our configuration data and organization structures are in place, we need a clean, consistent way for application services to access this information. As a Software Developer, I'll build a `SectorConfigService`. This service will encapsulate the logic for fetching sector-specific weights and calibration parameters from the database, transforming them into easy-to-use Python objects. This abstraction isolates the business logic from direct database interactions, making the system more modular and testable. The `SectorConfig` dataclass will hold the complete configuration for a sector, providing structured access to its unique parameters.

A key aspect of configuration integrity is ensuring that dimension weights for each sector sum to 1.0. This is a fundamental constraint ($ \sum w_{sd} = 1 $ where $w_{sd}$ is the weight of dimension $d$ for sector $s$) to ensure a balanced evaluation. The `SectorConfig` dataclass will include a method to validate this sum.

```python
@dataclass
class SectorConfig:
    """Complete configuration for a PE sector."""
    focus_group_id: str
    group_name: str
    group_code: str
    dimension_weights: Dict[str, Decimal] = field(default_factory=dict)
    calibrations: Dict[str, Decimal] = field(default_factory=dict)

    @property
    def h_r_baseline(self) -> Decimal:
        """Get H^R baseline for this sector."""
        return self.calibrations.get('h_r_baseline', Decimal('75')) # Default value for robustness

    @property
    def ebitda_multiplier(self) -> Decimal:
        """Get EBITDA multiplier for this sector."""
        return self.calibrations.get('ebitda_multiplier', Decimal('1.0'))

    @property
    def position_factor_delta(self) -> Decimal:
        """Get position factor delta (delta) for H^R calculation."""
        # Using delta for the Greek letter delta
        return self.calibrations.get('position_factor_delta', Decimal('0.15'))

    @property
    def talent_concentration_threshold(self) -> Decimal:
        """Get talent concentration threshold."""
        return self.calibrations.get('talent_concentration_threshold', Decimal('0.25'))

    def get_dimension_weight(self, dimension_code: str) -> Decimal:
        """Get weight for a specific dimension."""
        return self.dimension_weights.get(dimension_code, Decimal('0'))

    def validate_weights_sum(self) -> bool:
        """Verify dimension weights sum to 1.0. Allows for slight floating point deviation."""
        total = sum(self.dimension_weights.values())
        return abs(total - Decimal('1.0')) < Decimal('0.001') # Allow small epsilon for floating point

class SectorConfigService:
    """Service for loading sector configurations from the database."""

    def __init__(self, db_client: MockDatabaseClient):
        self._db = db_client

    async def _load_from_db(self, focus_group_id: str) -> Optional[SectorConfig]:
        """Load single configuration from database."""
        # Get base focus group info
        fg_query = """
        SELECT focus_group_id, group_name, group_code
        FROM focus_groups
        WHERE focus_group_id = %(focus_group_id)s
        AND platform = 'pe_org_air'
        AND is_active = TRUE;
        """
        fg_row = self._db.fetch_one(fg_query, {'focus_group_id': focus_group_id})
        if not fg_row:
            return None

        # Get dimension weights
        weights_query = """
        SELECT d.dimension_code, w.weight
        FROM focus_group_dimension_weights w
        JOIN dimensions d ON w.dimension_id = d.dimension_id
        WHERE w.focus_group_id = %(focus_group_id)s AND w.is_current = TRUE
        ORDER BY d.display_order;
        """
        weights_rows = self._db.fetch_all(weights_query, {'focus_group_id': focus_group_id})
        dimension_weights = {
            row['dimension_code']: Decimal(str(row['weight'])) # Convert to Decimal
            for row in weights_rows
        }

        # Get calibrations
        calib_query = """
        SELECT parameter_name, parameter_value
        FROM focus_group_calibrations
        WHERE focus_group_id = %(focus_group_id)s AND is_current = TRUE;
        """
        calib_rows = self._db.fetch_all(calib_query, {'focus_group_id': focus_group_id})
        calibrations = {
            row['parameter_name']: Decimal(str(row['parameter_value'])) # Convert to Decimal
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
            logger.warning("invalid_weights_sum", focus_group_id=focus_group_id, actual_sum=sum(config.dimension_weights.values()))

        return config

    async def _load_all_from_db(self) -> List[SectorConfig]:
        """Load all sector configurations from database."""
        fg_query = """
        SELECT focus_group_id
        FROM focus_groups
        WHERE platform = 'pe_org_air' AND is_active = TRUE
        ORDER BY display_order;
        """
        fg_rows = self._db.fetch_all(fg_query)
        configs = []
        for row in fg_rows:
            config = await self._load_from_db(row['focus_group_id'])
            if config:
                configs.append(config)
        return configs

    def _config_to_dict(self, config: SectorConfig) -> dict:
        """Convert config to dict for caching, ensuring Decimal is stringified."""
        return {
            'focus_group_id': config.focus_group_id,
            'group_name': config.group_name,
            'group_code': config.group_code,
            'dimension_weights': {k: str(v) for k, v in config.dimension_weights.items()},
            'calibrations': {k: str(v) for k, v in config.calibrations.items()},
        }

    def _dict_to_config(self, data: dict) -> SectorConfig:
        """Convert cached dict back to config, ensuring Decimal conversion."""
        return SectorConfig(
            focus_group_id=data['focus_group_id'],
            group_name=data['group_name'],
            group_code=data['group_code'],
            dimension_weights={k: Decimal(v) for k, v in data['dimension_weights'].items()},
            calibrations={k: Decimal(v) for k, v in data['calibrations'].items()},
        )

# Initialize the service (without caching for now)
sector_service_no_cache = SectorConfigService(db)

# Demonstrate loading a single sector config
import asyncio # Required for async functions in notebook

async def get_and_display_config(service: SectorConfigService, sector_id: str):
    config = await service._load_from_db(sector_id)
    if config:
        print(f"\n--- Configuration for {config.group_name} ({config.focus_group_id}) ---")
        print(f"H^R Baseline: {config.h_r_baseline}")
        print(f"EBITDA Multiplier: {config.ebitda_multiplier}")
        print(f"Talent Concentration Threshold: {config.talent_concentration_threshold}")
        print("\nDimension Weights:")
        for dim, weight in config.dimension_weights.items():
            print(f"  - {dim}: {weight}")
        print(f"Weights sum to 1.0: {config.validate_weights_sum()}")
        print("\nCalibrations (all):")
        for param, value in config.calibrations.items():
            print(f"  - {param}: {value}")
    else:
        print(f"Configuration not found for sector_id: {sector_id}")

await get_and_display_config(sector_service_no_cache, 'pe_manufacturing')
await get_and_display_config(sector_service_no_cache, 'pe_financial_services')
```

### Explanation of Execution

The `SectorConfig` dataclass and `SectorConfigService` class are defined here. The `_load_from_db` method within `SectorConfigService` executes SQL queries to fetch the base sector information, its dimension weights, and its calibration parameters. These are then aggregated into a `SectorConfig` object, providing a clean, object-oriented representation of the configuration. Properties like `h_r_baseline` and `ebitda_multiplier` on the `SectorConfig` dataclass demonstrate how specific parameters can be accessed directly, with sensible default values, preventing `KeyError` if a calibration is missing. The `validate_weights_sum` method ensures the integrity of our dimension weights. Calling `get_and_display_config` for 'Manufacturing' and 'Financial Services' demonstrates that the service correctly retrieves and structures the sector-specific configurations, ready for use by higher-level application logic.

---

## 8. Task 2.7: Implementing the Caching Layer

### Story + Context + Real-World Relevance

While our `SectorConfigService` effectively loads configurations, repeatedly querying the database for frequently accessed sector data can introduce latency and strain database resources. As a Data Engineer, implementing a caching layer (using Redis in a real-world scenario) is a standard optimization technique. The "Configuration Caching" concept means that once a sector's configuration is loaded from the database, it's stored in a fast-access cache for a specified duration (`CACHE_TTL`). Subsequent requests for the same configuration hit the cache, drastically improving response times. We'll also add cache invalidation mechanisms to ensure that updates to configurations are reflected in the cache.

The performance benefit of caching can be significant. Without caching, every request for a configuration involves a database lookup ($T_{\text{db}}$). With caching, most requests will be served from cache ($T_{\text{cache}}$), where $T_{\text{cache}} \ll T_{\text{db}}$. If $R$ is the total number of requests, and $H$ is the cache hit ratio, the total time for fetching configurations can be expressed as:
$$ T_{\text{total}} = (R \cdot H) \cdot T_{\text{cache}} + (R \cdot (1-H)) \cdot (T_{\text{cache}} + T_{\text{db}}) $$
This formula illustrates that a higher hit ratio ($H$) dramatically reduces the number of expensive database calls, improving overall system responsiveness.

```python
# Constants for caching
CACHE_KEY_SECTOR = "sector:{focus_group_id}"
CACHE_KEY_ALL = "sectors:all"
CACHE_TTL = 3600 # 1 hour

class SectorConfigServiceWithCache(SectorConfigService):
    """SectorConfigService augmented with Redis caching capabilities."""
    def __init__(self, db_client: MockDatabaseClient, cache_client: MockRedisClient):
        super().__init__(db_client)
        self._cache = cache_client

    async def get_config(self, focus_group_id: str) -> Optional[SectorConfig]:
        """Get configuration for a single sector, using cache."""
        cache_key = CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)

        # Check cache
        cached_data = self._cache.get(cache_key)
        if cached_data:
            return self._dict_to_config(json.loads(cached_data)) # Deserialize from JSON string

        # Load from database if not in cache
        config = await self._load_from_db(focus_group_id)
        if config:
            # Store in cache
            self._cache.setex(cache_key, CACHE_TTL, json.dumps(self._config_to_dict(config)))
        return config

    async def get_all_configs(self) -> List[SectorConfig]:
        """Get all PE sector configurations, using cache."""
        cache_key = CACHE_KEY_ALL

        # Check cache for all configs
        cached_data = self._cache.get(cache_key)
        if cached_data:
            # Deserialize list of dicts and convert each to SectorConfig
            return [self._dict_to_config(c) for c in json.loads(cached_data)]

        # Load all from database if not in cache
        configs = await self._load_all_from_db()
        if configs:
            # Store list of dicts in cache
            self._cache.setex(cache_key, CACHE_TTL, json.dumps([self._config_to_dict(c) for c in configs]))
        return configs

    def invalidate_cache(self, focus_group_id: Optional[str] = None) -> None:
        """Invalidate cached configurations. If no ID, invalidate all."""
        if focus_group_id:
            self._cache.delete(CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id))
            logger.info("sector_cache_invalidated", focus_group_id=focus_group_id)
        else:
            self._cache.invalidate_pattern(CACHE_KEY_SECTOR.replace('{focus_group_id}', '*'))
            self._cache.delete(CACHE_KEY_ALL)
            logger.info("sector_cache_invalidated", pattern="all")


# Initialize the service with caching
sector_service = SectorConfigServiceWithCache(db, cache)

# Demonstrate cache behavior
async def demonstrate_caching():
    print("--- First call for Manufacturing (should be a cache MISS) ---")
    mfg_config_1 = await sector_service.get_config('pe_manufacturing')
    print(f"Retrieved config (1): {mfg_config_1.group_name}, h_r_baseline={mfg_config_1.h_r_baseline}")

    print("\n--- Second call for Manufacturing (should be a cache HIT) ---")
    mfg_config_2 = await sector_service.get_config('pe_manufacturing')
    print(f"Retrieved config (2): {mfg_config_2.group_name}, h_r_baseline={mfg_config_2.h_r_baseline}")

    print("\n--- Invalidate cache for Manufacturing ---")
    sector_service.invalidate_cache('pe_manufacturing')

    print("\n--- Third call for Manufacturing (should be a cache MISS after invalidation) ---")
    mfg_config_3 = await sector_service.get_config('pe_manufacturing')
    print(f"Retrieved config (3): {mfg_config_3.group_name}, h_r_baseline={mfg_config_3.h_r_baseline}")

    print("\n--- First call for ALL sectors (should be a cache MISS) ---")
    all_configs_1 = await sector_service.get_all_configs()
    print(f"Retrieved {len(all_configs_1)} configs (1st call).")

    print("\n--- Second call for ALL sectors (should be a cache HIT) ---")
    all_configs_2 = await sector_service.get_all_configs()
    print(f"Retrieved {len(all_configs_2)} configs (2nd call).")

    print("\n--- Invalidate ALL cache ---")
    sector_service.invalidate_cache()

    print("\n--- Third call for ALL sectors (should be a cache MISS after invalidation) ---")
    all_configs_3 = await sector_service.get_all_configs()
    print(f"Retrieved {len(all_configs_3)} configs (3rd call).")

await demonstrate_caching()
```

### Explanation of Execution

The `SectorConfigService` is extended to `SectorConfigServiceWithCache`, integrating the `MockRedisClient` for caching. The `get_config` and `get_all_configs` methods now first attempt to retrieve configurations from the cache using specific `CACHE_KEY_SECTOR` or `CACHE_KEY_ALL` patterns. If a cache miss occurs, the data is fetched from the database and then stored in the cache with a `CACHE_TTL` (Time To Live) of 1 hour. This ensures that frequently accessed configurations are served quickly.

The demonstration clearly shows the sequence of cache hits and misses:
-   The first call for 'Manufacturing' results in a cache MISS (data loaded from DB).
-   The second call for 'Manufacturing' results in a cache HIT (data served from cache).
-   After explicitly calling `invalidate_cache` for 'Manufacturing', the third call correctly results in another cache MISS, proving the invalidation mechanism works.
-   Similar behavior is observed for `get_all_configs`, demonstrating caching for multiple configurations.

This cache implementation significantly improves the performance of the PE Org-AI-R platform, reducing load on the PostgreSQL database and providing faster insights to users.

---

## 9. Task 2.8: Creating the Unified Organization View

### Story + Context + Real-World Relevance

The final piece of our data architecture is to provide a comprehensive, easy-to-query view of organizations, combining their core details with all sector-specific attributes. As a Data Engineer, creating a "Unified Organization View" using a SQL `VIEW` is the most effective approach. This view will `JOIN` the `organizations` table with `focus_groups` and all seven `org_attributes` tables. This way, data analysts and other services can query a single logical entity (`vw_organizations_full`) without needing to understand the complex underlying table structure or perform multiple joins themselves. This abstraction simplifies data access and ensures consistency.

```python
def create_unified_organization_view():
    """
    Creates a SQL VIEW that unifies organization data with sector-specific attributes.
    """
    view_ddl = """
    CREATE OR REPLACE VIEW vw_organizations_full AS
    SELECT
        o.*,
        fg.group_name AS sector_name,
        fg.group_code AS sector_code,
        -- Manufacturing attributes
        mfg.ot_systems, mfg.it_ot_integration, mfg.scada_vendor, mfg.mes_system,
        mfg.plant_count, mfg.automation_level, mfg.iot_platforms, mfg.digital_twin_status,
        mfg.edge_computing, mfg.supply_chain_visibility, mfg.demand_forecasting_ai,
        -- Financial Services attributes
        fin.regulatory_bodies, fin.charter_type, fin.model_risk_framework, fin.mrm_team_size,
        fin.model_inventory_count, fin.algo_trading, fin.fraud_detection_ai, fin.credit_ai,
        fin.aml_ai, fin.aum_billions, fin.total_assets_billions,
        -- Healthcare attributes
        hc.hipaa_certified, hc.hitrust_certified, hc.fda_clearances, hc.fda_clearance_count,
        hc.ehr_system, hc.ehr_integration_level, hc.fhir_enabled, hc.clinical_ai_deployed,
        hc.imaging_ai, hc.org_type, hc.bed_count,
        -- Technology attributes
        tech.tech_category, tech.primary_language, tech.cloud_native, tech.github_org,
        tech.github_stars_total, tech.open_source_projects, tech.ml_platform, tech.llm_integration,
        tech.ai_product_features, tech.gpu_infrastructure,
        -- Retail & Consumer attributes
        rtl.retail_type, rtl.store_count, rtl.ecommerce_pct, rtl.cdp_vendor,
        rtl.loyalty_program, rtl.loyalty_members, rtl.personalization_ai, rtl.recommendation_engine,
        rtl.demand_forecasting,
        -- Energy & Utilities attributes
        enr.energy_type, enr.regulated, enr.scada_systems, enr.ami_deployed,
        enr.smart_grid_pct, enr.generation_capacity_mw, enr.grid_optimization_ai,
        enr.predictive_maintenance, enr.renewable_pct,
        -- Professional Services attributes
        ps.firm_type, ps.partnership_model, ps.partner_count, ps.professional_staff,
        ps.km_system, ps.document_ai, ps.knowledge_graph, ps.client_ai_services,
        ps.internal_ai_tools
    FROM organizations o
    JOIN focus_groups fg ON o.focus_group_id = fg.focus_group_id
    LEFT JOIN org_attributes_manufacturing mfg ON o.organization_id = mfg.organization_id
    LEFT JOIN org_attributes_financial_services fin ON o.organization_id = fin.organization_id
    LEFT JOIN org_attributes_healthcare hc ON o.organization_id = hc.organization_id
    LEFT JOIN org_attributes_technology tech ON o.organization_id = tech.organization_id
    LEFT JOIN org_attributes_retail rtl ON o.organization_id = rtl.organization_id
    LEFT JOIN org_attributes_energy enr ON o.organization_id = enr.organization_id
    LEFT JOIN org_attributes_professional_services ps ON o.organization_id = ps.organization_id;
    """
    db.execute_ddl(view_ddl)

create_unified_organization_view()

# Query the unified view (simulated by joining tables directly in mock client)
# Since MockDatabaseClient doesn't truly parse views, we simulate the join logic for demonstration
def query_unified_view(db_client: MockDatabaseClient, focus_group_id: Optional[str] = None):
    orgs = db_client.get_table_data('organizations')
    focus_groups = {fg['focus_group_id']: fg for fg in db_client.get_table_data('focus_groups')}
    
    # Fetch all attribute data once to make simulation simpler
    attrs_mfg = {a['organization_id']: a for a in db_client.get_table_data('org_attributes_manufacturing')}
    attrs_fin = {a['organization_id']: a for a in db_client.get_table_data('org_attributes_financial_services')}
    attrs_hc = {a['organization_id']: a for a in db_client.get_table_data('org_attributes_healthcare')}
    # Add others as needed for a more complete simulation

    results = []
    for org in orgs:
        if focus_group_id and org['focus_group_id'] != focus_group_id:
            continue
        
        merged_row = org.copy()
        merged_row['sector_name'] = focus_groups.get(org['focus_group_id'], {}).get('group_name')
        merged_row['sector_code'] = focus_groups.get(org['focus_group_id'], {}).get('group_code')

        # Simulate LEFT JOIN for attributes
        merged_row.update(attrs_mfg.get(org['organization_id'], {}))
        merged_row.update(attrs_fin.get(org['organization_id'], {}))
        merged_row.update(attrs_hc.get(org['organization_id'], {}))
        # Add updates for other sectors

        results.append(merged_row)
    
    # Filter out redundant columns if present due to update merging
    filtered_results = []
    for row in results:
        clean_row = {}
        for k, v in row.items():
            # Exclude specific attribute table FKs like 'organization_id' from attribute tables
            if k == 'organization_id' and (row.get('focus_group_id') == 'pe_manufacturing' and k in attrs_mfg.get(row['organization_id'], {})):
                continue
            if k == 'organization_id' and (row.get('focus_group_id') == 'pe_financial_services' and k in attrs_fin.get(row['organization_id'], {})):
                continue
            if k == 'organization_id' and (row.get('focus_group_id') == 'pe_healthcare' and k in attrs_hc.get(row['organization_id'], {})):
                continue
            clean_row[k] = v
        filtered_results.append(clean_row)

    return filtered_results

# Display data from the unified view (all organizations)
print("\n--- All Organizations from Unified View ---")
df_unified_all = pd.DataFrame(query_unified_view(db))
display(df_unified_all[['legal_name', 'sector_name', 'annual_revenue_usd', 'plant_count', 'aum_billions', 'bed_count']])

# Display data from the unified view (filtered by a specific sector, e.g., Financial Services)
print("\n--- Financial Services Organizations from Unified View ---")
df_unified_fin = pd.DataFrame(query_unified_view(db, focus_group_id='pe_financial_services'))
display(df_unified_fin[['legal_name', 'sector_name', 'aum_billions', 'fraud_detection_ai', 'regulatory_bodies']])
```

### Explanation of Execution

The `CREATE OR REPLACE VIEW vw_organizations_full` SQL statement is defined to join the `organizations` table with `focus_groups` and all the `org_attributes_` tables using `LEFT JOIN`. This ensures that even organizations without specific attribute entries in a particular sector's table are still included in the view.

The `query_unified_view` Python function simulates querying this view from the `MockDatabaseClient`. It manually performs the necessary joins and aggregations, mimicking what a real database would do when the `vw_organizations_full` view is queried.

The outputs demonstrate:
1.  **A complete view of all organizations**: Showing their general information alongside available sector-specific attributes (e.g., `plant_count` for Manufacturing, `aum_billions` for Financial Services). Notice that many attribute columns will be `None` for organizations not belonging to that specific sector, as expected with `LEFT JOIN` and sparse attribute tables.
2.  **Filtered view for a specific sector (Financial Services)**: This highlights how easily an analyst can now get all relevant data for a particular sector, including `aum_billions` (Assets Under Management) and `fraud_detection_ai` status, directly from a single, clean view.

This unified view drastically simplifies data access for downstream analytics, reporting, and other services on the PE Org-AI-R platform, fulfilling the requirement for easily queryable sector-specific data.

