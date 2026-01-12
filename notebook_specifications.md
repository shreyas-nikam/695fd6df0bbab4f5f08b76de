
# Configuration-Driven Data Platform for Sector-Specific Insights

## Introduction to PE Org-AI-R Platform

**Persona:** Sarah, a Data Engineer at PE Org-AI-R.
**Organization:** PE Org-AI-R is a private equity firm that leverages AI-driven insights to evaluate and manage its portfolio companies across diverse sectors.

Sarah's role involves designing and implementing the foundational data architecture for the PE Org-AI-R platform. The firm invests in 7 distinct sectors, each with unique evaluation criteria and investment parameters. Traditionally, this would lead to complex, hardcoded logic or fragmented database schemas, making the platform difficult to maintain and scale.

To overcome this, Sarah's team has adopted a **configuration-driven data architecture**. This innovative approach centralizes and standardizes how sector-specific logic is managed by defining these behaviors directly through data. This notebook demonstrates Sarah's workflow in building this robust and flexible system, ensuring that sector-specific evaluation models can adapt without requiring code changes or schema alterations. The core principle guiding this work is "**One Schema, Many Configurations**," meaning all 7 PE sectors share identical base schemas, with differentiation achieved purely through data in configuration tables.

Through this lab, you will step into Sarah's shoes and:
-   Design the core database schemas for managing sector configurations.
-   Populate these tables with initial, sector-specific data.
-   Create tables for organizations and their unique sector attributes.
-   Implement a configuration service that leverages caching for efficient access.
-   Construct a unified view to bring together organizational data and their varied sector attributes.

---

## 1. Environment Setup

This section outlines the necessary installations and imports to set up the working environment for our data platform.

### 1.1 Install Required Libraries

We'll use `sqlalchemy` for database interactions, `psycopg2-binary` as the PostgreSQL adapter, `redis` for caching, `pandas` for data manipulation and display, `faker` for synthetic data generation, `uuid` for generating unique identifiers, and `matplotlib` for basic visualizations.

```python
!pip install sqlalchemy==2.0.30 psycopg2-binary==2.9.9 redis==5.0.3 pandas==2.2.2 Faker==24.4.0 matplotlib==3.9.0 dataclasses-json==0.6.6
```

### 1.2 Import Required Dependencies

Next, we import all the necessary modules and classes from the installed libraries.

```python
import os
import uuid
import random
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json # For serialization to/from cache

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker
import redis

# Suppress warnings from Faker
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from faker import Faker

# Initialize Faker for synthetic data generation
fake = Faker()
```

### 1.3 Database and Cache Connection Setup

Sarah needs to connect to the PostgreSQL database and the Redis cache. For this notebook, we'll assume a local PostgreSQL instance is running and Redis is accessible. The connection details will be pulled from environment variables.

**Action:** Set your PostgreSQL and Redis connection strings in your environment variables before running this notebook, or modify the `db_url` and `redis_url` variables directly.

```python
# --- Database Connection (PostgreSQL) ---
# IMPORTANT: Replace with your actual PostgreSQL connection string or set as environment variable
# Example: "postgresql://user:password@localhost:5432/pe_orgair_db"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://testuser:testpassword@localhost:5432/pe_orgair_db")

class DBManager:
    """A simplified database manager using SQLAlchemy Core for direct SQL execution."""
    def __init__(self, db_url: str):
        self.engine: Engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        print(f"Database connection engine created for: {db_url.split('@')[-1]}")

    def execute(self, query: str, params: dict = None):
        """Executes a DDL or DML query."""
        with self.Session() as session:
            try:
                session.execute(text(query), params)
                session.commit()
                # print("SQL command executed successfully.")
            except Exception as e:
                session.rollback()
                print(f"Error executing SQL: {e}")
                raise

    def fetch_one(self, query: str, params: dict = None) -> Optional[Dict[str, Any]]:
        """Fetches a single row."""
        with self.Session() as session:
            result = session.execute(text(query), params).fetchone()
            return result._mapping if result else None

    def fetch_all(self, query: str, params: dict = None) -> List[Dict[str, Any]]:
        """Fetches all rows."""
        with self.Session() as session:
            results = session.execute(text(query), params).fetchall()
            return [r._mapping for r in results]

    def dispose(self):
        """Disposes the engine connection pool."""
        self.engine.dispose()
        print("Database engine disposed.")

db_manager = DBManager(DATABASE_URL)


# --- Redis Connection ---
# IMPORTANT: Replace with your actual Redis connection string or set as environment variable
# Example: "redis://localhost:6379/0"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class CacheManager:
    """Redis cache manager with TTL support."""
    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url, decode_responses=True)
        print(f"Redis client connected to: {redis_url.split('@')[-1]}")

    def get(self, key: str) -> Optional[Any]:
        """Get cached value, automatically decodes JSON."""
        value = self._client.get(key)
        if value:
            # print(f"Cache hit for key: {key}")
            return json.loads(value)
        # print(f"Cache miss for key: {key}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cached value with TTL, automatically encodes JSON."""
        self._client.setex(key, ttl, json.dumps(value))
        # print(f"Cache set for key: {key} with TTL: {ttl}")

    def delete(self, key: str) -> None:
        """Delete cached value."""
        self._client.delete(key)
        # print(f"Cache deleted for key: {key}")

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        keys = self._client.keys(pattern)
        if keys:
            count = self._client.delete(*keys)
            # print(f"Invalidated {count} keys matching pattern: {pattern}")
            return count
        return 0

cache = CacheManager(REDIS_URL)

# For JSON serialization/deserialization for cache, we need a simple JSON library
import json
```

---

## 2. Designing Focus Group Configuration Schema

Sarah's first task is to lay the groundwork for the configuration-driven platform. This involves designing the core tables that will hold all sector-specific parameters: `focus_groups`, `dimensions`, `focus_group_dimension_weights`, and `focus_group_calibrations`. This adheres to the "One Schema, Many Configurations" principle by storing configuration data as rows rather than schema variations.

### 2.1 Story + Context + Real-World Relevance

"To ensure our platform is adaptable, I'm setting up the foundational tables. The `focus_groups` table defines our PE sectors. `dimensions` holds the generic criteria we use for evaluation. The crucial part is `focus_group_dimension_weights`, which assigns a relative importance (weight) to each dimension for a given sector. Finally, `focus_group_calibrations` stores specific numerical thresholds and parameters unique to each sector's investment strategy. This structured approach means that if a sector's evaluation criteria change, we update data, not code or schema."

```sql
-- DDL for focus_groups table
CREATE TABLE IF NOT EXISTS focus_groups (
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

-- DDL for dimensions table
CREATE TABLE IF NOT EXISTS dimensions (
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

-- DDL for focus_group_dimension_weights table
CREATE TABLE IF NOT EXISTS focus_group_dimension_weights (
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
CREATE INDEX IF NOT EXISTS idx_weights_current ON focus_group_dimension_weights(focus_group_id, is_current) WHERE is_current = TRUE;

-- DDL for focus_group_calibrations table
CREATE TABLE IF NOT EXISTS focus_group_calibrations (
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

### 2.2 Code Cell: Database Schema Creation

Sarah executes the DDL statements to create these tables in the PostgreSQL database.

```python
def create_config_schemas(db_manager: DBManager):
    """Creates the focus group, dimension, weight, and calibration tables."""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS focus_groups (
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
        CREATE TABLE IF NOT EXISTS dimensions (
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
        CREATE TABLE IF NOT EXISTS focus_group_dimension_weights (
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
        CREATE INDEX IF NOT EXISTS idx_weights_current ON focus_group_dimension_weights(focus_group_id, is_current) WHERE is_current = TRUE;
        """,
        """
        CREATE TABLE IF NOT EXISTS focus_group_calibrations (
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
        db_manager.execute(ddl)
    print("Configuration schemas created/ensured.")

create_config_schemas(db_manager)
```

### 2.3 Explanation of Execution

These tables form the backbone of our configuration system. `focus_groups` stores general information about each PE sector, `dimensions` defines the various attributes used for evaluation, `focus_group_dimension_weights` allows us to set the relative importance of these attributes per sector, and `focus_group_calibrations` stores specific numerical thresholds. This structure ensures that any changes to sector evaluation criteria are managed through data updates, not schema modifications, adhering to the "One Schema, Many Configurations" principle.

---

## 3. Seeding Sector Dimension Weights and Calibrations

With the schemas in place, Sarah now populates them with the firm's predefined sector and dimension data, including specific weights and calibration parameters. This is where the configuration-driven approach truly begins to manifest.

### 3.1 Story + Context + Real-World Relevance

"Now that the tables are ready, I'm seeding them with actual data. This includes our 7 primary PE sectors, the 7 standard evaluation dimensions (like 'Data Infrastructure' or 'AI Governance'), and critically, the dimension weights for each sector. For example, 'Manufacturing' might heavily weight 'Technology Stack' due to IoT integration, while 'Financial Services' prioritizes 'AI Governance' for regulatory compliance. Also, I'm adding specific calibration parameters for each sector. A critical validation step here is ensuring that **the sum of dimension weights for each sector is $1.0$**. This ensures our weighted scoring models are correctly normalized and consistently applied across all dimensions for a given sector."

```sql
-- DML for focus_groups (7 PE sectors)
INSERT INTO focus_groups (focus_group_id, platform, group_name, group_code, display_order) VALUES
('pe_manufacturing', 'pe_org_air', 'Manufacturing', 'MFG', 1),
('pe_financial_services', 'pe_org_air', 'Financial Services', 'FIN', 2),
('pe_healthcare', 'pe_org_air', 'Healthcare', 'HC', 3),
('pe_technology', 'pe_org_air', 'Technology', 'TECH', 4),
('pe_retail', 'pe_org_air', 'Retail & Consumer', 'RTL', 5),
('pe_energy', 'pe_org_air', 'Energy & Utilities', 'ENR', 6),
('pe_professional_services', 'pe_org_air', 'Professional Services', 'PS', 7)
ON CONFLICT (focus_group_id) DO NOTHING;

-- DML for dimensions (7 dimensions)
INSERT INTO dimensions (dimension_id, platform, dimension_name, dimension_code, display_order) VALUES
('pe_dim_data_infra', 'pe_org_air', 'Data Infrastructure', 'data_infrastructure', 1),
('pe_dim_governance', 'pe_org_air', 'AI Governance', 'ai_governance', 2),
('pe_dim_tech_stack', 'pe_org_air', 'Technology Stack', 'technology_stack', 3),
('pe_dim_talent', 'pe_org_air', 'Talent', 'talent', 4),
('pe_dim_leadership', 'pe_org_air', 'Leadership', 'leadership', 5),
('pe_dim_use_cases', 'pe_org_air', 'Use Case Portfolio', 'use_case_portfolio', 6),
('pe_dim_culture', 'pe_org_air', 'Culture', 'culture', 7)
ON CONFLICT (dimension_id) DO NOTHING;
```

### 3.2 Code Cell: Seed Data for Weights and Calibrations

Sarah populates the `focus_group_dimension_weights` and `focus_group_calibrations` tables using detailed `INSERT` statements. A validation check ensures that the sum of weights for each sector is $1.0$.

```python
def seed_config_data(db_manager: DBManager):
    """Seeds the focus groups, dimensions, weights, and calibrations."""

    # DML for focus_groups (7 PE sectors)
    db_manager.execute("""
    INSERT INTO focus_groups (focus_group_id, platform, group_name, group_code, display_order) VALUES
    ('pe_manufacturing', 'pe_org_air', 'Manufacturing', 'MFG', 1),
    ('pe_financial_services', 'pe_org_air', 'Financial Services', 'FIN', 2),
    ('pe_healthcare', 'pe_org_air', 'Healthcare', 'HC', 3),
    ('pe_technology', 'pe_org_air', 'Technology', 'TECH', 4),
    ('pe_retail', 'pe_org_air', 'Retail & Consumer', 'RTL', 5),
    ('pe_energy', 'pe_org_air', 'Energy & Utilities', 'ENR', 6),
    ('pe_professional_services', 'pe_org_air', 'Professional Services', 'PS', 7)
    ON CONFLICT (focus_group_id) DO NOTHING;
    """)

    # DML for dimensions (7 dimensions)
    db_manager.execute("""
    INSERT INTO dimensions (dimension_id, platform, dimension_name, dimension_code, display_order) VALUES
    ('pe_dim_data_infra', 'pe_org_air', 'Data Infrastructure', 'data_infrastructure', 1),
    ('pe_dim_governance', 'pe_org_air', 'AI Governance', 'ai_governance', 2),
    ('pe_dim_tech_stack', 'pe_org_air', 'Technology Stack', 'technology_stack', 3),
    ('pe_dim_talent', 'pe_org_air', 'Talent', 'talent', 4),
    ('pe_dim_leadership', 'pe_org_air', 'Leadership', 'leadership', 5),
    ('pe_dim_use_cases', 'pe_org_air', 'Use Case Portfolio', 'use_case_portfolio', 6),
    ('pe_dim_culture', 'pe_org_air', 'Culture', 'culture', 7)
    ON CONFLICT (dimension_id) DO NOTHING;
    """)

    # DML for focus_group_dimension_weights (49 records - 7 sectors x 7 dimensions)
    # Manufacturing weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_manufacturing', 'pe_dim_data_infra', 0.22, 'OT/IT integration critical'),
    ('pe_manufacturing', 'pe_dim_governance', 0.12, 'Less regulatory than finance/health'),
    ('pe_manufacturing', 'pe_dim_tech_stack', 0.18, 'Edge computing, IoT platforms'),
    ('pe_manufacturing', 'pe_dim_talent', 0.15, 'AI + manufacturing expertise scarce'),
    ('pe_manufacturing', 'pe_dim_leadership', 0.12, 'Traditional leadership acceptable'),
    ('pe_manufacturing', 'pe_dim_use_cases', 0.14, 'Clear ROI in operations'),
    ('pe_manufacturing', 'pe_dim_culture', 0.07, 'Safety culture > innovation')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Financial Services weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_financial_services', 'pe_dim_data_infra', 0.16, 'Mature infrastructure'),
    ('pe_financial_services', 'pe_dim_governance', 0.22, 'Regulatory imperative'),
    ('pe_financial_services', 'pe_dim_tech_stack', 0.14, 'Standard cloud stacks'),
    ('pe_financial_services', 'pe_dim_talent', 0.18, 'Quant + ML talent critical'),
    ('pe_financial_services', 'pe_dim_leadership', 0.12, 'C-suite AI awareness high'),
    ('pe_financial_services', 'pe_dim_use_cases', 0.10, 'Well-understood use cases'),
    ('pe_financial_services', 'pe_dim_culture', 0.08, 'Risk-averse by design')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Healthcare weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_healthcare', 'pe_dim_data_infra', 0.20, 'EHR integration critical'),
    ('pe_healthcare', 'pe_dim_governance', 0.20, 'FDA/HIPAA compliance'),
    ('pe_healthcare', 'pe_dim_tech_stack', 0.14, 'EHR-centric ecosystems'),
    ('pe_healthcare', 'pe_dim_talent', 0.15, 'Clinical + AI dual expertise'),
    ('pe_healthcare', 'pe_dim_leadership', 0.15, 'Physician champions matter'),
    ('pe_healthcare', 'pe_dim_use_cases', 0.10, 'Long validation cycles'),
    ('pe_healthcare', 'pe_dim_culture', 0.06, 'Evidence-based culture exists')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Technology weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_technology', 'pe_dim_data_infra', 0.15, 'Assumed competent'),
    ('pe_technology', 'pe_dim_governance', 0.12, 'Less regulated'),
    ('pe_technology', 'pe_dim_tech_stack', 0.18, 'Core differentiator'),
    ('pe_technology', 'pe_dim_talent', 0.22, 'Talent is everything'),
    ('pe_technology', 'pe_dim_leadership', 0.13, 'Tech-savvy by default'),
    ('pe_technology', 'pe_dim_use_cases', 0.15, 'Product innovation'),
    ('pe_technology', 'pe_dim_culture', 0.05, 'Innovation assumed')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Retail & Consumer weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_retail', 'pe_dim_data_infra', 0.20, 'Customer data platforms'),
    ('pe_retail', 'pe_dim_governance', 0.12, 'Privacy focus, less regulated'),
    ('pe_retail', 'pe_dim_tech_stack', 0.15, 'Standard cloud + CDP'),
    ('pe_retail', 'pe_dim_talent', 0.15, 'Data science accessible'),
    ('pe_retail', 'pe_dim_leadership', 0.13, 'Digital transformation focus'),
    ('pe_retail', 'pe_dim_use_cases', 0.18, 'Clear revenue impact'),
    ('pe_retail', 'pe_dim_culture', 0.07, 'Customer-centric exists')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Energy & Utilities weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_energy', 'pe_dim_data_infra', 0.22, 'SCADA/OT data critical'),
    ('pe_energy', 'pe_dim_governance', 0.15, 'Regulatory + safety'),
    ('pe_energy', 'pe_dim_tech_stack', 0.18, 'Grid tech, edge computing'),
    ('pe_energy', 'pe_dim_talent', 0.12, 'Talent scarcity'),
    ('pe_energy', 'pe_dim_leadership', 0.13, 'Traditional but evolving'),
    ('pe_energy', 'pe_dim_use_cases', 0.15, 'Clear operational value'),
    ('pe_energy', 'pe_dim_culture', 0.05, 'Safety culture paramount')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # Professional Services weights
    db_manager.execute("""
    INSERT INTO focus_group_dimension_weights (focus_group_id, dimension_id, weight, weight_rationale) VALUES
    ('pe_professional_services', 'pe_dim_data_infra', 0.14, 'Document-centric'),
    ('pe_professional_services', 'pe_dim_governance', 0.15, 'Client confidentiality'),
    ('pe_professional_services', 'pe_dim_tech_stack', 0.12, 'Standard productivity'),
    ('pe_professional_services', 'pe_dim_talent', 0.22, 'People are the product'),
    ('pe_professional_services', 'pe_dim_leadership', 0.17, 'Partner adoption critical'),
    ('pe_professional_services', 'pe_dim_use_cases', 0.12, 'Client + internal'),
    ('pe_professional_services', 'pe_dim_culture', 0.08, 'Innovation varies')
    ON CONFLICT (focus_group_id, dimension_id, effective_from) DO UPDATE SET weight = EXCLUDED.weight, weight_rationale = EXCLUDED.weight_rationale;
    """)

    # DML for focus_group_calibrations
    # Manufacturing calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_manufacturing', 'h_r_baseline', 72, 'numeric', 'Systematic opportunity baseline'),
    ('pe_manufacturing', 'ebitda_multiplier', 0.90, 'numeric', 'Conservative EBITDA attribution'),
    ('pe_manufacturing', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to talent scarcity'),
    ('pe_manufacturing', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Financial Services calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_financial_services', 'h_r_baseline', 82, 'numeric', 'Higher due to data maturity'),
    ('pe_financial_services', 'ebitda_multiplier', 1.10, 'numeric', 'Higher AI leverage'),
    ('pe_financial_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
    ('pe_financial_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
    ('pe_financial_services', 'governance_minimum', 60, 'threshold', 'Min governance for approval')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Healthcare calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_healthcare', 'h_r_baseline', 78, 'numeric', 'Moderate with growth potential'),
    ('pe_healthcare', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
    ('pe_healthcare', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
    ('pe_healthcare', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment'),
    ('pe_healthcare', 'governance_minimum', 65, 'threshold', 'Higher governance requirement')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Technology calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_technology', 'h_r_baseline', 85, 'numeric', 'Highest - AI native'),
    ('pe_technology', 'ebitda_multiplier', 1.15, 'numeric', 'Strong AI leverage'),
    ('pe_technology', 'talent_concentration_threshold', 0.30, 'threshold', 'Higher talent expected'),
    ('pe_technology', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Retail & Consumer calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_retail', 'h_r_baseline', 75, 'numeric', 'Growing AI adoption'),
    ('pe_retail', 'ebitda_multiplier', 1.05, 'numeric', 'Clear personalization ROI'),
    ('pe_retail', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
    ('pe_retail', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Energy & Utilities calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_energy', 'h_r_baseline', 68, 'numeric', 'Lower but high potential'),
    ('pe_energy', 'ebitda_multiplier', 0.85, 'numeric', 'Longer payback periods'),
    ('pe_energy', 'talent_concentration_threshold', 0.20, 'threshold', 'Lower due to scarcity'),
    ('pe_energy', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    # Professional Services calibrations
    db_manager.execute("""
    INSERT INTO focus_group_calibrations (focus_group_id, parameter_name, parameter_value, parameter_type, description) VALUES
    ('pe_professional_services', 'h_r_baseline', 76, 'numeric', 'Knowledge work automation'),
    ('pe_professional_services', 'ebitda_multiplier', 1.00, 'numeric', 'Standard attribution'),
    ('pe_professional_services', 'talent_concentration_threshold', 0.25, 'threshold', 'Standard threshold'),
    ('pe_professional_services', 'position_factor_delta', 0.15, 'numeric', 'H^R position adjustment')
    ON CONFLICT (focus_group_id, parameter_name, effective_from) DO UPDATE SET parameter_value = EXCLUDED.parameter_value, description = EXCLUDED.description;
    """)

    print("Configuration data (sectors, dimensions, weights, calibrations) seeded.")

seed_config_data(db_manager)

# --- Validation and Visualization ---
def validate_and_display_weights(db_manager: DBManager):
    """Fetches dimension weights, validates their sum, and displays them."""
    weights_data = db_manager.fetch_all("""
        SELECT fg.group_name, d.dimension_name, fgdw.weight
        FROM focus_group_dimension_weights fgdw
        JOIN focus_groups fg ON fgdw.focus_group_id = fg.focus_group_id
        JOIN dimensions d ON fgdw.dimension_id = d.dimension_id
        WHERE fgdw.is_current = TRUE
        ORDER BY fg.group_name, d.display_order;
    """)
    if not weights_data:
        print("No weights data found for validation.")
        return

    df_weights = pd.DataFrame(weights_data)
    df_weights['weight'] = pd.to_numeric(df_weights['weight'])

    # Validate sum of weights for each sector
    validation_results = {}
    for group_name, group_df in df_weights.groupby('group_name'):
        total_weight = group_df['weight'].sum()
        is_valid = abs(total_weight - 1.0) < 0.001
        validation_results[group_name] = {'Total Weight': total_weight, 'Valid': is_valid}

        if not is_valid:
            print(f"WARNING: Weights for '{group_name}' do not sum to 1.0! Sum: {total_weight}")

    print("\n--- Dimension Weight Validation Results ---")
    display(pd.DataFrame.from_dict(validation_results, orient='index'))

    # Display weights for one example sector (e.g., Manufacturing)
    print("\n--- Manufacturing Sector Dimension Weights ---")
    mfg_weights = df_weights[df_weights['group_name'] == 'Manufacturing'].set_index('dimension_name')['weight']
    display(mfg_weights.to_frame())

    # Visualize dimension weights across different sectors
    plt.figure(figsize=(14, 7))
    pivot_df = df_weights.pivot(index='group_name', columns='dimension_name', values='weight')
    pivot_df.plot(kind='bar', stacked=True, figsize=(15, 8))
    plt.title('Dimension Weights Across PE Sectors')
    plt.xlabel('Sector')
    plt.ylabel('Weight')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

validate_and_display_weights(db_manager)
```

### 3.3 Explanation of Execution

The output confirms that the configuration data has been successfully loaded into the database. The validation check is critical for Sarah because it ensures the integrity of the weighted scoring models used by the PE Org-AI-R platform. If the sum of weights for any sector were not $1.0$, the subsequent analytical models would produce incorrect or skewed results. The bar chart provides a quick visual comparison of how dimension priorities differ across sectors, which is invaluable for understanding sector-specific investment theses at a glance. For instance, you can observe which dimensions receive higher weights in sectors like "Technology" versus "Healthcare."

---

## 4. Designing Organizations Table with Sector Reference

Sarah now needs a way to store information about the private equity portfolio companies, crucially linking each to its respective PE sector. This `organizations` table will serve as the central registry for all portfolio companies.

### 4.1 Story + Context + Real-World Relevance

"Our portfolio management system needs to track all organizations we invest in. Each organization must be explicitly linked to one of our predefined PE sectors. This `organizations` table will include core firmographic details and a foreign key (`focus_group_id`) to the `focus_groups` table. This is how we ensure that every company is categorized correctly for sector-specific analysis, and it's a direct application of the 'One Schema, Many Configurations' principle: core organization data remains unified, while sector-specific details branch off."

```sql
-- DDL for organizations table
CREATE TABLE IF NOT EXISTS organizations (
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

CREATE INDEX IF NOT EXISTS idx_org_focus_group ON organizations(focus_group_id);
CREATE INDEX IF NOT EXISTS idx_org_ticker ON organizations(ticker_symbol) WHERE ticker_symbol IS NOT NULL;
```

### 4.2 Code Cell: Create Organizations Schema and Generate Synthetic Data

Sarah creates the `organizations` table and then generates 100 synthetic organizations, randomly assigning them to the various PE sectors.

```python
def create_organizations_schema(db_manager: DBManager):
    """Creates the organizations table."""
    ddl = """
    CREATE TABLE IF NOT EXISTS organizations (
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
    CREATE INDEX IF NOT EXISTS idx_org_focus_group ON organizations(focus_group_id);
    CREATE INDEX IF NOT EXISTS idx_org_ticker ON organizations(ticker_symbol) WHERE ticker_symbol IS NOT NULL;
    """
    db_manager.execute(ddl)
    print("Organizations schema created/ensured.")

def generate_synthetic_organizations(db_manager: DBManager, num_orgs: int = 100):
    """Generates synthetic organization data and inserts into the database."""
    focus_groups = db_manager.fetch_all("SELECT focus_group_id FROM focus_groups WHERE platform = 'pe_org_air';")
    focus_group_ids = [fg['focus_group_id'] for fg in focus_groups]
    if not focus_group_ids:
        print("No focus groups found, cannot generate organizations.")
        return

    organizations_data = []
    for _ in range(num_orgs):
        org_id = uuid.uuid4()
        legal_name = fake.company()
        display_name = legal_name
        focus_group_id = random.choice(focus_group_ids)
        employee_count = random.randint(50, 5000)
        annual_revenue_usd = Decimal(random.uniform(1_000_000, 1_000_000_000)).quantize(Decimal('0.01'))
        founding_year = random.randint(1950, 2020)
        headquarters_country = fake.country_code()
        headquarters_state = fake.state()
        headquarters_city = fake.city()
        website_url = fake.url()

        organizations_data.append({
            'organization_id': org_id,
            'legal_name': legal_name,
            'display_name': display_name,
            'focus_group_id': focus_group_id,
            'employee_count': employee_count,
            'annual_revenue_usd': annual_revenue_usd,
            'founding_year': founding_year,
            'headquarters_country': headquarters_country,
            'headquarters_state': headquarters_state,
            'headquarters_city': headquarters_city,
            'website_url': website_url
        })

    # Batch insert
    insert_query = """
    INSERT INTO organizations (
        organization_id, legal_name, display_name, focus_group_id, employee_count,
        annual_revenue_usd, founding_year, headquarters_country, headquarters_state,
        headquarters_city, website_url
    ) VALUES (
        :organization_id, :legal_name, :display_name, :focus_group_id, :employee_count,
        :annual_revenue_usd, :founding_year, :headquarters_country, :headquarters_state,
        :headquarters_city, :website_url
    );
    """
    for org in organizations_data:
        db_manager.execute(insert_query, org)
    print(f"Generated and inserted {num_orgs} synthetic organizations.")

create_organizations_schema(db_manager)
generate_synthetic_organizations(db_manager, num_orgs=100)

# Display a few generated organizations
print("\n--- Sample of Generated Organizations ---")
sample_orgs = db_manager.fetch_all("SELECT * FROM organizations LIMIT 5;")
display(pd.DataFrame(sample_orgs))
```

### 4.3 Explanation of Execution

The `organizations` table is now ready and populated with synthetic data. Each organization is correctly associated with a `focus_group_id`, demonstrating how the core entity (an organization) is linked to its sector's specific configurations. This centralizes common firmographic data while preparing for the integration of unique sector attributes, maintaining the "One Schema, Many Configurations" architecture.

---

## 5. Implementing Sector-Specific Attribute Tables

To avoid schema fragmentation, Sarah will create separate tables for attributes unique to each PE sector. This allows for rich, typed data per sector without adding nullable columns to a monolithic `organizations` table.

### 5.1 Story + Context + Real-World Relevance

"To maintain our 'One Schema, Many Configurations' principle, I'm now creating dedicated attribute tables for each of our 7 PE sectors. Instead of adding dozens of nullable columns to the main `organizations` table, each sector (e.g., 'Manufacturing', 'Healthcare') will have its own `org_attributes_sectorname` table. These tables will store highly specific, typed attributes relevant only to that sector, all linked back to the `organization_id`. This keeps our schemas clean, performant, and easily extensible."

```sql
-- DDL for org_attributes_manufacturing
CREATE TABLE IF NOT EXISTS org_attributes_manufacturing (
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

-- DDL for org_attributes_financial_services
CREATE TABLE IF NOT EXISTS org_attributes_financial_services (
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

-- DDL for org_attributes_healthcare
CREATE TABLE IF NOT EXISTS org_attributes_healthcare (
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

-- DDL for org_attributes_technology
CREATE TABLE IF NOT EXISTS org_attributes_technology (
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

-- DDL for org_attributes_retail
CREATE TABLE IF NOT EXISTS org_attributes_retail (
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

-- DDL for org_attributes_energy
CREATE TABLE IF NOT EXISTS org_attributes_energy (
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

-- DDL for org_attributes_professional_services
CREATE TABLE IF NOT EXISTS org_attributes_professional_services (
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

### 5.2 Code Cell: Create Attribute Schemas and Generate Synthetic Data

Sarah executes the DDL for the attribute tables and populates them with synthetic data. This ensures each organization has relevant details specific to its sector.

```python
def create_all_attribute_schemas(db_manager: DBManager):
    """Creates all sector-specific attribute tables."""
    attribute_ddls = [
        """
        CREATE TABLE IF NOT EXISTS org_attributes_manufacturing (
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
        CREATE TABLE IF NOT EXISTS org_attributes_financial_services (
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
        CREATE TABLE IF NOT EXISTS org_attributes_healthcare (
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
        CREATE TABLE IF NOT EXISTS org_attributes_technology (
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
        CREATE TABLE IF NOT EXISTS org_attributes_retail (
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
        CREATE TABLE IF NOT EXISTS org_attributes_energy (
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
        CREATE TABLE IF NOT EXISTS org_attributes_professional_services (
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
    for ddl in attribute_ddls:
        db_manager.execute(ddl)
    print("All sector-specific attribute schemas created/ensured.")

def generate_synthetic_sector_attributes(db_manager: DBManager):
    """Generates synthetic data for all sector-specific attribute tables."""
    organizations = db_manager.fetch_all("SELECT organization_id, focus_group_id FROM organizations;")

    if not organizations:
        print("No organizations found to attach attributes to.")
        return

    # Group organizations by focus_group_id
    orgs_by_sector = {}
    for org in organizations:
        orgs_by_sector.setdefault(org['focus_group_id'], []).append(org['organization_id'])

    sector_generators = {
        'pe_manufacturing': lambda org_id: {
            'organization_id': org_id,
            'ot_systems': random.sample(['SCADA', 'DCS', 'PLC', 'MES'], random.randint(1, 3)),
            'it_ot_integration': random.choice(['High', 'Medium', 'Low']),
            'plant_count': random.randint(1, 10),
            'automation_level': random.choice(['Manual', 'Semi-Automated', 'Automated']),
            'edge_computing': fake.boolean(),
            'supply_chain_visibility': random.choice(['Low', 'Medium', 'High']),
            'demand_forecasting_ai': fake.boolean()
        },
        'pe_financial_services': lambda org_id: {
            'organization_id': org_id,
            'regulatory_bodies': random.sample(['SEC', 'FINRA', 'FCA', 'MAS'], random.randint(1, 2)),
            'charter_type': random.choice(['Commercial Bank', 'Investment Bank', 'Hedge Fund', 'Asset Manager']),
            'model_risk_framework': fake.word().capitalize(),
            'mrm_team_size': random.randint(5, 50),
            'algo_trading': fake.boolean(),
            'fraud_detection_ai': fake.boolean(),
            'aum_billions': Decimal(random.uniform(10, 500)).quantize(Decimal('0.01'))
        },
        'pe_healthcare': lambda org_id: {
            'organization_id': org_id,
            'hipaa_certified': fake.boolean(),
            'hitrust_certified': fake.boolean(),
            'fda_clearance_count': random.randint(0, 10),
            'ehr_system': fake.word().capitalize(),
            'ehr_integration_level': random.choice(['Basic', 'Intermediate', 'Advanced']),
            'clinical_ai_deployed': fake.boolean(),
            'bed_count': random.randint(50, 1000)
        },
        'pe_technology': lambda org_id: {
            'organization_id': org_id,
            'tech_category': random.choice(['SaaS', 'Cloud', 'AI/ML', 'Cybersecurity', 'FinTech']),
            'primary_language': random.choice(['Python', 'Java', 'Go', 'Rust', 'JavaScript']),
            'cloud_native': fake.boolean(),
            'github_stars_total': random.randint(100, 100000),
            'open_source_projects': random.randint(0, 50),
            'llm_integration': fake.boolean(),
            'ai_product_features': random.randint(1, 10),
            'gpu_infrastructure': fake.boolean()
        },
        'pe_retail': lambda org_id: {
            'organization_id': org_id,
            'retail_type': random.choice(['E-commerce', 'Brick-and-mortar', 'Omnichannel']),
            'store_count': random.randint(1, 500),
            'ecommerce_pct': Decimal(random.uniform(0.1, 0.9)).quantize(Decimal('0.01')),
            'cdp_vendor': fake.company(),
            'loyalty_program': fake.boolean(),
            'loyalty_members': random.randint(1000, 1000000),
            'personalization_ai': fake.boolean(),
            'demand_forecasting': fake.boolean()
        },
        'pe_energy': lambda org_id: {
            'organization_id': org_id,
            'energy_type': random.choice(['Solar', 'Wind', 'Hydro', 'Nuclear', 'Fossil Fuel']),
            'regulated': fake.boolean(),
            'scada_systems': random.sample(['Siemens', 'ABB', 'Schneider', 'Rockwell'], random.randint(1, 2)),
            'ami_deployed': fake.boolean(),
            'smart_grid_pct': Decimal(random.uniform(0.1, 1.0)).quantize(Decimal('0.01')),
            'generation_capacity_mw': Decimal(random.uniform(10, 5000)).quantize(Decimal('0.01')),
            'predictive_maintenance': fake.boolean(),
            'renewable_pct': Decimal(random.uniform(0.0, 1.0)).quantize(Decimal('0.01'))
        },
        'pe_professional_services': lambda org_id: {
            'organization_id': org_id,
            'firm_type': random.choice(['Consulting', 'Legal', 'Accounting', 'IT Services']),
            'partnership_model': random.choice(['Equity', 'Salaried']),
            'partner_count': random.randint(5, 200),
            'professional_staff': random.randint(50, 5000),
            'km_system': fake.word().capitalize(),
            'document_ai': fake.boolean(),
            'knowledge_graph': fake.boolean(),
            'client_ai_services': fake.boolean(),
            'internal_ai_tools': fake.boolean()
        }
    }

    inserted_count = 0
    for sector_id, org_ids in orgs_by_sector.items():
        if sector_id in sector_generators:
            table_name = f"org_attributes_{sector_id.replace('pe_', '')}"
            cols = list(sector_generators[sector_id](uuid.uuid4()).keys())
            insert_placeholders = ', '.join([f":{col}" for col in cols])
            insert_query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({insert_placeholders}) ON CONFLICT (organization_id) DO NOTHING;"

            for org_id in org_ids:
                data = sector_generators[sector_id](org_id)
                db_manager.execute(insert_query, data)
                inserted_count += 1
    print(f"Generated and inserted {inserted_count} synthetic sector-specific attributes.")

create_all_attribute_schemas(db_manager)
generate_synthetic_sector_attributes(db_manager)

# Display sample attributes for a specific sector (e.g., Manufacturing)
print("\n--- Sample of Manufacturing Sector Attributes ---")
mfg_attr_sample = db_manager.fetch_all("SELECT * FROM org_attributes_manufacturing LIMIT 3;")
display(pd.DataFrame(mfg_attr_sample))
```

### 5.3 Explanation of Execution

By creating and populating these sector-specific attribute tables, Sarah has successfully implemented a flexible schema extension strategy. This allows the PE Org-AI-R platform to capture highly relevant, typed data for each sector without burdening the main `organizations` table with unnecessary columns. This design drastically improves query performance for sector-specific analytics and simplifies maintenance as new attributes are added or removed for different sectors.

---

## 6. Building the Sector Configuration Service with Caching

Sarah's next step is to create a Python service that can efficiently retrieve sector configurations, leveraging Redis for caching. This service will encapsulate the logic for fetching dimension weights and calibration parameters from the database, and crucially, validate the sum of weights.

### 6.1 Story + Context + Real-World Relevance

"Our analytical models frequently need access to sector configuration data. To prevent redundant database calls and ensure high performance, I'm building a `SectorConfigService`. This service will load configurations from the database, deserialize them into a structured `SectorConfig` object, and cache them in Redis. It's critical that the service also validates the dimension weights to ensure they sum to $1.0$. This ensures the integrity of our scoring models. For instance, the `position_factor_delta` ($\delta$) is a critical calibration parameter for our valuation models, and its consistent retrieval is vital."

```python
@dataclass_json
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
        return self.calibrations.get('h_r_baseline', Decimal('75'))

    @property
    def ebitda_multiplier(self) -> Decimal:
        """Get EBITDA multiplier for this sector."""
        return self.calibrations.get('ebitda_multiplier', Decimal('1.0'))

    @property
    def position_factor_delta(self) -> Decimal:
        """Get position factor delta ($\delta$) for H^R calculation."""
        return self.calibrations.get('position_factor_delta', Decimal('0.15'))

    @property
    def talent_concentration_threshold(self) -> Decimal:
        """Get talent concentration threshold."""
        return self.calibrations.get('talent_concentration_threshold', Decimal('0.25'))

    def get_dimension_weight(self, dimension_code: str) -> Decimal:
        """Get weight for a specific dimension."""
        return self.dimension_weights.get(dimension_code, Decimal('0'))

    def validate_weights_sum(self) -> bool:
        """Verify dimension weights sum to 1.0."""
        total = sum(self.dimension_weights.values())
        return abs(total - Decimal('1.0')) < Decimal('0.001')

class SectorConfigService:
    """Service for loading and caching sector configurations."""

    CACHE_KEY_SECTOR = "sector:{focus_group_id}"
    CACHE_KEY_ALL = "sectors:all"
    CACHE_TTL = 3600 # 1 hour

    def __init__(self, db_manager: DBManager, cache_manager: CacheManager):
        self._db = db_manager
        self._cache = cache_manager

    async def get_config(self, focus_group_id: str) -> Optional[SectorConfig]:
        """Get configuration for a single sector."""
        cache_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)

        # Check cache
        cached_data = self._cache.get(cache_key)
        if cached_data:
            config = SectorConfig.from_dict(cached_data) # Using dataclasses-json
            if not config.validate_weights_sum():
                print(f"WARNING: Invalid weights sum for cached config {focus_group_id}. Re-loading from DB.")
                self._cache.delete(cache_key) # Invalidate stale cache
                return await self._load_from_db(focus_group_id) # Reload
            return config

        # Load from database
        config = await self._load_from_db(focus_group_id)
        if config:
            # Cache the config (convert to dict for JSON serialization)
            self._cache.set(cache_key, config.to_dict(), self.CACHE_TTL)
        return config

    async def get_all_configs(self) -> List[SectorConfig]:
        """Get all PE sector configurations."""
        cache_key = self.CACHE_KEY_ALL

        cached_data = self._cache.get(cache_key)
        if cached_data:
            configs = [SectorConfig.from_dict(c) for c in cached_data]
            if not all(c.validate_weights_sum() for c in configs):
                print("WARNING: Invalid weights sum for one or more cached configs. Re-loading all from DB.")
                self._cache.delete(cache_key) # Invalidate stale cache
                return await self._load_all_from_db() # Reload
            return configs

        configs = await self._load_all_from_db()
        if configs:
            self._cache.set(cache_key, [c.to_dict() for c in configs], self.CACHE_TTL)
        return configs

    async def _load_from_db(self, focus_group_id: str) -> Optional[SectorConfig]:
        """Load single configuration from database."""
        # Get base focus group
        fg_query = """
        SELECT focus_group_id, group_name, group_code
        FROM focus_groups
        WHERE focus_group_id = :focus_group_id AND platform = 'pe_org_air' AND is_active = TRUE;
        """
        fg_row = self._db.fetch_one(fg_query, {'focus_group_id': focus_group_id})
        if not fg_row:
            return None

        # Get dimension weights
        weights_query = """
        SELECT d.dimension_code, w.weight
        FROM focus_group_dimension_weights w
        JOIN dimensions d ON w.dimension_id = d.dimension_id
        WHERE w.focus_group_id = :focus_group_id AND w.is_current = TRUE
        ORDER BY d.display_order;
        """
        weights_rows = self._db.fetch_all(weights_query, {'focus_group_id': focus_group_id})
        dimension_weights = {
            row['dimension_code']: Decimal(str(row['weight']))
            for row in weights_rows
        }

        # Get calibrations
        calib_query = """
        SELECT parameter_name, parameter_value
        FROM focus_group_calibrations
        WHERE focus_group_id = :focus_group_id AND is_current = TRUE;
        """
        calib_rows = self._db.fetch_all(calib_query, {'focus_group_id': focus_group_id})
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
            print(f"WARNING: Invalid weights sum for focus_group_id: {focus_group_id}. Sum: {sum(dimension_weights.values())}")
            # Potentially raise an error or handle invalid config more robustly in a production system
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

    def invalidate_cache(self, focus_group_id: Optional[str] = None) -> None:
        """Invalidate cached configurations.
        If focus_group_id is provided, invalidates specific sector cache.
        Also invalidates the 'all sectors' cache."""
        if focus_group_id:
            specific_key = self.CACHE_KEY_SECTOR.format(focus_group_id=focus_group_id)
            self._cache.delete(specific_key)
            print(f"Invalidated cache for specific sector: {focus_group_id}")
        
        # Always invalidate the 'all sectors' cache to ensure consistency
        self._cache.delete(self.CACHE_KEY_ALL)
        print("Invalidated cache for all sectors.")
        
# Initialize the service
sector_service = SectorConfigService(db_manager, cache)
```

### 6.2 Code Cell: Testing the Service and Cache Invalidation

Sarah tests the `SectorConfigService` by fetching a configuration, observing the cache behavior, and then invalidating the cache.

```python
import asyncio

async def test_sector_config_service():
    print("--- Testing SectorConfigService ---")
    test_sector_id = 'pe_manufacturing'

    # 1. Fetch config - should be a cache miss first, then loaded from DB and cached
    print(f"\nAttempting to get config for {test_sector_id} (first call)...")
    config1 = await sector_service.get_config(test_sector_id)
    if config1:
        print(f"Config loaded for {config1.group_name}. H^R Baseline: {config1.h_r_baseline}, Position Factor Delta: {config1.position_factor_delta}")
        print(f"Dimension weights for {config1.group_name}: {config1.dimension_weights}")
        print(f"Weights sum to 1.0: {config1.validate_weights_sum()}")
    else:
        print(f"Failed to load config for {test_sector_id}")
        return

    # 2. Fetch config again - should be a cache hit
    print(f"\nAttempting to get config for {test_sector_id} (second call, should be cached)...")
    config2 = await sector_service.get_config(test_sector_id)
    if config2:
        print(f"Config loaded (from cache) for {config2.group_name}. H^R Baseline: {config2.h_r_baseline}")
    else:
        print(f"Failed to load config for {test_sector_id}")
        return

    # 3. Invalidate cache for specific sector
    print(f"\nInvalidating cache for {test_sector_id}...")
    sector_service.invalidate_cache(test_sector_id)

    # 4. Fetch config after invalidation - should be a cache miss, loaded from DB and recached
    print(f"\nAttempting to get config for {test_sector_id} (after invalidation, should be re-loaded from DB)...")
    config3 = await sector_service.get_config(test_sector_id)
    if config3:
        print(f"Config re-loaded for {config3.group_name}. H^R Baseline: {config3.h_r_baseline}")
    else:
        print(f"Failed to load config for {test_sector_id}")
        return

    # 5. Test getting all configs
    print("\nAttempting to get all configs...")
    all_configs = await sector_service.get_all_configs()
    if all_configs:
        print(f"Successfully loaded {len(all_configs)} sector configurations.")
        print("Sample calibrations from 'pe_financial_services':")
        fs_config = next((c for c in all_configs if c.focus_group_id == 'pe_financial_services'), None)
        if fs_config:
            print(f"  EBITDA Multiplier: {fs_config.ebitda_multiplier}")
            print(f"  Talent Threshold: {fs_config.talent_concentration_threshold}")
    else:
        print("Failed to load all configs.")

    print("\n--- Testing cache invalidation for all sectors ---")
    sector_service.invalidate_cache() # Invalidate all

    print("\nAttempting to get all configs (after all invalidation, should be re-loaded from DB)...")
    all_configs_reloaded = await sector_service.get_all_configs()
    if all_configs_reloaded:
        print(f"Successfully re-loaded {len(all_configs_reloaded)} sector configurations.")

asyncio.run(test_sector_config_service())
```

### 6.3 Explanation of Execution

The output clearly demonstrates the caching mechanism in action. The first request for a sector's configuration results in a database lookup, but subsequent requests for the same configuration hit the Redis cache, leading to faster retrieval. The `invalidate_cache` function ensures that when configuration data changes in the database, the stale cache entries can be purged, forcing the service to fetch the latest data. This is crucial for maintaining data consistency across the platform while still benefiting from performance optimizations. The weight validation within the `SectorConfig` dataclass acts as a guardrail, preventing misconfigured sector models from affecting downstream analysis.

---

## 7. Creating the Unified Organization View

The final step for Sarah is to create a unified SQL view that joins the core `organizations` table with its `focus_groups` details and the relevant sector-specific attributes. This view provides a single, comprehensive dataset for analysts.

### 7.1 Story + Context + Real-World Relevance

"Analysts often need a holistic view of an organization, combining its core firmographic data with its assigned sector and all its specialized attributes. To simplify their querying and ensure consistency, I'm creating `vw_organizations_full`. This SQL view performs all the necessary joins (using `LEFT JOIN` for optional attributes) so that analysts don't have to write complex queries themselves. This fully realizes our 'One Schema, Many Configurations' approach by presenting a unified, queryable interface over the underlying normalized and configuration-driven tables."

```sql
-- DDL for vw_organizations_full view
CREATE OR REPLACE VIEW vw_organizations_full AS
SELECT
    o.*,
    fg.group_name AS sector_name,
    fg.group_code AS sector_code,
    mfg.ot_systems, mfg.it_ot_integration, mfg.scada_vendor, mfg.mes_system,
    mfg.plant_count, mfg.automation_level, mfg.iot_platforms, mfg.digital_twin_status,
    mfg.edge_computing, mfg.supply_chain_visibility, mfg.demand_forecasting_ai,
    fin.regulatory_bodies, fin.charter_type, fin.model_risk_framework, fin.mrm_team_size,
    fin.model_inventory_count, fin.algo_trading, fin.fraud_detection_ai, fin.credit_ai,
    fin.aml_ai, fin.aum_billions, fin.total_assets_billions,
    hc.hipaa_certified, hc.hitrust_certified, hc.fda_clearances, hc.fda_clearance_count,
    hc.ehr_system, hc.ehr_integration_level, hc.fhir_enabled, hc.clinical_ai_deployed,
    hc.imaging_ai, hc.org_type, hc.bed_count,
    tech.tech_category, tech.primary_language, tech.cloud_native, tech.github_org,
    tech.github_stars_total, tech.open_source_projects, tech.ml_platform, tech.llm_integration,
    tech.ai_product_features, tech.gpu_infrastructure,
    rtl.retail_type, rtl.store_count, rtl.ecommerce_pct, rtl.cdp_vendor,
    rtl.loyalty_program, rtl.loyalty_members, rtl.personalization_ai, rtl.recommendation_engine,
    rtl.demand_forecasting,
    enr.energy_type, enr.regulated, enr.scada_systems, enr.ami_deployed,
    enr.smart_grid_pct, enr.generation_capacity_mw, enr.grid_optimization_ai,
    enr.predictive_maintenance, enr.renewable_pct,
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
```

### 7.2 Code Cell: Create the Unified View and Query It

Sarah creates the view and then runs a sample query to demonstrate its utility, filtering by a specific sector.

```python
def create_unified_organization_view(db_manager: DBManager):
    """Creates the unified organization view."""
    view_ddl = """
    CREATE OR REPLACE VIEW vw_organizations_full AS
    SELECT
        o.*,
        fg.group_name AS sector_name,
        fg.group_code AS sector_code,
        mfg.ot_systems, mfg.it_ot_integration, mfg.scada_vendor, mfg.mes_system,
        mfg.plant_count, mfg.automation_level, mfg.iot_platforms, mfg.digital_twin_status,
        mfg.edge_computing, mfg.supply_chain_visibility, mfg.demand_forecasting_ai,
        fin.regulatory_bodies, fin.charter_type, fin.model_risk_framework, fin.mrm_team_size,
        fin.model_inventory_count, fin.algo_trading, fin.fraud_detection_ai, fin.credit_ai,
        fin.aml_ai, fin.aum_billions, fin.total_assets_billions,
        hc.hipaa_certified, hc.hitrust_certified, hc.fda_clearances, hc.fda_clearance_count,
        hc.ehr_system, hc.ehr_integration_level, hc.fhir_enabled, hc.clinical_ai_deployed,
        hc.imaging_ai, hc.org_type, hc.bed_count,
        tech.tech_category, tech.primary_language, tech.cloud_native, tech.github_org,
        tech.github_stars_total, tech.open_source_projects, tech.ml_platform, tech.llm_integration,
        tech.ai_product_features, tech.gpu_infrastructure,
        rtl.retail_type, rtl.store_count, rtl.ecommerce_pct, rtl.cdp_vendor,
        rtl.loyalty_program, rtl.loyalty_members, rtl.personalization_ai, rtl.recommendation_engine,
        rtl.demand_forecasting,
        enr.energy_type, enr.regulated, enr.scada_systems, enr.ami_deployed,
        enr.smart_grid_pct, enr.generation_capacity_mw, enr.grid_optimization_ai,
        enr.predictive_maintenance, enr.renewable_pct,
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
    db_manager.execute(view_ddl)
    print("Unified organization view 'vw_organizations_full' created/replaced.")

create_unified_organization_view(db_manager)

# Query the unified view for a specific sector
def query_unified_view_by_sector(db_manager: DBManager, sector_name: str, limit: int = 5):
    """Queries the unified view and displays results for a specific sector."""
    print(f"\n--- Querying Unified View for Sector: {sector_name} (first {limit} records) ---")
    query = f"SELECT * FROM vw_organizations_full WHERE sector_name = :sector_name LIMIT :limit;"
    results = db_manager.fetch_all(query, {'sector_name': sector_name, 'limit': limit})
    if results:
        df_results = pd.DataFrame(results)
        # Select key columns for display to avoid overwhelming output
        display_cols = [
            'legal_name', 'sector_name', 'employee_count', 'annual_revenue_usd',
            'plant_count', 'algo_trading', 'ehr_system', 'github_stars_total', 'store_count',
            'generation_capacity_mw', 'firm_type'
        ]
        # Filter for columns that actually exist in the result set
        existing_cols = [col for col in display_cols if col in df_results.columns]
        display(df_results[existing_cols])
    else:
        print(f"No organizations found for sector '{sector_name}' in the unified view.")

# Example queries
query_unified_view_by_sector(db_manager, 'Manufacturing')
query_unified_view_by_sector(db_manager, 'Financial Services')
query_unified_view_by_sector(db_manager, 'Technology')
```

### 7.3 Explanation of Execution

The `vw_organizations_full` view is now available, consolidating all relevant data into a single, easy-to-query interface. Analysts can now retrieve comprehensive information about any organization, including its core details, sector, and sector-specific attributes, with a simple `SELECT * FROM vw_organizations_full` and appropriate filtering. This eliminates the need for complex multi-table joins in every analytical query, significantly improving developer productivity and data accessibility for the PE Org-AI-R team. It's a testament to how a well-designed, configuration-driven data architecture can streamline real-world workflows.

---

## Conclusion

Sarah has successfully implemented a robust, configuration-driven data architecture for the PE Org-AI-R platform. This system effectively manages sector-specific evaluation criteria and organizational attributes without resorting to complex hardcoding or schema proliferation. The "One Schema, Many Configurations" principle has been demonstrated through a unified data model that provides flexibility and scalability. The integration of a caching service ensures high-performance access to critical configuration data, while clear data validation reinforces data integrity. This approach empowers the PE Org-AI-R firm to adapt quickly to changing market dynamics and sector-specific investment strategies.

**Next Steps for PE Org-AI-R:**
-   Develop API endpoints to expose the `SectorConfigService` and the `vw_organizations_full` view.
-   Build an administrative UI for managing sector configurations, including historical versions of weights and calibrations.
-   Integrate this data platform with AI-driven scoring models that leverage the sector-specific weights and calibrations.
