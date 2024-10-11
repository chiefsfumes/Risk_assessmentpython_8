import os
import logging
from typing import Dict, NamedTuple

class Scenario(NamedTuple):
    name: str
    temp_increase: float
    carbon_price: float
    renewable_energy: float
    policy_stringency: float
    biodiversity_loss: float
    ecosystem_degradation: float
    financial_stability: float
    supply_chain_disruption: float

# Scenario definitions
SCENARIOS: Dict[str, Scenario] = {
    "Net Zero 2050": Scenario(
        name="Net Zero 2050",
        temp_increase=1.5,
        carbon_price=250,
        renewable_energy=0.75,
        policy_stringency=0.9,
        biodiversity_loss=0.1,
        ecosystem_degradation=0.2,
        financial_stability=0.8,
        supply_chain_disruption=0.3
    ),
    "Delayed Transition": Scenario(
        name="Delayed Transition",
        temp_increase=2.5,
        carbon_price=125,
        renewable_energy=0.55,
        policy_stringency=0.6,
        biodiversity_loss=0.3,
        ecosystem_degradation=0.4,
        financial_stability=0.6,
        supply_chain_disruption=0.5
    ),
    "Current Policies": Scenario(
        name="Current Policies",
        temp_increase=3.5,
        carbon_price=35,
        renewable_energy=0.35,
        policy_stringency=0.2,
        biodiversity_loss=0.5,
        ecosystem_degradation=0.6,
        financial_stability=0.4,
        supply_chain_disruption=0.7
    ),
    "Nature Positive": Scenario(
        name="Nature Positive",
        temp_increase=1.8,
        carbon_price=200,
        renewable_energy=0.7,
        policy_stringency=0.8,
        biodiversity_loss=-0.1,  # Net gain
        ecosystem_degradation=-0.2,  # Net restoration
        financial_stability=0.75,
        supply_chain_disruption=0.4
    ),
    "Nature Degradation": Scenario(
        name="Nature Degradation",
        temp_increase=3.0,
        carbon_price=50,
        renewable_energy=0.4,
        policy_stringency=0.3,
        biodiversity_loss=0.6,
        ecosystem_degradation=0.7,
        financial_stability=0.5,
        supply_chain_disruption=0.6
    ),
    "Resilient Systems": Scenario(
        name="Resilient Systems",
        temp_increase=2.0,
        carbon_price=150,
        renewable_energy=0.6,
        policy_stringency=0.7,
        biodiversity_loss=0.2,
        ecosystem_degradation=0.3,
        financial_stability=0.9,
        supply_chain_disruption=0.2
    ),
    "Cascading Failures": Scenario(
        name="Cascading Failures",
        temp_increase=3.5,
        carbon_price=75,
        renewable_energy=0.45,
        policy_stringency=0.4,
        biodiversity_loss=0.5,
        ecosystem_degradation=0.6,
        financial_stability=0.3,
        supply_chain_disruption=0.8
    ),
    "Global Instability": Scenario(
        name="Global Instability",
        temp_increase=4.0,
        carbon_price=50,
        renewable_energy=0.4,
        policy_stringency=0.3,
        biodiversity_loss=0.6,
        ecosystem_degradation=0.7,
        financial_stability=0.2,
        supply_chain_disruption=0.8
    )
}

# Keep existing content below this line