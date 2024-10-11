from typing import List, Dict, Any
from src.models import Risk, Scenario, PESTELAnalysis, SystemicRisk
from src.config import LLM_MODEL, LLM_API_KEY, SCENARIOS
from src.prompts import (RISK_NARRATIVE_PROMPT, EXECUTIVE_INSIGHTS_PROMPT, 
                         SYSTEMIC_RISK_PROMPT, MITIGATION_STRATEGY_PROMPT, 
                         PESTEL_ANALYSIS_PROMPT)
import openai
import numpy as np
import re
from src.risk_analysis.pestel_analysis import perform_pestel_analysis
from src.risk_analysis.sasb_integration import integrate_sasb_materiality
from src.risk_analysis.systemic_risk_analysis import analyze_systemic_risks, identify_trigger_points, assess_resilience
from src.risk_analysis.interaction_analysis import analyze_risk_interactions, build_risk_network

openai.api_key = LLM_API_KEY

def conduct_advanced_risk_analysis(risks: List[Risk], scenarios: Dict[str, Scenario], company_industry: str, key_dependencies: List[str], external_data: Dict) -> Dict:
    comprehensive_analysis = {}
    for scenario_name, scenario in scenarios.items():
        scenario_analysis = {}
        for risk in risks:
            scenario_analysis[risk.id] = llm_risk_assessment(risk, scenario, company_industry)
        comprehensive_analysis[scenario_name] = scenario_analysis
    
    risk_narratives = generate_risk_narratives(risks, comprehensive_analysis)
    executive_insights = generate_executive_insights(comprehensive_analysis, risks)
    systemic_risks = analyze_systemic_risks(risks, company_industry, key_dependencies)
    risk_interactions = analyze_risk_interactions(risks)
    risk_network = build_risk_network(risks, risk_interactions)
    cross_scenario_results = perform_cross_scenario_analysis(comprehensive_analysis)
    key_uncertainties = identify_key_uncertainties(cross_scenario_results)
    mitigation_strategies = generate_mitigation_strategies(risks, comprehensive_analysis)
    
    pestel_analysis = perform_pestel_analysis(risks, external_data)
    sasb_material_risks = integrate_sasb_materiality(risks, company_industry)
    
    trigger_points = identify_trigger_points(risks, risk_network, external_data)
    resilience_assessment = assess_resilience(risks, comprehensive_analysis, cross_scenario_results)
    
    return {
        "comprehensive_analysis": comprehensive_analysis,
        "risk_narratives": risk_narratives,
        "executive_insights": executive_insights,
        "systemic_risks": systemic_risks,
        "cross_scenario_results": cross_scenario_results,
        "key_uncertainties": key_uncertainties,
        "mitigation_strategies": mitigation_strategies,
        "pestel_analysis": pestel_analysis,
        "sasb_material_risks": sasb_material_risks,
        "trigger_points": trigger_points,
        "resilience_assessment": resilience_assessment
    }

def llm_risk_assessment(risk: Risk, scenario: Scenario, company_industry: str) -> Dict[str, Any]:
    prompt = f"""
    As an expert in climate risk assessment for the {company_industry} industry, analyze the following risk under the given scenario:

    Risk: {risk.description}
    Category: {risk.category}
    Subcategory: {risk.subcategory}
    Current likelihood: {risk.likelihood}
    Current impact: {risk.impact}

    Scenario: {scenario.name}
    - Temperature increase: {scenario.temp_increase}°C
    - Carbon price: ${scenario.carbon_price}/ton
    - Renewable energy share: {scenario.renewable_energy * 100}%
    - Policy stringency: {scenario.policy_stringency * 100}%
    - Biodiversity loss: {scenario.biodiversity_loss * 100}%
    - Ecosystem degradation: {scenario.ecosystem_degradation * 100}%
    - Financial stability: {scenario.financial_stability * 100}%
    - Supply chain disruption: {scenario.supply_chain_disruption * 100}%

    Provide a detailed analysis addressing:
    1. How does this risk's likelihood and impact change under the given scenario?
    2. What are the potential financial implications for the company over the next 5 years?
    3. Are there any emerging opportunities related to this risk in this scenario?
    4. What additional challenges might arise from this risk in this specific context?
    5. Suggest 2-3 possible mitigation strategies tailored to this scenario.

    Structure your response with clear headings for each point.
    """

    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in climate risk assessment."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return parse_llm_response(response.choices[0].message['content'])

def parse_llm_response(content: str) -> Dict[str, Any]:
    sections = content.split('\n\n')
    parsed_response = {}
    current_section = ""
    for section in sections:
        if ':' in section:
            title, text = section.split(':', 1)
            current_section = title.strip().lower().replace(' ', '_')
            parsed_response[current_section] = text.strip()
        else:
            parsed_response[current_section] += '\n' + section.strip()
    return parsed_response

# Keep existing functions below this line