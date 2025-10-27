"""
Configuration module for Cancer Test Performance Visualizer

Provides centralized configuration management for risk factors,
thresholds, and other medical parameters.
"""

from .config_loader import RiskConfig, get_risk_config

__all__ = ['RiskConfig', 'get_risk_config']
