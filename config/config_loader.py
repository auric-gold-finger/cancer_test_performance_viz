"""
Configuration Loader for Cancer Risk Factors

This module provides a RiskConfig class that loads and manages risk factor
configuration from YAML files, making medical data easily updateable without
code changes.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class RiskConfig:
    """Load and manage risk factor configuration from YAML files"""

    def __init__(self, config_path: str = None):
        """
        Initialize RiskConfig with path to configuration file

        Args:
            config_path: Path to risk_factors.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/risk_factors.yaml relative to this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "risk_factors.yaml"

        self.config_path = Path(config_path)
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        """
        Lazy load configuration file

        Returns:
            Dictionary containing all configuration data
        """
        if self._config is None:
            self._load_config()
        return self._config

    def _load_config(self):
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please ensure config/risk_factors.yaml exists."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get_smoking_risk(
        self,
        smoking_status: str,
        cancer_type: str,
        pack_years: int = 0
    ) -> float:
        """
        Get smoking risk multiplier for a specific cancer type

        Args:
            smoking_status: One of 'Never smoked', 'Former smoker', 'Current smoker'
            cancer_type: Cancer type key (e.g., 'lung', 'bladder')
            pack_years: Pack-years of smoking exposure (packs/day × years)

        Returns:
            Risk multiplier (1.0 = no increased risk)

        Examples:
            >>> config.get_smoking_risk('Current smoker', 'lung', 30)
            15.0  # 30 pack-years = 15x risk
        """
        # Normalize smoking status to config key format
        status_map = {
            'Never smoked': 'never_smoked',
            'Former smoker': 'former_smoker',
            'Current smoker': 'current_smoker'
        }
        status_key = status_map.get(smoking_status, smoking_status.lower().replace(' ', '_'))

        # Never smokers have no increased risk
        if status_key == 'never_smoked':
            return 1.0

        # Get smoking risk data for this status
        smoking_data = self.config.get('smoking_risk', {})
        if status_key not in smoking_data:
            return 1.0

        # Get cancer-specific data
        cancer_data = smoking_data[status_key].get(cancer_type)
        if not cancer_data:
            return 1.0

        # Handle pack-year ranges (for lung cancer)
        if isinstance(cancer_data, list):
            for range_data in cancer_data:
                min_py, max_py = range_data['pack_years']
                if min_py <= pack_years < max_py:
                    return range_data['multiplier']
            # If pack_years exceeds all ranges, return last range multiplier
            return cancer_data[-1]['multiplier']

        # Handle simple multiplier (for other cancers)
        return cancer_data.get('multiplier', 1.0)

    def get_family_history_risk(
        self,
        cancer_type: str,
        min_diagnosis_age: int,
        relative_count: int
    ) -> float:
        """
        Get family history risk multiplier

        Args:
            cancer_type: Cancer type key (e.g., 'breast', 'colorectal')
            min_diagnosis_age: Age of youngest affected relative
            relative_count: Number of affected first/second degree relatives

        Returns:
            Risk multiplier accounting for family history and age

        Examples:
            >>> config.get_family_history_risk('breast', 45, 1)
            2.2  # One relative diagnosed < 50

            >>> config.get_family_history_risk('breast', 45, 3)
            3.08  # Three relatives: 2.2 * (1 + 0.2*2) = 2.2 * 1.4
        """
        fh_data = self.config.get('family_history_risk', {}).get(cancer_type)

        # Use default if cancer type not specified
        if not fh_data:
            fh_data = self.config.get('family_history_risk', {}).get('default', {})
            if not fh_data:
                return 1.3  # Fallback default

        # Check for age-based multipliers
        if 'age_threshold' in fh_data:
            threshold = fh_data['age_threshold']
            if min_diagnosis_age < threshold:
                base_multiplier = fh_data['base_multiplier_early']
            else:
                base_multiplier = fh_data['base_multiplier_late']
        else:
            base_multiplier = fh_data.get('base_multiplier', 1.3)

        # Apply diminishing returns for multiple relatives
        if relative_count > 1:
            bonus_per_relative = fh_data.get('multiple_relatives_bonus', 0.2)
            max_bonus_factor = fh_data.get('max_multiple_factor', 1.4)

            # Cap the number of relatives that contribute to bonus
            effective_extra_relatives = min(relative_count - 1, 2)
            family_factor = 1.0 + (bonus_per_relative * effective_extra_relatives)
            family_factor = min(family_factor, max_bonus_factor)

            return base_multiplier * family_factor

        return base_multiplier

    def get_genetic_mutation_risk(
        self,
        mutation: str,
        cancer_type: str
    ) -> float:
        """
        Get genetic mutation risk multiplier

        Args:
            mutation: Mutation name (e.g., 'BRCA1', 'Lynch syndrome')
            cancer_type: Cancer type key

        Returns:
            Risk multiplier for this mutation/cancer combination

        Examples:
            >>> config.get_genetic_mutation_risk('BRCA1', 'breast')
            8.0  # ~70% lifetime risk
        """
        mutations = self.config.get('genetic_mutations', {})
        mutation_data = mutations.get(mutation, {})
        return mutation_data.get(cancer_type, 1.0)

    def get_personal_history_risk(self) -> float:
        """
        Get risk multiplier for personal cancer history

        Returns:
            Risk multiplier for those with prior cancer diagnosis
        """
        return self.config.get('personal_history', {}).get('multiplier', 2.0)

    def get_cancer_family_mapping(self) -> Dict[str, str]:
        """
        Get mapping from user-friendly family history names to cancer type keys

        Returns:
            Dictionary mapping display names to cancer type keys

        Examples:
            >>> config.get_cancer_family_mapping()
            {'Breast cancer': 'breast', 'Colorectal cancer': 'colorectal', ...}
        """
        return self.config.get('cancer_family_mapping', {})

    def get_safety_limits(self) -> Dict[str, float]:
        """
        Get safety limit values for risk calculations

        Returns:
            Dictionary with safety limit parameters

        Examples:
            >>> config.get_safety_limits()
            {'max_individual_risk': 0.60, 'max_overall_risk': 0.85, 'max_risk_multiplier': 25.0}
        """
        return self.config.get('safety_limits', {
            'max_individual_cancer_risk': 0.60,
            'max_overall_cancer_risk': 0.85,
            'max_risk_multiplier': 25.0
        })

    def get_risk_thresholds(self) -> Dict[str, float]:
        """
        Get risk classification thresholds

        Returns:
            Dictionary with low, moderate, high risk thresholds
        """
        return self.config.get('risk_thresholds', {
            'low': 0.05,
            'moderate': 0.15,
            'high': 0.15
        })

    def get_display_thresholds(self) -> Dict[str, float]:
        """
        Get display threshold values

        Returns:
            Dictionary with display thresholds
        """
        return self.config.get('display_thresholds', {
            'minimum_risk_to_show': 0.001,
            'high_risk_recommendation': 0.10,
            'moderate_risk_recommendation': 0.05,
            'low_risk_recommendation': 0.02
        })

    def get_risk_colors(self) -> Dict[str, str]:
        """
        Get color codes for risk level visualization

        Returns:
            Dictionary mapping risk levels to hex color codes
        """
        return self.config.get('risk_colors', {
            'low': '#22c55e',
            'moderate': '#f59e0b',
            'high': '#ef4444'
        })

    def reload(self):
        """Force reload of configuration from file"""
        self._config = None
        # Accessing .config will trigger reload
        _ = self.config


# Singleton instance for application-wide use
_risk_config_instance = None


def get_risk_config() -> RiskConfig:
    """
    Get singleton instance of RiskConfig

    Returns:
        Shared RiskConfig instance

    Examples:
        >>> config = get_risk_config()
        >>> multiplier = config.get_smoking_risk('Current smoker', 'lung', 30)
    """
    global _risk_config_instance
    if _risk_config_instance is None:
        _risk_config_instance = RiskConfig()
    return _risk_config_instance
