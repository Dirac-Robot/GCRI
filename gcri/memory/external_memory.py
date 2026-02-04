"""
External Memory System for GCRI.

Provides persistent storage for learned rules across tasks.
"""
import json
import os
from typing import List, Optional

from loguru import logger


class ExternalMemory:
    """
    JSON-based persistent memory for cross-task learning.

    Stores:
    - global_rules: Apply to all tasks
    - domain_rules: Apply to specific domains (coding, math, etc.)
    """

    def __init__(self, path: str):
        self.path = path
        self._data = {'global_rules': [], 'domain_rules': {}}
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
                logger.debug(f'External memory loaded from {self.path}')
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f'Failed to load external memory: {e}')

    def _save_to_disk(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def load(self, domain: Optional[str] = None) -> List[str]:
        """
        Load rules from external memory.

        Args:
            domain: Optional domain hint (e.g., 'coding', 'math')

        Returns:
            List of rules (global + domain-specific if matched)
        """
        rules = list(self._data.get('global_rules', []))
        if domain and domain in self._data.get('domain_rules', {}):
            rules.extend(self._data['domain_rules'][domain])
        return rules

    def save(self, rules: List[str], domain: Optional[str] = None, as_global: bool = False):
        """
        Save rules to external memory.

        Args:
            rules: List of rules to save
            domain: Domain to categorize rules (None = global)
            as_global: Force save as global rules
        """
        if not rules:
            return
        # Deduplicate
        existing_global = set(self._data.get('global_rules', []))
        if as_global or domain is None:
            for rule in rules:
                if rule not in existing_global:
                    self._data.setdefault('global_rules', []).append(rule)
        else:
            existing_domain = set(self._data.get('domain_rules', {}).get(domain, []))
            for rule in rules:
                if rule not in existing_domain and rule not in existing_global:
                    self._data.setdefault('domain_rules', {}).setdefault(domain, []).append(rule)
        self._save_to_disk()
        logger.info(f'External memory updated: {len(rules)} rules saved')

    def clear(self, domain: Optional[str] = None):
        """Clear rules (all or domain-specific)."""
        if domain:
            self._data.get('domain_rules', {}).pop(domain, None)
        else:
            self._data = {'global_rules': [], 'domain_rules': {}}
        self._save_to_disk()

    @property
    def stats(self) -> dict:
        """Get memory statistics."""
        return {
            'global_count': len(self._data.get('global_rules', [])),
            'domains': list(self._data.get('domain_rules', {}).keys()),
            'domain_counts': {
                k: len(v) for k, v in self._data.get('domain_rules', {}).items()
            }
        }
