"""
External Memory System for GCRI.

Provides persistent storage for learned rules and knowledge across tasks.
"""
import json
import os
from typing import List, Optional, Dict, Any

from loguru import logger


class ExternalMemory:
    """
    JSON-based persistent memory for cross-task learning.

    Stores:
    - global_rules: Apply to all tasks
    - domain_rules: Apply to specific domains (coding, math, etc.)
    - knowledge: Structured knowledge (patterns, concepts, algorithms)
    """

    def __init__(self, path: str):
        self.path = path
        self._data = {'global_rules': [], 'domain_rules': {}, 'knowledge': {}}
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
                # Ensure knowledge key exists for backward compatibility
                if 'knowledge' not in self._data:
                    self._data['knowledge'] = {}
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

    def save_knowledge(
        self,
        domain: str,
        knowledge_type: str,
        title: str,
        content: str,
        code: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_date: Optional[str] = None
    ):
        """
        Save structured knowledge to external memory.

        Args:
            domain: Domain to categorize (e.g., 'dp_algorithms')
            knowledge_type: Type of knowledge ('pattern', 'concept', 'algorithm')
            title: Title of the knowledge entry
            content: Description/explanation
            code: Optional code example
            tags: Optional tags for search
            source_date: Optional source document date (for web search results)
        """
        from datetime import datetime
        entry = {
            'type': knowledge_type,
            'title': title,
            'content': content,
            'created_at': datetime.now().isoformat(),
        }
        if source_date:
            entry['source_date'] = source_date
        if code:
            entry['code'] = code
        if tags:
            entry['tags'] = tags
        # Avoid duplicates by title
        domain_knowledge = self._data.setdefault('knowledge', {}).setdefault(domain, [])
        existing_titles = {k.get('title') for k in domain_knowledge}
        if title not in existing_titles:
            domain_knowledge.append(entry)
            self._save_to_disk()
            logger.info(f'Knowledge saved: "{title}" in domain "{domain}"')
        else:
            logger.debug(f'Knowledge "{title}" already exists, skipping')

    def load_knowledge(self, domain: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load knowledge from external memory.

        Args:
            domain: Optional domain filter
            tags: Optional tags filter (matches any)

        Returns:
            List of knowledge entries
        """
        knowledge = self._data.get('knowledge', {})
        results = []
        domains_to_search = [domain] if domain else list(knowledge.keys())
        for d in domains_to_search:
            if d not in knowledge:
                continue
            for entry in knowledge[d]:
                if tags:
                    entry_tags = set(entry.get('tags', []))
                    if not entry_tags.intersection(set(tags)):
                        continue
                results.append({**entry, 'domain': d})
        return results

    def clear(self, domain: Optional[str] = None):
        """Clear rules (all or domain-specific)."""
        if domain:
            self._data.get('domain_rules', {}).pop(domain, None)
            self._data.get('knowledge', {}).pop(domain, None)
        else:
            self._data = {'global_rules': [], 'domain_rules': {}, 'knowledge': {}}
        self._save_to_disk()

    @property
    def stats(self) -> dict:
        """Get memory statistics."""
        knowledge_counts = {k: len(v) for k, v in self._data.get('knowledge', {}).items()}
        return {
            'global_count': len(self._data.get('global_rules', [])),
            'domains': list(self._data.get('domain_rules', {}).keys()),
            'domain_counts': {
                k: len(v) for k, v in self._data.get('domain_rules', {}).items()
            },
            'knowledge_domains': list(self._data.get('knowledge', {}).keys()),
            'knowledge_counts': knowledge_counts
        }
