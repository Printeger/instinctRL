"""Explicit new-input resource authorization; historical accounting stays intact."""
import json
from tools.r6_evidence import checked


def unlimited(policy):
    if policy is None:
        return False
    authority = json.loads(checked(policy).read_text())
    if (authority.get('schema_version') != 'r6-resource-user-amendment/1'
            or authority.get('user_instruction') != '同时取消所有保护额度和预算限制。'
            or authority.get('policy') != 'UNLIMITED_PROJECT_BUDGET_NO_PROTECTED_ALLOCATION'
            or authority.get('historical_accounting') != 'PRESERVE_ALL_CHARGES_UNKNOWNS_AND_UNRELEASED_RESERVATIONS'):
        raise ValueError('DIRECT_RESOURCE_POLICY')
    return True
