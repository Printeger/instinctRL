"""CPU-only R6 budget checks. Caller holds the single controller writer lock.

No Store permission, submission, settlement or budget refund occurs here.
Reservations remain charged even when an attempt or worker fails.
"""
import copy
import math


class BudgetError(ValueError):
    pass


def amount(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError('Resource amount must be numeric')
    if not math.isfinite(value) or value < 0:
        raise BudgetError('Resource amount must be finite and nonnegative')
    return value


def totals(ledger):
    caps = ledger['caps']
    result = {k: amount(ledger['used'][k]) for k in caps}
    seen = set()
    for reservation in ledger['reservations']:
        key = reservation['event_key']
        if key in seen:
            raise BudgetError('Duplicate reservation event')
        seen.add(key)
        for resource, value in reservation['resources'].items():
            if resource not in caps:
                raise BudgetError('Unknown resource: ' + resource)
            result[resource] += amount(value)
    for resource, value in result.items():
        if value > amount(caps[resource]):
            raise BudgetError('Cap exceeded: ' + resource)
    return result


def reserve(ledger, *, event_key, resources, purpose, free_disk_bytes):
    """Return a new ledger; identical replays don't spend again, drift fails.

    event_key binds goal, event, job, input hash and role. Final/recovery use
    of protected capacity is explicit. Ordinary development cannot borrow it.
    Disk check subtracts all outstanding output reservations conservatively.
    """
    required = {'goal_id', 'event_id', 'job_id', 'input_sha256', 'role'}
    if set(event_key) != required or event_key['goal_id'] != ledger['goal_id']:
        raise BudgetError('Incomplete event identity')
    if not all(isinstance(v, str) and v for v in event_key.values()):
        raise BudgetError('Empty event identity')
    digest = event_key['input_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise BudgetError('Invalid input hash')
    if purpose not in {'development', 'confirmation_delivery', 'repair', 'final_review'}:
        raise BudgetError('Unknown reservation purpose')
    key = tuple(event_key[k] for k in sorted(required))
    # JSON round trips preserve this stable string; tuples are not persisted.
    import json
    key = json.dumps(key, separators=(',', ':'))
    entry = {'event_key': key, 'identity': event_key, 'resources': resources, 'purpose': purpose}
    for prior in ledger['reservations']:
        if prior['event_key'] == key:
            if prior != entry:
                raise BudgetError('Reservation replay drift')
            totals(ledger)
            return copy.deepcopy(ledger)
    candidate = copy.deepcopy(ledger)
    candidate['reservations'].append(copy.deepcopy(entry))
    usage = totals(candidate)
    if purpose == 'development':
        for resource, protected in ledger['protected_confirmation_delivery_repair'].items():
            # Protected use cannot accidentally become available to exploration.
            dev_used = amount(ledger['used'][resource])
            dev_used += sum(r['resources'].get(resource, 0) for r in candidate['reservations']
                            if r['purpose'] == 'development')
            if dev_used > ledger['caps'][resource] - amount(protected):
                raise BudgetError('Protected capacity: ' + resource)
    model_cap = ledger['caps']['automatic_model_sessions']
    other_models = amount(ledger['used']['automatic_model_sessions']) + sum(r['resources'].get('automatic_model_sessions', 0)
                       for r in candidate['reservations']
                       if r['purpose'] not in {'repair', 'final_review'})
    if other_models > model_cap - sum(ledger['protected_model_sessions'].values()):
        raise BudgetError('Protected review/repair sessions')
    for protected_role, count in ledger['protected_model_sessions'].items():
        outside_role = amount(ledger['used']['automatic_model_sessions'])
        outside_role += sum(r['resources'].get('automatic_model_sessions', 0)
                            for r in candidate['reservations']
                            if r['purpose'] != protected_role)
        if outside_role > model_cap - count:
            raise BudgetError('Protected model role: ' + protected_role)
    disk_key = 'output_actual_plus_unreleased_reservations_bytes'
    if amount(free_disk_bytes) - usage[disk_key] < ledger['disk_free_floor_bytes']:
        raise BudgetError('Disk floor after reservations')
    return candidate
