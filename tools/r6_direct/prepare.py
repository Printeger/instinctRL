"""CPU-only builders for explicit local inputs. No launch, model calls or ledger writes."""
import argparse
import json
from pathlib import Path
from tools.r6_direct.dependencies import install
from tools.r6_direct.evidence import read, validate_binding, require
from tools.r6_evidence import canonical, persist_exact


def final_freeze(recipe_freeze, baseline_bindings, analyses):
    from training_workloads.unified_h1 import robustness_contract as base
    from training_workloads.unified_h1 import robustness_training_plan as training
    from training_workloads.unified_h1.direct_selection import training_model
    freeze = read(recipe_freeze)
    recipe = training.validate_recipe(read(freeze['recipe']))
    baseline = base.baseline(dict(baseline_bindings=baseline_bindings))
    require(len(analyses) == 3, 'THREE_CONFIRMATIONS')
    models = {'baseline': baseline['selected_model']}; streams = []; experiments = set()
    for ordinal, analysis in enumerate(analyses):
        bound, ctx, report = validate_binding(analysis)
        plan = training.validate(ctx['contract']['parameters']['r6_training_plan'])
        require(plan['schema_version'] == 'r6-local-training-plan/1' and plan['phase'] == 'confirmation'
                and plan['recipe'] == freeze['recipe'] and plan['selection_freeze'] == recipe_freeze
                and plan['training_seed'] == recipe['confirmation_seeds'][ordinal]
                and plan['start_update'] == 640 and plan['target_update'] == 640 + recipe['branch_updates']
                and ctx['contract']['parameters']['baseline_bindings'] == baseline_bindings, 'CONFIRMATION_IDENTITY')
        require(plan['experiment_id'] not in experiments, 'DISTINCT_RUNS')
        experiments.add(plan['experiment_id'])
        rows = [r for r in report['summary']['checkpoints'] if r['update'] == plan['target_update']]
        require(len(rows) == 1, 'TERMINAL_CHECKPOINT')
        checkpoint = rows[0]['pin']
        training_model(plan, analysis, checkpoint, read(recipe['method']))
        models['seed' + str(ordinal + 1)] = checkpoint
        streams.append(dict(plan=plan, baseline_bindings=baseline_bindings, analysis=analysis, checkpoint=checkpoint,
                            plan_sha256=base.object_sha(plan), training_seed=plan['training_seed'], target_update=plan['target_update']))
    evidence = read(freeze['selection_evidence'])
    return dict(schema_version='r6-final-freeze/2', branch='THREE_CONTINUATIONS', protocol=evidence['protocol'],
                recipe=freeze['recipe'], selection_rule=evidence['checkpoint_selection'], models=models,
                evidence=dict(recipe_freeze=recipe_freeze, baseline_bindings=baseline_bindings, streams=streams))


def final_batch(freeze, freeze_review, publication, publication_commit, realization):
    from training_workloads.unified_h1.robustness_holdout import reveal
    from training_workloads.unified_h1 import robustness_contract as base
    frozen = read(freeze); public = read(publication); actual = read(realization)
    require(frozen['branch'] == 'THREE_CONTINUATIONS', 'ACTUAL_TRAINING_REQUIRED')
    base.accepted_review(freeze_review, 'PASS_R6_FINAL_FREEZE', freeze_sha256=freeze['sha256'])
    require(public['freeze_sha256'] == freeze['sha256'] and public['protocol_sha256'] == frozen['protocol']['sha256']
            and actual == reveal(public, actual['nonce']), 'COMMITMENT')
    from training_workloads.unified_h1.direct_selection import verify_publication
    verify_publication(publication_commit, freeze, freeze_review, publication)
    assignments = [dict(model_role=role, checkpoint=frozen['models'][role], realization_index=0,
                        evaluation_seed=actual['realization_seeds'][0], split='final')
                   for role in ('baseline', 'seed1', 'seed2', 'seed3')]
    return dict(schema_version='r6-local-final-batch-inputs/1', freeze=freeze, freeze_review=freeze_review,
                publication=publication, publication_commit=publication_commit, realization=realization,
                assignments=assignments, protocol=frozen['protocol'], scientific_acceptance=False, dispatch_authorized=False)


def build(kind, request):
    install()
    if kind == 'contract':
        from training_workloads.unified_h1.direct_contract import (
            baseline_runtime_contract, training_runtime_contract, evaluation_runtime_contract, render_runtime_contract)
        builders = {'baseline': baseline_runtime_contract, 'train': training_runtime_contract,
                    'evaluate': evaluation_runtime_contract, 'final': lambda p: evaluation_runtime_contract(p, final=True),
                    'render': render_runtime_contract}
        return builders[request['operation']](request['parameters'])
    if kind == 'confirmation-plans':
        from training_workloads.unified_h1.robustness_confirmation import build_plans
        require('local' not in request and 'recipe_review' not in request, 'LOCAL_REQUEST')
        return build_plans(**request, recipe_review=None, local=True)
    if kind == 'final-freeze':
        return final_freeze(**request)
    if kind == 'final-batch':
        return final_batch(**request)
    raise ValueError('UNKNOWN_PREPARATION')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('contract', 'confirmation-plans', 'final-freeze', 'final-batch'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    from tools.r6_direct.import_legacy import stop_paths
    if stop_paths():
        raise RuntimeError('STOP')
    result = build(args.kind, json.loads(args.request.read_text()))
    if stop_paths():
        raise RuntimeError('STOP')
    if args.destination.exists():
        raise FileExistsError(args.destination)
    persist_exact(args.destination.absolute(), canonical(result) + b'\n')


if __name__ == '__main__':
    main()
