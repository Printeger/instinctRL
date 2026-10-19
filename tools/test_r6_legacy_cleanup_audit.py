"""Reject incomplete or altered evidence; audits never waive the live guard."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.audit_r6_legacy_model_cleanup import audit_event,LEGACY_EVENTS
from tools.r6_recovery import file_pin,RecoveryError


class LegacyAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.key=sorted(LEGACY_EVENTS)[0]

    def event(self,rows):
        f=self.root/'events';f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        other=self.root/'other';other.write_text('{}')
        receipt=self.root/'receipt';receipt.write_text(json.dumps(dict(event_key=self.key,events=file_pin(f),stderr=file_pin(other),last_message=file_pin(other),exit_code=0,error=None,process=None,child_identity=None)))
        return dict(kind='model',receipt=file_pin(receipt))

    def command(self,kind='item.started',command='cat evidence'):
        return dict(type=kind,item=dict(type='command_execution',id='c',command=command,exit_code=0 if kind=='item.completed' else None))

    def test_terminal_commands_are_not_a_cleanup_attestation(self):
        e=self.event([self.command(),self.command('item.completed'),dict(type='turn.completed')])
        before=(self.root/'receipt').read_bytes();r=audit_event(self.key,e)
        self.assertTrue(r['all_reported_tool_items_terminal']);self.assertFalse(r['historical_descendant_cleanup_proven'])
        self.assertTrue(r['independent_reconciliation_required']);self.assertEqual(before,(self.root/'receipt').read_bytes())

    def test_partial_or_substituted_command_not_terminal(self):
        for rows in [[self.command(),dict(type='turn.completed')],[self.command(),self.command('item.completed','different'),dict(type='turn.completed')]]:
            self.assertFalse(audit_event(self.key,self.event(rows))['all_reported_tool_items_terminal'])

    def test_mcp_calls_require_matching_terminal_items(self):
        start=dict(type='item.started',item=dict(type='mcp_tool_call',id='m',server='codex',tool='list_mcp_resources',arguments={},status='in_progress'))
        end=json.loads(json.dumps(start));end['type']='item.completed';end['item']['status']='completed'
        self.assertFalse(audit_event(self.key,self.event([start,dict(type='turn.completed')]))['all_reported_tool_items_terminal'])
        self.assertTrue(audit_event(self.key,self.event([start,end,dict(type='turn.completed')]))['all_reported_tool_items_terminal'])

    def test_pin_drift_and_unknown_event_rejected(self):
        e=self.event([dict(type='turn.completed')]);(self.root/'events').write_text('changed')
        with self.assertRaises(RecoveryError):audit_event(self.key,e)
        with self.assertRaises(ValueError):audit_event('unknown',e)

    def test_duplicate_completion_rejected(self):
        e=self.event([self.command(),self.command('item.completed'),self.command('item.completed')])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):audit_event(self.key,e)


if __name__=='__main__':unittest.main()
