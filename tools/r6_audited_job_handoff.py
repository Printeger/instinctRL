"""Add local wait records around the immutable, independently reviewed handoff.

The original waiter retains all process/STOP/closure checks. This adapter records
its durable local observations without changing the historical review's sources.
"""
from tools import r6_job_handoff as original
from tools import r6_wait_audit as audit


class ObservedJournal:
    def __init__(self,journal,key):
        self.journal=journal;self.key=key
        self.observations=journal.read()['events'][key].get('local_wait_observations',0)

    def __getattr__(self,name):return getattr(self.journal,name)

    def save(self,state):
        # The old waiter may hold state from before an added audit write. Preserve
        # that audit rather than letting a subsequent waiter save erase it.
        current=self.journal.read()['events'][self.key]
        entry=state['events'][self.key]
        if 'local_wait_audit' in current:entry['local_wait_audit']=current['local_wait_audit']
        count=entry.get('local_wait_observations',0)
        self.journal.save(state)
        if count>self.observations:
            original.no_model_workers(self.journal)
            audit.record(self.journal,self.key)
            self.observations=count


def job_handoff(journal,key,store,*,poll_seconds=15):
    journal.require_writer();original.no_model_workers(journal)
    audit.record(journal,key)
    result=original.job_handoff(ObservedJournal(journal,key),key,store,poll_seconds=poll_seconds)
    original.no_model_workers(journal);audit.record(journal,key,finish=True)
    from tools.r6_platform_wait import capture
    capture(journal,key)
    return result
