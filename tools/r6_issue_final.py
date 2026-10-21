"""Issue one exact assignment covered by the original complete-batch review."""
from tools import r6_final_plans as plans
from tools import r6_final_plans_review as review
from tools import r6_issue_next as issuer
from tools import r6_profile_preparation as profile


def issue(journal,materializations_pin,index,review_key,review_pin,master_pin,store,*,free_disk_bytes):
    profile.guard(journal)
    review.verify(journal,review_key,review_pin,materializations_pin,store)
    aggregate=plans.read(materializations_pin)
    row=plans.assignment(journal,aggregate,index,store)
    plan=plans.read(row['plan']);spec=plans.read(row['spec'])
    return issuer._issue_reviewed(journal,'FINAL:'+review_key+':'+str(index),row['plan'],
        review_key,review_pin,master_pin,store,plan,spec,row['job_id'],
        free_disk_bytes=free_disk_bytes,role='final-batch-authorization')
