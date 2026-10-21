"""CPU audit into a new output directory, preserving original candidate evidence."""
import argparse
import json
from pathlib import Path
from rlflow.store import atomic
from tools.r6_candidate_analysis import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('authorization',type=Path)
    parser.add_argument('destination',type=Path);parser.add_argument('--authorization-sha256',required=True)
    args=parser.parse_args();output=args.output.absolute();destination=args.destination.absolute()
    if (output.resolve()!=output or destination.resolve()!=destination or output==destination
            or output in destination.parents or destination in output.parents):parser.error('separate canonical destination required')
    report,summary=analyze(output,dict(path=str(args.authorization.absolute()),sha256=args.authorization_sha256))
    data=json.dumps(report,sort_keys=True,allow_nan=False).encode()+b'\n'
    small=json.dumps(summary,sort_keys=True,allow_nan=False).encode()+b'\n'
    if len(data)>16*1024**2 or len(small)>48*1024:raise ValueError('CANDIDATE_ANALYSIS_REPORT_BOUND')
    destination.mkdir(parents=True,exist_ok=False)
    atomic(destination/'report.json',data);atomic(destination/'summary.json',small)
    print(str(destination/'summary.json'))


if __name__=='__main__':main()
