"""Rerun frozen comparison locally; preserve Windows extended-length I/O paths.

Python3.9 Path.resolve removes the extended-length prefix, which otherwise
mislabels deep existing legacy transfer manifests as missing. This adapter
changes only process-local path resolution. Frozen analysis source and evidence
are read unchanged; no training or checkpoint pickle load is performed.
"""

import argparse
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    original_resolve=Path.resolve
    def resolved(self,*values,**keywords):
        value=original_resolve(self,*values,**keywords)
        if os.name=="nt" and not str(value).startswith("\\\\?\\"):
            value=Path("\\\\?\\"+str(value))
        return value
    Path.resolve=resolved
    import compare_legacy_control as frozen
    root=args.extracted.resolve()
    result=frozen.compare(root/"legacy_single_gpu_v3/control/legacy_control_plan.json",
        root/"pinn4soh_legacy_formal_control_20261110",
        [root/"pinn4soh_full_paper_stage1",root/"pinn4soh_full_paper_stage2"],args.output)
    import json
    print(json.dumps({key:result[key] for key in ["task_counts","assessment","method_direction_changes","transfer_direction_changes"]}))


if __name__=="__main__":
    main()
