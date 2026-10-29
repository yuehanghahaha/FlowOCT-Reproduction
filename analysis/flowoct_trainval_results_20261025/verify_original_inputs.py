"""CPU-only original-data checks for all20 downloaded projection cases."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=Path(r"D:\data\FSdata\flowoct")
sys.path.insert(0,str(ROOT/"flowoct_reproduction"))
from flowoct.data import PairedVolumeDataset,read_manifest


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    cfg=json.loads((SOURCE/"protocol.json").read_text(encoding="utf-8"))["config"]
    rows=read_manifest(ROOT/"data/OCTA500_common_v1/split_3M_z256.csv","3M")
    with (SOURCE/"case_list.csv").open(encoding="utf-8-sig",newline="") as stream:
        cases=list(csv.DictReader(stream))
    proofs=json.loads((SOURCE/"case_preprocessing_proof.json").read_text(encoding="utf-8"))
    records=[]
    for index,row in enumerate(cases,1):
        cid,split=row["case_id"],row["source_split"]
        original=next(r for r in rows if r["case_id"]==cid)
        assert original["split"]==split
        dataset=PairedVolumeDataset([original],split,cfg)
        dataset.split="validation"
        item=dataset[0]
        gt=item["B"].numpy()[0].transpose(2,0,1).copy()
        proof=proofs[f"{split}_{cid}"]
        assert array_hash(item["A"].numpy())==proof["input_oct_sha256"]
        assert array_hash(gt)==proof["gt_octa_sha256"]
        assert item["geometry"]==proof["geometry"]
        full=gt.mean(axis=0,dtype=np.float64).T.astype(np.float32)
        mip=gt.max(axis=0).T.copy()
        with np.load(SOURCE/"projections"/f"{split}_{cid}.npz",allow_pickle=False) as data:
            assert np.array_equal(data["gt_full"],full) and np.array_equal(data["gt_mip"],mip)
        records.append({"split":split,"case_id":cid,"input_oct_voxel_hash_matches":True,
                        "gt_octa_voxel_hash_matches":True,"geometry_matches":True,
                        "gt_full_exactly_matches":True,"gt_mip_exactly_matches":True})
        print(f"SOURCE {index}/20 {split} {cid}: OCT/GT voxel hashes, geometry and both GT maps match",flush=True)
        del item,gt,full,mip,dataset
    result={"cases":len(records),"records":records,
            "scope":"All20 original common z256 cases reloaded and center-preprocessed on CPU; full input OCT and OCTA voxel hashes, metadata and both GT projections match downloaded records",
            "limits":"Only GT and input can be regenerated without epoch70/140/313 weights; the current package cannot independently re-evaluate 3D predictions",
            "source_modified":False}
    (HERE/"original_input_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"cases":len(records),"all_checks_passed":True},indent=2))


if __name__=="__main__":
    main()
