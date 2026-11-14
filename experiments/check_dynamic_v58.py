"""CPU/GPU equivalence, rejected-suffix exclusion and recency-selection tests."""
import json,hashlib
from pathlib import Path
import torch
from dynamic_vocab_v58 import candidate_ids,update_recency
for device in ['cpu','cuda']:
    logits=torch.zeros((7,32),device=device)
    # Counts include a correction/bonus. Rows2,3 belong to rejected suffix.
    for row,token in enumerate([4,5,30,31,6,7,8]):logits[row,token]=10
    cu=torch.tensor([0,4,7],device=device);counts=torch.tensor([2,3],device=device)
    ids,valid=candidate_ids(logits,cu,counts,k=1)
    assert valid.cpu().tolist()==[True,True,False,False,True,True,True]
    state=torch.zeros(32,dtype=torch.int64,device=device)
    selected=update_recency(state,ids,valid,2,5)
    assert selected.cpu().tolist()==[4,5,6,7,8]
    assert state[30].item()==state[31].item()==0
    selected=update_recency(state,torch.tensor([[9],[9]],device=device),torch.tensor([True,True],device=device),3,3)
    assert selected.cpu().tolist()==[4,5,9]
    # Independent scalar reference for randomized histories and duplicate IDs.
    gen=torch.Generator().manual_seed(18657);ref=[0]*97;state=torch.zeros(97,dtype=torch.int64,device=device)
    for step in range(1,101):
        candidates=torch.randint(0,97,(11,4),generator=gen);mask=torch.rand(11,generator=gen)>.4
        for row,keep in zip(candidates.tolist(),mask.tolist()):
            if keep:
                for token in row:ref[token]=step
        wanted=sorted(sorted(range(97),key=lambda v:(-ref[v],v))[:13])
        got=update_recency(state,candidates.to(device),mask.to(device),step,13).cpu().tolist()
        assert got==wanted
out=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),module_sha256=hashlib.sha256(Path(__file__).with_name('dynamic_vocab_v58.py').read_bytes()).hexdigest(),passed=True,devices=['cpu','cuda'],random_steps_per_device=100,scope='Finite algorithm tests only; full-model output gates remain required.')
Path(__file__).with_name('dynamic_v58_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
