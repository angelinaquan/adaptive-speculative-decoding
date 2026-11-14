"""Expanded real-width acceptance labels for budgeted allocation headroom. No training or speed claim."""
import json,hashlib,traceback,subprocess,os
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'labels_v73_checks.json';assert not out.exists()
result={'status':'initializing','scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model':'RedHatAI/Qwen3-4B-speculator.dflash2','revision':'27aa911ddb040c61adeeade08e92949e24c9e17b','checks':[]}
try:
 gate=json.loads((r/'backbone_v65_checks.json').read_text()); assert gate['passed']
 result['plan_sha256']=hashlib.sha256((r/'labels_v73_plan.json').read_bytes()).hexdigest()
 result['manifest_sha256']=hashlib.sha256((r/'oracle_v57_manifest.json').read_bytes()).hexdigest()
 import torch
 from huggingface_hub import snapshot_download
 from safetensors import safe_open
 from speculators.models.dflash2 import DFlash2DraftModel,DFlash2SpeculatorConfig
 from forespark.models.draft_model import DFlash2ConfDraftModel,DFlash2ConfSpeculatorConfig
 torch.set_num_threads(4);torch.manual_seed(18657)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 root=Path('/tmp/aquan-forespark-v63')
 result['forespark_commit']=subprocess.check_output(['git','-C',str(root/'ForeSpark'),'rev-parse','HEAD'],text=True).strip()
 assert result['forespark_commit']=='5d457ddab73a76f28db34acc577abfbd9e7c6eee'
 paths=json.loads((r/'model_paths.json').read_text());target=paths['Qwen/Qwen3-4B']
 result['target_path']=target
 checkpoint=snapshot_download(result['model'],revision=result['revision'],cache_dir=root/'models',allow_patterns=['*.json','*.safetensors'])
 config=DFlash2SpeculatorConfig.from_pretrained(checkpoint)
 assert config.block_size==9 and config.draft_vocab_size==151936 and config.sliding_window_non_causal
 config.speculators_config.verifier.name_or_path=target
 config.transformer_layer_config._attn_implementation='eager'
 base=DFlash2DraftModel.from_pretrained(checkpoint,config=config,dtype=torch.float32).eval().requires_grad_(False)
 state=base.state_dict();checkpoint_keys=set()
 for path in sorted(Path(checkpoint).glob('*.safetensors')):
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():
    assert key in state, ('Unexpected checkpoint key',key)
    assert torch.equal(state[key].cpu(),f.get_tensor(key).to(state[key].dtype)),('Weight mismatch',key)
    checkpoint_keys.add(key)
 assert checkpoint_keys
 allowed_shared={'embed_tokens.weight','lm_head.weight','verifier_lm_head.weight','verifier_norm.weight'}
 missing=set(dict(base.named_parameters()))-checkpoint_keys
 assert missing<=allowed_shared,('Unloaded draft parameters',missing-allowed_shared)
 result['checkpoint_tensor_count']=len(checkpoint_keys);result['target_owned_parameters']=sorted(missing)
 values=base.config.to_dict()
 values.update(speculators_model_type='dflash2_conf',architectures=['DFlash2ConfDraftModel'],enable_confidence_head=True,enable_fore_d_predictor=True,fore_d_rank=256)
 values['speculators_config']['algorithm']='dflash2_conf'
 extended=DFlash2ConfDraftModel(DFlash2ConfSpeculatorConfig(**values)).eval().requires_grad_(False)
 loading=extended.load_state_dict(state,strict=False)
 expected_new={n for n in extended.state_dict() if n.startswith(('confidence_head.','fore_d_predictor.'))}
 assert set(loading.missing_keys)==expected_new and not loading.unexpected_keys
 assert expected_new
 for key,value in state.items():assert torch.equal(value,extended.state_dict()[key]),('Shared state changed',key)
 result['new_head_parameters']=sorted(expected_new)

 from transformers import AutoModelForCausalLM,AutoTokenizer
 del base,state
 extended=extended.cuda()
 target_model=AutoModelForCausalLM.from_pretrained(target,dtype=torch.float32,attn_implementation='eager',local_files_only=True).cuda().eval().requires_grad_(False)
 tokenizer=AutoTokenizer.from_pretrained(target,local_files_only=True)
 raw={}
 def capture_norm(module,args):raw['h']=args[0].detach()
 handle=target_model.model.norm.register_forward_pre_hook(capture_norm)
 tasks=json.loads((r/'oracle_v57_manifest.json').read_text())
 result['records']=[];features=[]
 with torch.inference_mode():
  for task_index in range(48):
   task=tasks[task_index]
   prompt_text=tokenizer.apply_chat_template([{'role':'user','content':task['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert isinstance(prompt_text,str)
   input_ids=tokenizer(prompt_text,return_tensors='pt',add_special_tokens=False)['input_ids'].to('cuda')
   prompt=input_ids[0].tolist();p=input_ids.shape[1]
   assert p+96<=2048,('Context limit',task_index,p)
   sequence=target_model.generate(input_ids,do_sample=False,max_new_tokens=96)
   offsets=[k for k in [0,8,16,24,32,40] if p+k+8<sequence.shape[1]]
   target_out=target_model(sequence,output_hidden_states=True,use_cache=False)
   expected=sequence[0,p:]
   teacher=target_out.logits[0,p-1:-1].argmax(-1)
   if not torch.equal(teacher,expected):
    result['target_replay_failure']={'task_index':task_index,'expected':expected.tolist(),'actual':teacher.tolist()}
    raise AssertionError(('Target cached-vs-full greedy mismatch',task_index))
   if not offsets:
    result['records'].append({'task_index':task_index,'domain':task['domain'],'input_ids':prompt,'output_ids':sequence[0,p:].tolist(),'anchors':[],'widths':{},'skipped':'EOS before a complete width8 comparison','target_greedy_replay_exact':True})
    continue
   hs=torch.cat([target_out.hidden_states[i] for i in [1,9,17,25,33]],dim=-1).detach()
   last=raw['h']
   loss_mask=torch.zeros_like(sequence,dtype=torch.float32);loss_mask[:,p:]=1.
   batch=(hs,sequence,loss_mask,last,torch.zeros_like(sequence),None)
   positions=torch.tensor([p+k for k in offsets],device='cuda',dtype=torch.long)
   na=len(offsets)
   anchors=(positions,torch.ones_like(positions,dtype=torch.bool))
   record={'task_index':task_index,'domain':task['domain'],'input_ids':prompt,'output_ids':sequence[0,p:].tolist(),'anchors':positions.tolist(),'widths':{}}
   result['records'].append(record)
   reference_features=None
   for width in range(1,9):
    block=width+1
    draft=extended._backbone_forward(*batch,max_anchors=na,block_size=block,anchors=anchors,return_pre_draft_features=True)
    hidden,logits,targets,mask,indices=draft[:5]
    current_features=draft[5][:2]
    if reference_features is None:reference_features=tuple(t.clone() for t in current_features)
    else:assert all(torch.equal(a,b) for a,b in zip(reference_features,current_features)), ('Width-dependent pre-draft features',task_index,width)
    assert torch.all(mask.view(na,block)[:,1:]==1)
    hidden=hidden.view(na,block,-1);logits=logits.view(na,block,-1)
    prev=sequence[:,positions];proposed=[]
    for j in range(1,block):
     unary=logits[:,j,:].unsqueeze(0);h=hidden[:,j,:].unsqueeze(0)
     candidates=unary.topk(extended.candidate_selector.top_k,dim=-1).indices
     scores=extended.candidate_selector.score_candidates(unary,h,prev,candidates)
     prev=candidates.gather(-1,scores.argmax(-1,keepdim=True)).squeeze(-1)
     proposed.append(prev.squeeze(0))
    proposed=torch.stack(proposed,dim=1)
    truth=sequence[0,positions[:,None]+torch.arange(1,block,device='cuda')[None,:]]
    accepted=(proposed==truth).long().cumprod(-1).sum(-1)
    # Independent target verification on the actual autoregressively selected draft.
    a=int(positions[0]);verify_ids=torch.cat([sequence[:,:a+1],proposed[0:1]],dim=1)
    verified=target_model(verify_ids,use_cache=False).logits[0,a:a+width].argmax(-1)
    observed=(verified==proposed[0]).long().cumprod(-1).sum()
    record['widths'][str(width)]={'accepted':accepted.tolist(),'proposed_ids':proposed.tolist(),'independent_first_anchor_accepted':int(observed),'independent_target_ids':verified.tolist()}
    assert int(observed)==int(accepted[0]), ('Independent target verification mismatch',task_index,width)
    if width==8:
     changed_ids=sequence.clone();changed_ids[:,p+1:]=torch.randint(0,151936,changed_ids[:,p+1:].shape,device='cuda')
     changed_hs=hs.clone();changed_hs[:,p:]=torch.randn_like(changed_hs[:,p:])
     changed_last=last.clone();changed_last[:,p:]=torch.randn_like(changed_last[:,p:])
     altered=extended._backbone_forward(changed_hs,changed_ids,loss_mask,changed_last,batch[4],None,max_anchors=na,block_size=block,anchors=anchors,return_pre_draft_features=True)
     assert torch.equal(draft[1][:,:block],altered[1][:,:block]), ('Future affected draft',task_index)
     assert all(torch.equal(x[0],y[0]) for x,y in zip(draft[5][:2],altered[5][:2])), ('Future affected pre-draft feature',task_index)
     record['future_perturbation_invariant']=True
   features.append({'task_index':task_index,'context':reference_features[0].cpu(),'anchor_embedding':reference_features[1].cpu(),'anchors':positions.cpu()})
   record['target_greedy_replay_exact']=True
   record['features_identical_across_widths']=True
   torch.save(features,r/'labels_v73_features.pt')
   out.write_text(json.dumps(result,indent=2)+'\n')
   print('PASSED_REAL_LABELS',task_index,task['domain'],na,flush=True)
 handle.remove()
 torch.save(features,r/'labels_v73_features.pt')
 result['feature_sha256']=hashlib.sha256((r/'labels_v73_features.pt').read_bytes()).hexdigest()
 result['status']='completed';result['passed']=True
except BaseException as e:
 result.update(status='failed',passed=False,error=repr(e),traceback=traceback.format_exc())
 raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
