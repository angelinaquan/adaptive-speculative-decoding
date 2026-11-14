"""Export unchanged released backbone with explicitly UNTRAINED heads for forced-policy profiling."""
import json,hashlib,traceback,subprocess,os
from pathlib import Path
r=Path(__file__).resolve().parent
out=r/'export_v68_checks.json';assert not out.exists()
result={'status':'initializing','scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model':'RedHatAI/Qwen3-4B-speculator.dflash2','revision':'27aa911ddb040c61adeeade08e92949e24c9e17b','checks':[]}
try:
 assert json.loads((r/'labels_v67_checks.json').read_text())['passed']
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

 destination=root/'checkpoints'/'dflash2-4b-untrained-heads-v68'
 assert not destination.exists()
 extended.save_pretrained(destination,safe_serialization=True)
 result.update(status='completed',passed=True,checkpoint=str(destination),original_checkpoint=str(checkpoint),UNTRAINED_HEADS=True)
 (destination/'UNTRAINED_HEADS.json').write_text(json.dumps({'scope':'Forced-policy controls only. Both added heads are untrained; not an adaptive baseline.','source_sha256':result['source_sha256'],'original_revision':result['revision']},indent=2)+'\n')
except BaseException as e:
 result.update(status='failed',passed=False,error=repr(e),traceback=traceback.format_exc())
 raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
