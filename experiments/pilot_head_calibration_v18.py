"""Matched-initialization validation pilot; no held-out efficacy claim."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['ar','calibration'],required=True)
    args=parser.parse_args()
    os.environ['VLLM_BATCH_INVARIANT']='1'
    import numpy as np
    import wandb
    from vllm import LLM,SamplingParams
    root=Path(__file__).resolve().parent;scratch=Path(os.environ['SERVING_SCRATCH'])
    plan=json.loads((root/'head_calibration_v18_plan.json').read_text())
    contents=(root/'head_calibration_v18_manifest.json').read_bytes()
    assert hashlib.sha256(contents).hexdigest()==plan['manifest_sha256']
    manifest=json.loads(contents);paths=json.loads((root/'model_paths.json').read_text())
    assert json.loads((root/'head_calibration_v21_smoke.json').read_text())['passed']
    config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,max_num_seqs=32,
        max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]})
    if args.mode=='calibration':
        config.update(worker_extension_cls='head_calibration_v17.HeadCalibrationWorker',
            speculative_config=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],
                num_speculative_tokens=7,draft_sample_method='probabilistic',enable_adaptive_verification=True))
    run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='head-calibration-v18',
        name=args.mode+'-validation',dir=str(scratch),config={'plan':plan,'engine':config})
    result={'status':'initializing','run_url':run.url,'records':[],'source_trajectory':[],'phase_scores':[],
            'plan':plan,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output=scratch/f'head-calibration-v18-{args.mode}.json'
    assert not output.exists(),'Refusing to overwrite a previous pilot'
    try:
        llm=LLM(**config);tokenizer=llm.get_tokenizer()
        params=SamplingParams(temperature=0.,max_tokens=128)
        def encode(rows):
            prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['text']}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False) for r in rows]
            assert all(len(tokenizer.encode(p))+128<=2048 for p in prompts)
            return prompts
        source=encode(manifest['source']);validation=encode(manifest['validation'])
        def generate(prompts):
            return llm.generate(prompts,params,use_tqdm=False)
        def ids(outputs):return [list(o.outputs[0].token_ids) for o in outputs]
        def control(**kwargs):
            values=llm.collective_rpc('calibration_control_v17',kwargs=kwargs)
            assert len(values)==1
            return values[0]
        if args.mode=='ar':
            result['outputs']=[]
            for start in range(0,len(validation),8):
                result['outputs'].extend(ids(generate(validation[start:start+8])))
            result['status']='completed'
            return
        reference=json.loads((scratch/'head-calibration-v18-ar.json').read_text())
        assert reference['status']=='completed' and reference['plan']['manifest_sha256']==plan['manifest_sha256']
        expected=reference['outputs']
        # Warm all validation shapes before fitting; learning is explicitly off.
        control(learning=False,slope=1.,intercepts=[0.]*7,reset_statistics=True)
        for start in range(0,len(validation),8):
            assert ids(generate(validation[start:start+8]))==expected[start:start+8]
        control(learning=True,reset_statistics=True,refit_interval=100)
        for epoch in range(3):
            for start in range(0,len(source),8):generate(source[start:start+8])
            state=control();result['source_trajectory'].append(state)
            assert np.isfinite(state['slope']+state['intercepts']).all()
            print('SOURCE_EPOCH',epoch,json.dumps(state),flush=True)
        fitted=control(learning=False)
        assert fitted['refits']>0
        result['source_fit']=fitted
        # Exercise the update kernel before timing, then reset to the source fit.
        control(learning=True,refit_interval=25)
        generate(validation[:8])
        for round_index,order in enumerate(plan['round_orders']):
            for method in order:
                interval=25 if method=='online25' else 100
                control(learning=method!='frozen',slope=fitted['slope'][0],intercepts=fitted['intercepts'],
                        reset_statistics=True,refit_interval=interval)
                for domain in plan['validation_domains']:
                    indices=[i for i,row in enumerate(manifest['validation']) if row['domain']==domain]
                    before=control()
                    for offset in range(0,len(indices),8):
                        selected=indices[offset:offset+8]
                        start=time.perf_counter();generated=generate([validation[i] for i in selected]);seconds=time.perf_counter()-start
                        token_ids=ids(generated)
                        assert token_ids==[expected[i] for i in selected],('greedy mismatch',method,domain,selected)
                        tokens=sum(map(len,token_ids))
                        row=dict(round=round_index,method=method,domain=domain,indices=selected,
                                 seconds=seconds,tokens=tokens,tokens_per_second=tokens/seconds)
                        result['records'].append(row);run.log({k:v for k,v in row.items() if k!='indices'})
                    after=control()
                    assert np.isfinite(after['slope']+after['intercepts']+after['scores']).all()
                    if method=='frozen':
                        assert after['slope']==fitted['slope'] and after['intercepts']==fitted['intercepts']
                    result['phase_scores'].append(dict(round=round_index,method=method,domain=domain,
                        observed_score_delta=(np.array(after['scores'])-np.array(before['scores'])).tolist(),
                        state=after))
                    print('PHASE',round_index,method,domain,flush=True)
        result['status']='completed'
    except BaseException as error:
        result['status']='failed';result['error']=repr(error);raise
    finally:
        output.write_text(json.dumps(result,indent=2)+'\n')
        (root/f'head_calibration_v18_{args.mode}.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary['status']=result['status']
        artifact=wandb.Artifact('head-calibration-v18-'+args.mode,'validation-pilot')
        for path in [output,Path(__file__),root/'head_calibration_v17.py',root/'head_calibration_v18_plan.json',root/'head_calibration_v18_manifest.json']:
            artifact.add_file(str(path))
        run.log_artifact(artifact);run.finish()


if __name__=='__main__':main()
