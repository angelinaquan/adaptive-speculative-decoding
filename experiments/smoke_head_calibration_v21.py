"""GPU integration gate only; repeated validation prompts are not efficacy evidence."""
import json
import os
from pathlib import Path


def main():
    os.environ['VLLM_BATCH_INVARIANT']='1'
    import numpy as np
    import wandb
    from vllm import LLM,SamplingParams
    root=Path(__file__).resolve().parent
    scratch=Path(os.environ['SERVING_SCRATCH'])
    paths=json.loads((root/'model_paths.json').read_text())
    tasks=json.loads((root/'serving_v16_manifest.json').read_text())
    prior=json.loads((root/'parity_diagnostic_v20_ar.json').read_text())
    assert prior['status']=='completed' and prior['batch_invariant']
    reference=[row['ids'] for row in prior['records'] if row['repeat']==0]
    config=dict(model=paths['Qwen/Qwen3-4B'],dtype='float32',max_model_len=2048,
        max_num_seqs=32,max_num_batched_tokens=4096,gpu_memory_utilization=.65,enable_prefix_caching=False,attention_config={'backend':'TRITON_ATTN'},
        worker_extension_cls='head_calibration_v17.HeadCalibrationWorker',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]},
        speculative_config=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],
            num_speculative_tokens=7,draft_sample_method='probabilistic',enable_adaptive_verification=True))
    run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='head-calibration-v21',
        name='gpu-integration-gate',dir=str(scratch),config={'scope':'Correctness/integration only; no speed or calibration efficacy claim','engine':config})
    result=dict(passed=False,states={},run_url=run.url)
    try:
        llm=LLM(**config)
        tokenizer=llm.get_tokenizer()
        prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['text']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False) for r in tasks]
        def control(**kw):
            states=llm.collective_rpc('calibration_control_v17',kwargs=kw)
            assert len(states)==1
            return states[0]
        def generate():
            for start in range(0,len(prompts),8):
                outputs=llm.generate(prompts[start:start+8],SamplingParams(temperature=0.,max_tokens=128),use_tqdm=False)
                assert [list(o.outputs[0].token_ids) for o in outputs]==reference[start:start+8], ('Greedy mismatch',start)
        control(learning=False,slope=1.,intercepts=[0.]*7,reset_statistics=True)
        generate()
        frozen=control();result['states']['identity_frozen']=frozen
        assert frozen['scores'][0]>0 and frozen['refits']==0
        assert abs(frozen['scores'][1]-frozen['scores'][2])<1e-3
        control(learning=True,reset_statistics=True,refit_interval=4)
        generate()
        trained=control();result['states']['online']=trained
        assert trained['refits']>0
        assert np.isfinite(trained['slope']+trained['intercepts']+trained['scores']).all()
        assert abs(trained['slope'][0]-1)+sum(abs(x) for x in trained['intercepts'])>1e-6
        control(learning=False,reset_statistics=True)
        generate()
        frozen_after=control();result['states']['frozen_after_update']=frozen_after
        assert frozen_after['slope']==trained['slope'] and frozen_after['intercepts']==trained['intercepts']
        assert frozen_after['refits']==0
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        output=scratch/'head-calibration-v21-smoke.json'
        output.write_text(json.dumps(result,indent=2)+'\n')
        (root/'head_calibration_v21_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(result)
        artifact=wandb.Artifact('head-calibration-v21-integration','correctness')
        for path in [output,Path(__file__),root/'head_calibration_v17.py',root/'test_head_calibration_v17.py']:
            artifact.add_file(str(path))
        run.log_artifact(artifact);run.finish()
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
