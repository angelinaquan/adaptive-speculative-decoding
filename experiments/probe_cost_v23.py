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
        worker_extension_cls='diagnostic_cost_v23.CostWorker',
        compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]},
        speculative_config=dict(method='dspark',model=paths['deepseek-ai/dspark_qwen3_4b_block7'],
            num_speculative_tokens=7,draft_sample_method='probabilistic',enable_adaptive_verification=True))
    run=wandb.init(entity='aquan-world-model',project='18657-speculative-pilot',group='cost-v23',
        name='gpu-integration-gate',dir=str(scratch),config={'scope':'Correctness/integration only; no speed or calibration efficacy claim','engine':config})
    result=dict(passed=False,states={},run_url=run.url)
    try:
        llm=LLM(**config)
        tokenizer=llm.get_tokenizer()
        selected=list(range(8))+list(range(88,96))
        tasks=[tasks[i] for i in selected]
        reference=[reference[i] for i in selected]
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
        result['profile']=llm.collective_rpc('cost_profile_v23')[0]
        generate()
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        output=scratch/'cost-v23-smoke.json'
        output.write_text(json.dumps(result,indent=2)+'\n')
        (root/'cost_v23_results.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(result)
        artifact=wandb.Artifact('cost-v23-integration','correctness')
        for path in [output,Path(__file__),root/'head_calibration_v17.py',root/'diagnostic_cost_v23.py']:
            artifact.add_file(str(path))
        run.log_artifact(artifact);run.finish()
        print('COST_PROFILE_COMPLETE',result['passed'],flush=True)


if __name__=='__main__':main()
