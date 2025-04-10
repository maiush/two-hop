import os, argparse, subprocess

import pandas as pd
import torch as t
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tqdm import trange
from twohop.constants import DATA_PATH

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_vllm(
        args: argparse.Namespace
) -> None:
    # === LOAD MODEL ===
    llm_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": 0.98,
        "tensor_parallel_size": t.cuda.device_count(),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "seed": 123456,
        "task": "generate",
    }
    if args.lora:
        print(f"applying LoRA adapter: {args.lora}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 16
    model = LLM(**llm_kwargs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=1,
        skip_special_tokens=False,
        logprobs=5,
        temperature=args.temperature,
        top_p=args.top_p,
        use_tqdm=False,
    )
    # === LOAD DATASET AND PREPROCESS PROMPTS ===
    path = f"{DATA_PATH}/current_test.jsonl"
    dataset = pd.read_json(path, orient="records", lines=True)
    prompts = dataset["messages"].to_list()
    prompts = [
        p[:p.rindex("\n")+1]
        for p in prompts
    ]
    labels = dataset["answer"].astype(str).to_list()
    # === GENERATE ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "lora_request": LoRARequest("adapter", 1, lora_path=args.lora) if args.lora else None
    }
    scores = []
    for _ in trange(args.N):
        outputs = model.generate(**gen_kwargs)
        # === PREDICTIONS ===
        predictions = []
        for output in outputs:
            # grab logits
            valid_tks = ["True", "False"]
            prediction = None
            logprobs = output.outputs[0].logprobs
            if logprobs:
                for _, logprob in logprobs[0].items():
                    if logprob.decoded_token.strip() in valid_tks:
                        prediction = logprob.decoded_token.strip()
                        break
            predictions.append(prediction)
        # === SCORE ===
        score = sum(
            p == l
            for p, l in zip(predictions, labels)
        ) / len(predictions)
        scores.append(score)
    return sum(scores) / len(scores)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="original")
    args = parser.parse_args()

    command = f"python preprocess.py --split test --prefix {args.prefix}"
    subprocess.run(command, shell=True)

    score = generate_vllm(args)
    print("="*100)
    print(f"SCORE: {score:.4f}")
    print("="*100)