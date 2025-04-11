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
        "enforce_eager": True
    }
    if args.lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 16
    model = LLM(**llm_kwargs)
    # === SET SAMPLING PARAMS ===
    sampling_params = SamplingParams(
        max_tokens=1,
        skip_special_tokens=False,
        logprobs=5,
        temperature=args.temperature,
        top_p=args.top_p
    )
    # === LOAD DATASET AND PREPROCESS PROMPTS ===
    path = f"{DATA_PATH}/geometry_of_truth/test.jsonl"
    dataset = pd.read_json(path, orient="records", lines=True)
    prompts = dataset["messages"].to_list()
    prompts = [
        p[:p.rindex("\n")+1]
        for p in prompts
    ]
    # === GENERATE ===
    gen_kwargs = {
        "prompts": prompts,
        "sampling_params": sampling_params,
        "lora_request": LoRARequest("adapter", 1, lora_path=args.lora) if args.lora else None,
        "use_tqdm": False
    }
    if gen_kwargs["lora_request"]: print(f"using LoRA adapter: {args.lora}")
    scores_trues, scores_lies = [], []
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
        labels = dataset["is_city"].to_list()
        answers = dataset["label"].to_list()
        answers = ["True" if a == 1 else "False" for a in answers]
        lies, trues = [], []
        for idx in range(len(predictions)):
            if not labels[idx]:
                trues.append(predictions[idx] == answers[idx])
            else:
                lies.append(predictions[idx] == answers[idx])
        score = sum(trues) / len(trues)
        scores_trues.append(score)
        score = sum(lies) / len(lies)
        scores_lies.append(score)
    return sum(scores_trues) / len(scores_trues), sum(scores_lies) / len(scores_lies)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    score_trues, score_lies = generate_vllm(args)
    print("="*100)
    print(f"TRUE: {score_trues:.4f}")
    print(f"LIE: {score_lies:.4f}")
    print("="*100)
