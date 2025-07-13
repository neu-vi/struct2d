import os
import json
import time
import base64
import yaml
import tqdm
import argparse
import numpy as np

from io import BytesIO
from PIL import Image
from typing import List
from copy import deepcopy
from collections import Counter
from multiprocessing import Pool, cpu_count

from datasets import load_dataset
import requests as url_requests

import vsibench

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="zero-shot evaluation with prompting")
    parser.add_argument("--config", type=str, required=True, help="path to YAML config file")
    parser.add_argument("--mode", type=str, default="struct2d", help="visual mode to use (e.g. video, struct2d)")
    parser.add_argument("--model-version", type=str, default="o3", help="chatgpt model version")
    parser.add_argument("--subset-id-path", type=str, default="./subset_ids/rel_direction.json", help="path to subset JSON file")
    parser.add_argument("--log-dir", type=str, default="./logs", help="directory to save log files")
    parser.add_argument("--num-frames", type=int, default=16, help="number of video frames to sample")
    parser.add_argument("--num-votes", type=int, default=5, help="number of votes per sample")
    parser.add_argument("--num-workers", type=int, default=16, help="number of workers")
    return parser.parse_args()

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def encode_video(video_path: str, frame_count: int) -> List[str]:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int).tolist()
    if total_frames - 1 not in frame_indices:
        frame_indices.append(total_frames - 1)

    frames = vr.get_batch(frame_indices).asnumpy()
    base64_frames = []

    for frame in frames:
        img = Image.fromarray(frame)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return base64_frames

def encode_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_test_docs(dataset_path, subset_path=None):
    dataset = load_dataset(dataset_path)['test']
    if subset_path:
        with open(subset_path, "r") as f:
            subset_ids = json.load(f)
        dataset = dataset.filter(lambda x: x['id'] in subset_ids)
    return dataset

def process_doc(doc):
    doc_id = str(doc['id'])
    if doc_id in log_dict:
        return doc_id, log_dict[doc_id]

    visuals = flatten([vsibench.doc_to_visual(doc, args.mode, args.num_frames)])
    images_encoded = []

    for visual in visuals:
        try:
            if args.mode == 'video':
                images_encoded.extend(encode_video(visual, args.num_frames))
            else:
                images_encoded.append(encode_image(visual))
        except Exception as e:
            print(f"[ERROR] Failed to encode visual: {visual} — {e}")
            return doc_id, ""

    context = vsibench.doc_to_text(doc, args.mode)
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": context}] +
                   [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images_encoded]
    }]

    payload = {
        "model": args.model_version,
        "messages": messages,
        "seed": 42,
    }

    if args.model_version.startswith("gpt-4o"):
        payload.update({
            "max_tokens": generation_kwargs['max_new_tokens'],
            "temperature": generation_kwargs['temperature'],
            "top_p": generation_kwargs['top_p'],
        })
    elif args.model_version.startswith("o3"):
        payload["max_completion_tokens"] = generation_kwargs['max_new_tokens']
    else:
        raise ValueError(f"Unsupported model version: {args.model_version}")

    responses = []
    for vote_idx in range(args.num_votes):
        for attempt in range(3):
            try:
                response = url_requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
                response_text = response.json()["choices"][0]["message"]["content"].strip()
                print(response_text)
                responses.append(response_text)
                break
            except Exception as e:
                print(f"[Vote {vote_idx+1} - Retry {attempt+1}] Error for doc {doc_id}: {e}")
                time.sleep(NUM_SECONDS_TO_SLEEP)
        else:
            responses.append("")

    filtered = [r for r in responses if r]

    if filtered and doc['question_type'] in vsibench.MCA_QUESTION_TYPES:
        answer, _ = Counter(filtered).most_common(1)[0]
    elif filtered and doc['question_type'] in vsibench.NA_QUESTION_TYPES:
        valid_vals = [float(r) for r in filtered if r.replace(".", "", 1).isdigit()]
        answer = str(sum(valid_vals) / len(valid_vals)) if valid_vals else ""
    else:
        answer = ""

    return doc_id, answer

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # pdate vsibench with config
    vsibench.update(config)

    generation_kwargs = config['generation_kwargs']
    dataset_path = config['dataset_path']

    # API setup
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    NUM_SECONDS_TO_SLEEP = 30
    TIMEOUT = 300

    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = f"log_{args.model_version}_{args.mode}_{args.subset_id_path.split('/')[-1].split('.')[0]}.json"
    log_path = os.path.join(args.log_dir, log_filename)

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_dict = json.load(f)
    else:
        log_dict = {}

    test_docs = get_test_docs(dataset_path, args.subset_id_path)

    # multiprocessing
    with Pool(processes=min(cpu_count(), args.num_workers)) as pool:
        results = list(tqdm.tqdm(pool.imap(process_doc, test_docs), total=len(test_docs)))

    for doc_id, result in results:
        log_dict[doc_id] = result
    with open(log_path, 'w') as f:
        json.dump(log_dict, f)

    # metrics
    eval_scores = []
    for doc in test_docs:
        doc_id = str(doc['id'])
        result = log_dict.get(doc_id, "")
        score = vsibench.process_results(doc, result)["vsibench_score"]
        eval_scores.append(score)

    final_result = vsibench.vsibench_aggregate_results(eval_scores)