import os
import yaml
from loguru import logger as eval_logger
import numpy as np
import pandas as pd
from PIL import Image
import json
from collections import defaultdict
import cv2
from utils import *
from functools import partial

video_folder = None
doc_data_folder = None
lmms_eval_kwargs = None

def update(cfg):
    global video_folder, doc_data_folder, lmms_eval_kwargs
    video_folder = cfg['video_folder']
    doc_data_folder = cfg['doc_data_folder']
    lmms_eval_kwargs = cfg['lmms_eval_specific_kwargs']

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

def get_object_bbox_string(doc_id, dataset, scene_name, question, question_type):
    file_path = os.path.join(doc_data_folder, f"doc_{doc_id}/instance_info.json")
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_objs = json.load(json_file)

    all_objs = transform_label(dataset, all_objs)
    label_to_ids = defaultdict(set)
    for instance_id, obj in all_objs.items():
        label_to_ids[obj["label"].lower()].add(instance_id)

    bbox_strings = []
    filtered_objs, _ = extract_labels_and_filter_objs(question, question_type, label_to_ids, all_objs)
    for obj in filtered_objs:
        bbox_pts = obj['OBB_points']
        label = obj['label']
        bbox_strings.append(f'{label}: {bbox_pts}')

    result = f"The bounding box corner points for the objects are: {', '.join(bbox_strings)}."
    return result

# for room size estimation
def get_centroid_distance(doc_id, dataset, scene_name):
    file_path = os.path.join(doc_data_folder, f"doc_{doc_id}/instance_info.json")
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_objs = json.load(json_file)

    all_objs = transform_label(dataset, all_objs)
    selected_objs = [{**{"id": key}, **value} for key, value in all_objs.items() if value['label'] not in ['wall', 'floor', 'ceiling']][:2]
    if len(selected_objs) < 2:
        return "Directly estimate the size of the room as there are less than two objects."
    p0 = np.array(selected_objs[0]['world_position'][:2]) # only compute the distance on the xy plane
    p1 = np.array(selected_objs[1]['world_position'][:2])
    dist = np.linalg.norm(p0 - p1)

    result = f"The distance (in meters) between the marks ({selected_objs[0]['id']} and {selected_objs[1]['id']}) is {dist}."
    return result

def get_label_string(doc_id, dataset, scene_name, question, question_type, mode):
    file_path = os.path.join(doc_data_folder, f"doc_{doc_id}/instance_info.json")
  
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_objs = json.load(json_file)
    if question_type in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard', 'route_planning']:
        labels_string = ", ".join(f"{item['id']}:{item['label']}" for item in all_objs)
    else:
        all_objs = transform_label(dataset, all_objs)
        label_to_ids = defaultdict(set)
        for instance_id, obj in all_objs.items():
            label_to_ids[obj["label"].lower()].add(instance_id)

        filtered_objs, _ = extract_labels_and_filter_objs(question, question_type, label_to_ids, all_objs)
        labels_string = ", ".join(f"{item['id']}:{item['label']}" for item in filtered_objs)
    result = f"The category names of object marks are {labels_string}."

    return result

def get_object_centroid_string(doc_id, dataset, scene_name, question, question_type):
    file_path = os.path.join(doc_data_folder, f"doc_{doc_id}/instance_info.json")
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_objs = json.load(json_file)

    all_objs = transform_label(dataset, all_objs)
    label_to_ids = defaultdict(set)
    for instance_id, obj in all_objs.items():
        label_to_ids[obj["label"].lower()].add(instance_id)

    filtered_objs, _ = extract_labels_and_filter_objs(question, question_type, label_to_ids, all_objs)
    centroid_string = ", ".join(f"{item['label']}:{item['id']}:{str([round(x, 2) for x in item['world_position']])}" for item in filtered_objs)

    result = f"The center coordinates of the objects in 3D point cloud are {centroid_string}."
    return result

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def process_results(doc, result):
    doc['prediction'] = result
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    if 'object_rel_direction_easy_accuracy' in output:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.

def doc_to_visual(doc, mode, num_frames=8):
    lmms_eval_specific_kwargs = lmms_eval_kwargs[mode]
    dataset = doc['dataset']
    question = doc['question']
    question_type = doc['question_type']
    doc_id = doc['id']
    if dataset != 'scannet':
        scene_prefix = doc["scene_name"].split('_')[0]
    else:
        scene_prefix = doc["scene_name"][:len("scenexxxx_xx")]

    if mode == 'video':
        video_path = dataset + "/" + doc["scene_name"] + ".mp4"
        video_path = os.path.join(video_folder, video_path)
        if os.path.exists(video_path):
            video_path = video_path
        else:
            raise FileExistsError(f"video path:{video_path} does not exist.")
        return [video_path]
    
    elif mode == "struct2d" and lmms_eval_specific_kwargs.get("rotation", False):
        images = []

        if doc['question_type'] in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard', 'route_planning']:
            bev_image_folder_id = os.path.join(doc_data_folder, f"doc_{doc['id']}")

            bev_image_path = os.path.join(bev_image_folder_id, 'bev_image.png')
            bev_image = Image.open(bev_image_path).convert('RGB')
            images.append(bev_image)
        else: 
            bev_image_path = os.path.join(doc_data_folder, f"doc_{doc['id']}", 'bev_image.png')
            bev_image = Image.open(bev_image_path).convert('RGB')
            bev_image = cv2.cvtColor(np.array(bev_image), cv2.COLOR_RGB2BGR)
            
            obj_info_path = os.path.join(doc_data_folder, f"doc_{doc['id']}", 'instance_info.json')
            with open(obj_info_path, 'r') as f:
                bev_objs = json.load(f)
            
            bev_objs = transform_label(dataset, bev_objs)
            label_to_ids = defaultdict(set)
            for instance_id, obj in bev_objs.items():
                label_to_ids[obj["label"].lower()].add(instance_id)
            selected_objs, _ = extract_labels_and_filter_objs(question, question_type, label_to_ids, bev_objs)

            # draw filtered marks and crop image
            marked_image = draw_marks(bev_image, selected_objs)
            cropped_image = crop_image(marked_image)

            bev_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            images.append(bev_image)
        return images

def doc_to_text(doc, mode):
    question = doc["question"]
    question_type = doc["question_type"]
    scene_name = doc['scene_name']
    dataset = doc['dataset']
    doc_id = doc['id']
    lmms_eval_specific_kwargs = lmms_eval_kwargs[mode]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")

    if mode == 'video':
        if doc['question_type'] in NA_QUESTION_TYPES:
            post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") 
            return pre_prompt + "\n" + question + "\n" + post_prompt
        elif doc['question_type'] in MCA_QUESTION_TYPES:
            options = "Options:\n" + "\n".join(doc["options"])
            post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") 
            return "\n".join([pre_prompt, question, options, post_prompt])
        else:
            raise ValueError(f"Unknown question type: {doc['question_type']}")
    elif mode == 'struct2d':
        if lmms_eval_specific_kwargs.get("use_labels", False):
            label_prompt = get_label_string(doc['id'], dataset, scene_name, question, question_type, mode)
        else:
            label_prompt = ""

        if doc['question_type'] in NA_QUESTION_TYPES:
            post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "")

            if lmms_eval_specific_kwargs.get("use_bbox", False):
                if doc['question_type'] == 'object_abs_distance':
                    bbox_string = get_object_bbox_string(doc['id'], dataset, scene_name, question, question_type)
                    label_prompt += f"\n{bbox_string}"
                    guide_prompt = "You are given the 3D bounding box coordinates (8 corner points) of two marked objects in the scene. Each point is in meters. Your task is to compute or estimate the shortest distance between the two objects by finding the minimum distance between any pair of corner points, one from each bounding box. Use these coordinates directly to estimate the closest possible distance between the two objects."
                    return "\n".join([pre_prompt, guide_prompt, label_prompt, question, post_prompt])

            if lmms_eval_specific_kwargs.get("use_coord", False):
                if doc['question_type'] == 'room_size_estimation':
                    distance_string = get_centroid_distance(doc['id'], dataset, scene_name)
                    label_prompt += f"\n{distance_string}"
                    guide_prompt = "You are given the distance (in meters) between two marked objects in the image. Use this distance as a reference scale. First, estimate the relative width and length of the room by comparing them visually to the marked distance. For example, if the distance is x meters, and from the image, the room appears approximately a times wider and b times longer than this distance, then estimate estimate the width to be a times x meters, and the length to be b times x meters. Then compute the area using a times b times x^2."
                    return "\n".join([pre_prompt, guide_prompt, label_prompt, question, post_prompt])
            return pre_prompt + "\n" + label_prompt + "\n" + question + "\n" + post_prompt
        elif doc['question_type'] in MCA_QUESTION_TYPES:
            options = "Options:\n" + "\n".join(doc["options"])
            post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "")

            if lmms_eval_specific_kwargs.get("use_coord", False):
                if doc['question_type'] == 'object_rel_distance':
                    centroid_string = get_object_centroid_string(doc['id'], dataset, scene_name, question, question_type)
                    label_prompt += f"\n{centroid_string}"
                    guide_prompt = "Please also refer to the given object coordinates (object class name: mark number: 3D coordinates). If there are multiple instances of an object category, the minimum absolute distance to the primary object is used."
                    return "\n".join([pre_prompt, guide_prompt, label_prompt, question, options, post_prompt])

            if lmms_eval_specific_kwargs.get("direction_guide", False):
                if doc['question_type'] in ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard']:
                    guide_prompt = "Follow the steps below to complete the task accurately.\n1. Draw Line A from the point where I am standing to the object I am facing.\n2. Draw Line B, which is perpendicular to Line A and intersects it at my standing position.\n3. To determine the object's relative position in the question:\n\tCheck which side of Line B the object is on.\n\tIf the object is on the same side as the object I am facing, it is in the front; otherwise, it is in the back.\n4. Next, determine the object's position relative to Line A:\n\tIf it is to the left of Line A, it is on the left.\n\tIf it is to the right of Line A, it is on the right.\n\tLeft and right should always be determined based on the direction in which I am facing. You can imagine rotate the image to make the standing to object and facing object being vertical, which may help you get the answer more accurate."
                    return "\n".join([pre_prompt, label_prompt, question, options, guide_prompt, post_prompt])

            if lmms_eval_specific_kwargs.get("planning_guide", False):
                if doc['question_type'] == 'route_planning':
                    guide_prompt = "To navigate towards an object in your route, ensure that the object is directly in front of you. You may have to imagine rotating the given view image to get the correct turning direction. If it is not, determine whether you need to turn left or right to align yourself with it before proceeding forward."
                    return "\n".join([pre_prompt, label_prompt, question, options, guide_prompt, post_prompt])

            return "\n".join([pre_prompt, label_prompt, question, options, post_prompt])
        else:
            raise ValueError(f"Unknown question type: {doc['question_type']}")