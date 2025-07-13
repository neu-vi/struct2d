import os
import numpy as np
from PIL import Image
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import cv2
import re
import string

def transform_label(dataset, instance_info):
    rules = {
        "arkitscenes": {
            "tv_monitor": "tv"
        },
        "scannet": {
            "object": "discard_object",
            "desk": "table",
            "trash can": "trash bin",
            "recycling bin": "trash bin",
            "office chair": "chair",
            "dining table": "table",
            "mouse": "computer mouse",
            "armchair": "chair",
            "couch": "sofa",
            "kitchen counter": "counter",
            "beanbag chair": "chair",
            "desk lamp": "lamp",
            "lamp base": "lamp",
            "night lamp": "lamp",
        },
        "scannetpp": {
            "object": "discard_object",
            "ceiling lamp": "ceiling light",
            "soap dispenser": "hand soap",
            "office chair": "chair",
            "dining table": "table",
            "dining chair": "chair",
            "coat hanger": "coat rack",
            "mouse": "computer mouse",
            "mouse pad": "computer mouse",
            "armchair": "chair",
        }
    }

    dataset_rules = rules[dataset]
    for instance_id, obj in instance_info.items():
        label = obj["label"]
        if label in dataset_rules:
            instance_info[instance_id]["label"] = dataset_rules[label]

    return instance_info


def filter_objs(question, label_to_ids, all_objs):
    selected_labels = []
    words = word_tokenize(question.lower())
    bigrams = [' '.join(bg) for bg in ngrams(words, 2)]
    
    for label, ids in label_to_ids.items():
        if (label in words or label in bigrams) and label not in selected_labels:
            selected_labels.append(label)  
    selected_objs = [{**{"id": key}, **value} for key, value in all_objs.items() if value['label'] in selected_labels]

    return selected_objs

def extract_labels(question, question_type):
    object_list = []
    if 'object_rel_direction' in question_type:
        pattern = r"I am standing by the (.*?) and facing the (.*?), is the (.*?) to"
        match = re.search(pattern, question)
        if match:
            standing = match.group(1).strip()
            facing = match.group(2).strip()
            target = match.group(3).strip()
            object_list = [standing, facing, target]
    
    elif question_type == 'object_rel_distance':
        list_match = re.search(r"\((.*?)\)", question)
        object_list = [obj.strip() for obj in list_match.group(1).split(',')] if list_match else []
        reference_match = re.search(r"closest to the ([a-zA-Z0-9_ ]+)", question)
        reference_object = reference_match.group(1).strip() if reference_match else None
        if reference_object is not None:
            object_list.append(reference_object)

    elif question_type == 'obj_appearance_order':
        list_match = re.search(r"categories.*?:\s*(.*)", question)
        object_list = [obj.strip().rstrip(string.punctuation) for obj in list_match.group(1).split(',')] if list_match else []

    elif question_type == 'object_counting':
        obj_match = re.search(r"How many (.+?)\(s\)", question)
        if obj_match:
            object_list = [obj_match.group(1)]

    elif question_type == 'object_size_estimation':
        obj_match = re.search(r"of the ([^,]+?), measured", question)
        if obj_match:
            object_list = [obj_match.group(1)]

    elif question_type == 'object_abs_distance':
        pattern = r"distance between the (.+?) and the (.+?) \(in meters\)\?"
        match = re.search(pattern, question)
        if match:
            src_obj = match.group(1)
            tgt_obj = match.group(2)
            object_list = [src_obj, tgt_obj]

    elif question_type == 'route_planning':
        location_obj, facing_obj, target_obj = None, None, None
        location_pattern = r"beginning (?:at|by) the (.*?)(?:\s+and\s+facing|\s+facing|,|$)"
        match = re.search(location_pattern, question)
        if match:
            location_obj = match.group(1)
        facing_pattern = r"facing(?:\s+(?:the|to|toward the|into the|out the))?\s+(.*?)(?:\.|\?|$)"
        match = re.search(facing_pattern, question)
        if match:
            facing_obj = match.group(1)
        target_pattern = r"navigate to the (.*?)(?:\.|\?|$)"
        match = re.search(target_pattern, question)
        if match:
            target_obj = match.group(1)
        object_list = [location_obj, facing_obj, target_obj]
        object_list = [x for x in object_list if x]

    return object_list

def filter_objs_with_labels(question_labels, label_to_ids, all_objs):
    selected_labels = []
    for label, ids in label_to_ids.items():
        if label in question_labels and label not in selected_labels:
            selected_labels.append(label)
    selected_objs = [{**{"id": key}, **value} for key, value in all_objs.items() if value['label'] in selected_labels]
    return selected_objs

def extract_labels_and_filter_objs(question, question_type, label_to_ids, all_objs):

    question_labels = extract_labels(question, question_type)
    return filter_objs_with_labels(question_labels, label_to_ids, all_objs), question_labels



def draw_marks(image, objects, alpha=0.5, base_square_size=36):
    overlay = image.copy()

    for obj_data in objects:
        obj_id = str(obj_data['id']) 
        x, y = map(int, obj_data["image_position"])

        square_width = base_square_size + 10 * (len(obj_id) - 1)
        square_height = base_square_size

        top_left = (x - square_width // 2, y - square_height // 2)
        bottom_right = (x + square_width // 2, y + square_height // 2)

        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    for obj_data in objects:
        obj_id = str(obj_data['id'])
        x, y = map(int, obj_data["image_position"])

        font_scale = 0.9
        text_size = cv2.getTextSize(obj_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(image, obj_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

    return image

def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No significant content detected in the image, skipping cropping.")
        return image

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def get_images_from_folder(folder, img_suffix='.png'):
    images = []
    if os.path.exists(folder) and os.path.isdir(folder):
        img_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(img_suffix)])

        for img_file in img_files:
            try:
                with Image.open(img_file) as img:
                    img = img.convert('RGB')
                    images.append(img)  # Pass opened image
            except Exception as e:
                print(f"Error encoding image {img_file}: {e}")
    else:
        print(f"Warning: frame image folder {folder} does not exist.")

    return images