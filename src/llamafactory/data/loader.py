# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image
import cv2
from io import BytesIO
import math
import uuid

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

def draw_marks(image, objects, alpha=0.5, base_square_size=36):
    overlay = image.copy()

    for obj_data in objects:
        obj_id = str(obj_data['id'])  # Ensure the ID is a string
        x, y = map(int, obj_data["image_position"])

        # Adjust rectangle width based on ID length
        square_width = base_square_size + 10 * (len(obj_id) - 1)
        square_height = base_square_size

        top_left = (x - square_width // 2, y - square_height // 2)
        bottom_right = (x + square_width // 2, y + square_height // 2)

        # Draw red rectangle
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)

    # Blend overlay with image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Add ID number centered in the rectangle
    for obj_data in objects:
        obj_id = str(obj_data['id'])
        x, y = map(int, obj_data["image_position"])

        # Determine text size
        font_scale = 0.9
        text_size = cv2.getTextSize(obj_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(image, obj_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

    return image

def draw_marks_kf(image, objects, alpha=0.5, base_square_size=36):
    # 1. draw the marks
    overlay = image.copy()

    for obj_data in objects:
        obj_id = str(obj_data['id'])  # Ensure the ID is a string

        x, y = map(int, obj_data["world_position"])

        # Adjust rectangle width based on ID length
        square_width = base_square_size + 10 * (len(obj_id) - 1)
        square_height = base_square_size

        top_left = (x - square_width // 2, y - square_height // 2)
        bottom_right = (x + square_width // 2, y + square_height // 2)

        # Draw red rectangle
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)

    # Blend overlay with image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Add ID number centered in the rectangle
    for obj_data in objects:
        obj_id = str(obj_data['id'])
        x, y = map(int, obj_data["world_position"])

        # Determine text size
        font_scale = 0.9
        text_size = cv2.getTextSize(obj_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(image, obj_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)

    return image

def crop_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a mask of white areas
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours of the non-white areas
    contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No significant content detected in the image, skipping cropping.")
        return image
    
    # Get bounding box of all non-white areas
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    # Crop the image
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

image_folder = '/projects/vig/Datasets' # for eval

# image_folder = '/projects/vig/Datasets/Spatial-Reasoning'
frame_folder = '/projects/vig/Datasets/ScanNet/scans_uncomp'
# frame_folder = "/projects/vig/Datasets/Spatial-Reasoning/train_video_frames_2x4x2"

scannet_folder = "/projects/vig/Datasets/ScanNet/scans_uncomp"
# arkitscenes_folder = "/projects/vig/Datasets/ARKitScenes/3dod/Training"
# scannetpp_folder = "/projects/vig/Datasets"
# video_242_folder = "/projects/vig/Datasets/Spatial-Reasoning/train_video_frames_2x4x2"
# video_242_folder = "/projects/vig/Datasets/Spatial-Reasoning/val_video_frames_2x4x2"
# def _draw_crop_and_buffer_image(example):
#     processed_images = []
#     objects = example['_metadata']['object_info']

#     for img_path in example['_images']:
#         full_path = os.path.join(image_folder, img_path)
#         bev_image = cv2.imread(full_path)
#         if bev_image is None:
#             raise FileNotFoundError(f"Image not found or unreadable: {full_path}")
#         marked_image = draw_marks(bev_image, objects)
#         cropped_image = crop_image(marked_image)

#         resized_image = cv2.resize(cropped_image, (640, 640), interpolation=cv2.INTER_LINEAR)
       
#         # Encode final image to PNG in memory
#         success, encoded_image = cv2.imencode('.png', resized_image)
#         if not success:
#             raise RuntimeError("Failed to encode image as PNG")

#         buffer = BytesIO(encoded_image.tobytes())
        
#         processed_images.append({
#             "bytes": buffer.read(),
#             "path": full_path  # optional, useful if downstream still logs path
#         })

#     example["_images"] = processed_images
#     return example


def compute_best_grid(num_images):
    """Compute the best grid layout (rows, cols) for a given number of images."""
    best_diff = float('inf')
    best_shape = (1, num_images)
    for rows in range(1, num_images + 1):
        cols = math.ceil(num_images / rows)
        diff = abs(rows - cols)
        if rows * cols >= num_images and diff < best_diff:
            best_diff = diff
            best_shape = (rows, cols)
    return best_shape

def _draw_crop_and_buffer_image_kf(example):
    processed_image_paths = []
    objects_list = example['_metadata']['object_info']
    image_paths = example['_images']

    # === Process first image (e.g., BEV) ===
    if image_paths[0] is not None:
        full_path = os.path.join(image_folder, image_paths[0])
        bev_image = cv2.imread(full_path)
        if bev_image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {full_path}")
        marked_image = draw_marks(bev_image, objects_list)  # draw all objects on BEV
        cropped_image = crop_image(marked_image)
        resized_image = cv2.resize(cropped_image, (480, 480), interpolation=cv2.INTER_LINEAR)

        save_path = f"/scratch/zhu.fang/processed_images/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, resized_image)
        processed_image_paths.append(save_path)

    # === Process and mark all remaining images ===
    # === Keyframe images ===
    cnt = 1
    for obj_data in objects_list:
        if obj_data["world_position"] == [-1.0, -1.0]:
            continue
        if cnt >= len(image_paths):
            break
        image_path = image_paths[cnt]
        cnt += 1

        full_path = os.path.join(frame_folder, image_path)
        raw_image = cv2.imread(full_path)
        if raw_image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {full_path}")

        marked_image = draw_marks_kf(raw_image, [obj_data])
        resized_image = cv2.resize(marked_image, (512, 492), interpolation=cv2.INTER_LINEAR)

        save_path = f"/scratch/zhu.fang/processed_images/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, resized_image)
        processed_image_paths.append(save_path)

    example["_images"] = processed_image_paths
    return example


def _draw_crop_and_buffer_image(example):
    processed_images = []
    objects = example['_metadata']['object_info']
    image_paths = example['_images']

    for idx, image_path in enumerate(image_paths):
        if image_path is None:
            continue

        if idx == 0:
            # Process the first image with drawing and cropping
            full_path = os.path.join(image_folder, image_path)
            bev_image = cv2.imread(full_path)
            if bev_image is None:
                raise FileNotFoundError(f"Image not found or unreadable: {full_path}")
            marked_image = draw_marks(bev_image, objects)
            cropped_image = crop_image(marked_image)
            resized_image = cv2.resize(cropped_image, (480, 480), interpolation=cv2.INTER_LINEAR)
        else:
            # Process other images with only resizing
            full_path = os.path.join(frame_folder, image_path)
            raw_image = cv2.imread(full_path)
            if raw_image is None:
                raise FileNotFoundError(f"Image not found or unreadable: {full_path}")
            resized_image = cv2.resize(raw_image, (768, 384), interpolation=cv2.INTER_LINEAR)

        success, encoded_image = cv2.imencode('.png', resized_image)
        if not success:
            raise RuntimeError(f"Failed to encode image as PNG: {full_path}")

        buffer = BytesIO(encoded_image.tobytes())
        processed_images.append({
            "bytes": buffer.read(),
            "path": full_path
        })

    example["_images"] = processed_images
    return example

def _process_path_and_resize_buffer_image(example):
    processed_images = []
    # dataset = example['_metadata']['dataset']
    image_paths = example['_images']

    bev_image_path = image_paths[0]
    bev_image = cv2.imread(bev_image_path)

    resized_image = cv2.resize(bev_image, (480, 480), interpolation=cv2.INTER_LINEAR)
    success, encoded_image = cv2.imencode('.png', resized_image)
    if not success:
        raise RuntimeError("Failed to encode second image as PNG")

    buffer = BytesIO(encoded_image.tobytes())
    processed_images.append({
        "bytes": buffer.read(),
        "path": bev_image_path
    })

    frame_image_path = image_paths[1]
    frame_image = cv2.imread(frame_image_path)
    resized_image = cv2.resize(frame_image, (768, 364), interpolation=cv2.INTER_LINEAR)
    success, encoded_image = cv2.imencode('.png', resized_image)
    if not success:
        raise RuntimeError("Failed to encode second image as PNG")

    buffer = BytesIO(encoded_image.tobytes())
    processed_images.append({
        "bytes": buffer.read(),
        "path": frame_image_path
    })

    if len(image_paths) > 2:
        frame_image_path = image_paths[2]
        frame_image = cv2.imread(frame_image_path)
        resized_image = cv2.resize(frame_image, (768, 364), interpolation=cv2.INTER_LINEAR)
        success, encoded_image = cv2.imencode('.png', resized_image)
        if not success:
            raise RuntimeError("Failed to encode second image as PNG")

        buffer = BytesIO(encoded_image.tobytes())
        processed_images.append({
            "bytes": buffer.read(),
            "path": frame_image_path
        })

    example["_images"] = processed_images
    return example

def _process_path_and_buffer_image(example):
    processed_images = []
    # dataset = example['_metadata']['dataset']
    image_paths = example['_images']

    # if dataset == 'scannet':
    #     image_folder = scannet_folder
    # elif dataset == 'arkitscenes':
    #     image_folder = arkitscenes_folder
    # elif dataset == 'scannetpp':
    #     image_folder = scannetpp_folder
    image_folder = video_242_folder

    for image_path in image_paths:
        if image_path is not None:
            full_path = os.path.join(image_folder, image_path)
            raw_image = cv2.imread(full_path)
            # full_path = image_path
            # raw_image = cv2.imread(full_path)
            if raw_image is None:
                raise FileNotFoundError(f"image not found or unreadable: {full_path}")

            resized_image = cv2.resize(raw_image, (768, 369), interpolation=cv2.INTER_LINEAR)
            success, encoded_image = cv2.imencode('.png', resized_image)
            if not success:
                raise RuntimeError("Failed to encode second image as PNG")

            buffer = BytesIO(encoded_image.tobytes())
            processed_images.append({
                "bytes": buffer.read(),
                "path": full_path
            })

    example["_images"] = processed_images
    return example

def _process_path(example):
    processed_images = []

    image_paths = example['_images']
    image_folder = video_242_folder

    for image_path in image_paths:
        if image_path is not None:
            full_path = os.path.join(image_folder, image_path)
            processed_images.append(full_path)
    example['_images'] = processed_images
    return example


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None
    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None
    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    # draw marks and crop the bev image -- added 
    # run vsibench setting train and eval
    # dataset = dataset.map(_draw_crop_and_buffer_image, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=False, desc="Draw crop and buffer image")
    dataset = dataset.map(_draw_crop_and_buffer_image_kf, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=False, desc="Draw crop and buffer image")
    # dataset = dataset.map(_proscess_path_and_buffer_image, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=False, desc="Process path and buffer image")
    # dataset = dataset.map(_process_path, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=False, desc="Process path")

    # scannet all train (scannet-sft-short-wkf-train)
    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)