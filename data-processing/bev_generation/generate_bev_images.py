import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import save_results, process_scannet_scene, process_scannetpp_scene, process_arkitscenes_scene

def _process_single_scene(args):
    scene_id, dataset, data_path, annotation, save_dir = args
    if dataset == 'scannet':
        bev_image, instance_info = process_scannet_scene(scene_id, data_path, annotation)
    elif dataset == 'scannetpp':
        bev_image, instance_info = process_scannetpp_scene(scene_id, data_path, annotation)
    elif dataset == 'arkitscenes':
        bev_image, instance_info = process_arkitscenes_scene(scene_id, data_path, annotation)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    bev_image_path = os.path.join(save_dir, dataset, 'bev_images', f'{scene_id}.png')
    instance_info_path = os.path.join(save_dir, dataset, 'instance_info', f'{scene_id}.json')
    save_results(bev_image, instance_info, bev_image_path, instance_info_path)

def process_dataset(dataset, data_path, annotation, scene_id_len, save_dir, cpu_num):
    bev_image_dir = os.path.join(save_dir, dataset, 'bev_images')
    instance_info_dir = os.path.join(save_dir, dataset, 'instance_info')
    os.makedirs(bev_image_dir, exist_ok=True)
    os.makedirs(instance_info_dir, exist_ok=True)

    plys = [x for x in os.listdir(os.path.join(data_path, 'data')) if x.endswith('.ply')]
    scene_ids = [x[:scene_id_len] for x in plys]

    args_list = [(scene_id, dataset, data_path, annotation, save_dir) for scene_id in scene_ids]
    with Pool(processes=min(cpu_count(), cpu_num)) as pool:
        list(tqdm(pool.imap_unordered(_process_single_scene, args_list), total=len(args_list)))

def main():
    parser = argparse.ArgumentParser(description='Process dataset for BEV and instance info.')
    parser.add_argument('--data_path', type=str, default='../data', help='Root path to datasets (should contain scannet/, scannetpp/, arkitscenes/)')
    parser.add_argument('--dataset', type=str, choices=['scannet', 'scannetpp', 'arkitscenes', 'all'], default='all', help='Which dataset to process')
    parser.add_argument('--annotation', type=str, default='gt_annotation', help='Annotation type to use')
    parser.add_argument('--save_dir', type=str, default='../data/struct2d_info', help='Directory to save BEV images and instance info')
    parser.add_argument('--cpu_num', type=int, default=48, help='Number of CPU cores to use')

    args = parser.parse_args()

    dataset_info = {
        'scannet':     {'scene_id_len': 12, 'path': os.path.join(args.data_path, 'scannet')},
        'scannetpp':   {'scene_id_len': 10, 'path': os.path.join(args.data_path, 'scannetpp')},
        'arkitscenes': {'scene_id_len': 8,  'path': os.path.join(args.data_path, 'arkitscenes')},
    }

    target_datasets = dataset_info.keys() if args.dataset == 'all' else [args.dataset]
    for ds in target_datasets:
        print(f"\nProcessing {ds} ...")
        info = dataset_info[ds]
        process_dataset(
            dataset=ds,
            data_path=info['path'],
            annotation=args.annotation,
            scene_id_len=info['scene_id_len'],
            save_dir=args.save_dir,
            cpu_num=args.cpu_num
        )

if __name__ == '__main__':
    main()