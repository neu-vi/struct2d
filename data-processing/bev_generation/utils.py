import os
import json
import numpy as np
import open3d as o3d

# Dataset utils
def load_ply(ply_file):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    return mesh

# ScanNet + ScanNet++ utils
def load_segmentation(seg_file):
    with open(seg_file, 'r') as f:
        seg_data = json.load(f)
    return np.array(seg_data['segIndices'])

def load_aggregation(agg_file):
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)

    seg_to_label = {}
    instance_to_segments = {}
    instance_to_label = {}

    for obj in agg_data['segGroups']:
        label = obj['label']
        segments = obj['segments']
        instance_id = obj['id']
        for seg_id in obj['segments']:
            seg_to_label[seg_id] = label
        instance_to_segments[instance_id] = segments
        instance_to_label[instance_id] = label
        
    return seg_to_label, instance_to_segments, instance_to_label

def filter_ceiling(seg_ids, seg_to_label):
    filtered_mask = np.array([seg_to_label.get(seg, '') != 'ceiling' for seg in seg_ids])
    return filtered_mask

def compute_instance_centers(xyz, seg_ids, instance_to_segments, use_obb=True):
    instance_centers = {}
    instance_OBBs = {}

    for instance_id, segment_list in instance_to_segments.items():
        mask = np.isin(seg_ids, segment_list)
        if mask.sum() >= 100:
            instance_points = xyz[mask]
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(instance_points))
            if use_obb:
                instance_center = obb.get_center()
            else:
                instance_center = instance_points.mean(axis=0)

            instance_centers[instance_id] = instance_center
            instance_OBBs[instance_id] = np.asarray(obb.get_box_points())
        
    return instance_centers, instance_OBBs

# ARKitScenes utils
def load_annotation(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['data']

def compute_obb_vertices(centroid, axes_lengths, normalized_axes):
    centroid = np.array(centroid)
    axes_lengths = np.array(axes_lengths) / 2  # Convert to half extents
    normalized_axes = np.array(normalized_axes).reshape(3, 3)

    # Compute half extents along each principal axis
    half_extents = normalized_axes * axes_lengths[:, np.newaxis]

    # Define offsets for 8 vertices
    offsets = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ])

    # Compute vertices
    vertices = np.array([centroid + np.sum(offsets[i, :, np.newaxis] * half_extents, axis=0) for i in range(8)])
    
    return vertices

# BEV utils
def project_point_to_image(point, camera_param):
    intrinsic = camera_param.intrinsic.intrinsic_matrix
    extrinsic = camera_param.extrinsic
    
    point_homo = np.append(point, 1)
    point_camera = extrinsic @ point_homo
    
    point_image = intrinsic @ point_camera[:3]
    point_image = point_image / point_image[2]
    
    return np.array([point_image[0], point_image[1]])

def create_bev_visualization(mesh, instance_centers, instance_OBBs, instance_to_label, window_size=(1600, 1600)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=window_size[0], height=window_size[1])
    
    vis.add_geometry(mesh)
    
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, 1])  # look from top to bottom
    # view_ctl.set_up([1, 0, 0])
    view_ctl.set_lookat(mesh.get_center())
    camera_param = view_ctl.convert_to_pinhole_camera_parameters()
    
    vis.poll_events()
    vis.update_renderer()
    
    rgb_img = np.asarray(vis.capture_screen_float_buffer())

    instance_info = []
    for instance_id, center in instance_centers.items():
        label = instance_to_label[instance_id]
        image_pos = project_point_to_image(center, camera_param)
        instance_info.append({
            'id': instance_id,
            'label': label,
            'world_position': center.tolist(),
            'OBB_points': instance_OBBs[instance_id].tolist(),
            'image_position': image_pos.tolist()
        })

    return rgb_img, instance_info

def create_bev_visualization_arkitscenes(mesh, instance_annotation, window_size=(1600,1600)):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=window_size[0], height=window_size[1])
    
    vis.add_geometry(mesh)

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, 1])  # look from top to bottom
    # view_ctl.set_up([1, 0, 0])
    view_ctl.set_lookat(mesh.get_center())
    camera_param = view_ctl.convert_to_pinhole_camera_parameters()
    
    vis.poll_events()
    vis.update_renderer()
    
    rgb_img = np.asarray(vis.capture_screen_float_buffer())

    instance_info = []
    for instance_id, instance in enumerate(instance_annotation):
        label = instance['label']
        obb_aligned = instance['segments']['obbAligned']

        centroid = obb_aligned['centroid']
        axes_length = obb_aligned['axesLengths']
        normalized_axes = obb_aligned['normalizedAxes']

        obb_vertices = compute_obb_vertices(centroid, axes_length, normalized_axes)

        image_pos = project_point_to_image(centroid, camera_param)

        instance_info.append({
            'id': instance_id,
            'label': label,
            'world_position': centroid,
            'OBB_points': obb_vertices.tolist(),
            'image_position': image_pos.tolist()
        })

    return rgb_img, instance_info

def save_results(rgb_img, instance_info, image_path, info_path):
    rgb_img = (rgb_img * 255).astype(np.uint8)
    o3d.io.write_image(image_path, o3d.geometry.Image(rgb_img))
    
    with open(info_path, 'w') as f:
        json.dump(instance_info, f, indent=4)

def process_scannet_scene(scene_id, base_path='../data/scannet', annotation='gt_annotation'):
    ply_file = os.path.join(base_path, f'data/{scene_id}_vh_clean_2.ply')
    seg_file = os.path.join(base_path, f'{annotation}/{scene_id}_vh_clean_2.0.010000.segs.json')
    agg_file = os.path.join(base_path, f'{annotation}/{scene_id}.aggregation.json')

    # load data
    mesh = load_ply(ply_file)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    seg_ids = load_segmentation(seg_file)
    seg_to_label, instance_to_segments, instance_to_label = load_aggregation(agg_file)

    # remove ceiling using ground truth
    filtered_mask = filter_ceiling(seg_ids, seg_to_label)
    new_indices = np.where(filtered_mask)[0]

    # Create a mapping from old vertex indices to new ones
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}

    new_vertices = vertices[filtered_mask]
    new_colors = colors[filtered_mask]

    triangles = np.asarray(mesh.triangles)
    valid_triangles = np.all(np.isin(triangles, new_indices), axis=1)
    filtered_triangles = triangles[valid_triangles]
    new_triangles = np.array([[index_map[v] for v in tri] for tri in filtered_triangles])
        
    seg_ids = seg_ids[filtered_mask]

    # compute the instance centers
    instance_centers, instance_OBBs = compute_instance_centers(new_vertices, seg_ids, instance_to_segments)
    
    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    cropped_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    bev_image, instance_info = create_bev_visualization(cropped_mesh, instance_centers, instance_OBBs, instance_to_label)
    return bev_image, instance_info

def process_scannetpp_scene(scene_id, base_path='../data/scannetpp', annotation='gt_annotation'):
    ply_file = os.path.join(base_path, f'data/{scene_id}_mesh_aligned_0.05.ply')
    seg_file = os.path.join(base_path, f'{annotation}/{scene_id}_segments.json')
    agg_file = os.path.join(base_path, f'{annotation}/{scene_id}_segments_anno.json')

    mesh = load_ply(ply_file)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    seg_ids = load_segmentation(seg_file)
    seg_to_label, instance_to_segments, instance_to_label = load_aggregation(agg_file)

    filtered_mask = filter_ceiling(seg_ids, seg_to_label)
    new_indices = np.where(filtered_mask)[0]

    # remove ceiling using ground truth
    filtered_mask = filter_ceiling(seg_ids, seg_to_label)
    new_indices = np.where(filtered_mask)[0]
    # Create a mapping from old vertex indices to new ones
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_indices)}

    new_vertices = vertices[filtered_mask]
    new_colors = colors[filtered_mask]

    triangles = np.asarray(mesh.triangles)
    valid_triangles = np.all(np.isin(triangles, new_indices), axis=1)
    filtered_triangles = triangles[valid_triangles]
    new_triangles = np.array([[index_map[v] for v in tri] for tri in filtered_triangles])
            
    seg_ids = seg_ids[filtered_mask]

    # compute the instance centers
    instance_centers, instance_OBBs = compute_instance_centers(new_vertices, seg_ids, instance_to_segments)
        
    cropped_mesh = o3d.geometry.TriangleMesh()
    cropped_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    cropped_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    bev_image, instance_info = create_bev_visualization(cropped_mesh, instance_centers, instance_OBBs, instance_to_label)
    return bev_image, instance_info

def process_arkitscenes_scene(scene_id, base_path='../data/arkitscenes', annotation='gt_annotation'):
    mesh = load_ply(f'{base_path}/data/{scene_id}_3dod_mesh_wo_ceiling.ply')
    annotation = load_annotation(f'{base_path}/{annotation}/{scene_id}_3dod_annotation.json')
    bev_image, instance_info = create_bev_visualization_arkitscenes(mesh, annotation)
    return bev_image, instance_info 