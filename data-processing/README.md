## Data Processing

### 🔽 Download 3D Scenes and Annotations

To generate bird’s-eye-view (BEV) images and project 3D object marks, you need to download three indoor scene datasets:

- [ARKitScenes](https://github.com/apple/ARKitScenes)
- [ScanNet](https://github.com/ScanNet/ScanNet)
- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)

After downloading, organize the datasets in the following directory structure:

```
.
├── arkitscenes
│   ├── data
│   │   ├── ...
│   │   └── {scene_id}_3dod_mesh_wo_ceiling.ply          # point cloud files
│   ├── gt_annotation                                     # ground truth annotations
│   │   └── {scene_id}_3dod_annotation.json
│   └── detection_annotation                              # noisy detection annotations
│
├── scannet
│   ├── data
│   │   ├── ...
│   │   └── {scene_id}_vh_clean_2.ply
│   ├── gt_annotation
│   │   ├── {scene_id}.aggregation.json
│   │   └── {scene_id}_vh_clean_2.0.010000.segs.json
│   └── detection_annotation
│
├── scannetpp
│   ├── data
│   │   ├── ...
│   │   └── {scene_id}_mesh_aligned_0.05.ply
│   ├── gt_annotation
│   │   ├── {scene_id}_segments_anno.json
│   │   └── {scene_id}_segments.json
│   └── detection_annotation
└── ...

```

### Generate BEV Images for VSI-Bench Evaluation

Use the script below to generate BEV images and save 3D object bounding boxes as JSON files:

```
python generate_bev_images.py --dataset all
```