"""
OSZ/modules/
============
Core OSZ computation modules.

  depth_estimator.py  : Monocular depth prediction (MiDaS v2.1 Small) + LiDAR alignment
  image_to_ego.py     : Back-project camera depth maps to ego-frame 3D points
  bev_height_builder.py : Build BEV height maps from camera + LiDAR point clouds
  ray_casting.py      : 3D voxel casting + height-aware 2D ego-centric ray casting
  drivable_filter.py  : Intersect OSZ with HD-map drivable area
"""
