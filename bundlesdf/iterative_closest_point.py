import open3d as o3d
import numpy as np
import os
import glob
import cv2
from copy import deepcopy


def extract_floats_from_camk(lines):
    # Initialize an empty list to store the extracted floats
    extracted_floats = []

    # Iterate through each line
    for line in lines:
        # Split the line into individual elements using spaces
        elements = line.split()

        # Convert each element to a float and append to the list
        floats = [float(element) for element in elements]
        extracted_floats.append(floats)
    return extracted_floats


def load_mesh_from_obj(obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    return mesh


def get_image_files_from_folder(folder_path):
    return sorted(glob.glob(os.path.join(folder_path, "*.png")))


def load_depth_and_mask_from_paths(old_depth_path, proj_depth_path, old_mask_path, proj_mask_path):
    old_depth_image = o3d.io.read_image(old_depth_path)
    old_mask_image = o3d.io.read_image(old_mask_path)
    proj_depth_image = o3d.io.read_image(proj_depth_path)
    proj_mask_image = o3d.io.read_image(proj_mask_path)
    return old_depth_image, proj_depth_image, old_mask_image, proj_mask_image


def depth_to_point_cloud(depth_values, intrinsic_matrix, mask_values):
    depth_values = np.array(depth_values)#.astype(float)
    mask_values = np.array(mask_values).flatten()

    # Create a grid of coordinates
    rows, cols = depth_values.shape
    u_coords, v_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    u_coords, v_coords = u_coords.flatten(), v_coords.flatten()

    # if sc_factor is not None:
    #     depth_values /= sc_factor
    depth_values = depth_values.flatten()
    depth_background, mask_background = depth_values[0], mask_values[0]

    print('depth_values before', len(depth_values))
    u_coords = u_coords[np.logical_and(mask_values != mask_background, depth_values != depth_background)]
    v_coords = v_coords[np.logical_and(mask_values != mask_background, depth_values != depth_background)]
    depth_values = depth_values[np.logical_and(mask_values != mask_background, depth_values != depth_background)]
    print('depth_values after', len(depth_values))

    # u_coords = u_coords[depth_values > 0]
    # v_coords = v_coords[depth_values > 0]
    # depth_values = depth_values[depth_values > 0]
    # if target:
    #     np.savetxt('old_depth.txt', depth_values, fmt='%d', delimiter=', ')
    # else:
    #     np.savetxt('proj_depth.txt', depth_values, fmt='%d', delimiter=', ')
    # print('depth_values at depth_to_point_cloud', np.min(depth_values), depth_values)

    # Apply the inverse intrinsic matrix to get 3D coordinates
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values

    # Stack coordinates into a point cloud array
    point_cloud_data = np.stack((x_coords, y_coords, z_coords), axis=-1)

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    return point_cloud, depth_values


def normalize_point_cloud(pc: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    # Deep copy to avoid modifications on the original point cloud
    pc_copy = deepcopy(pc)

    # Convert points to numpy array
    points = np.asarray(pc_copy.points)

    # Compute the centroid of the point cloud
    centroid = points.mean(axis=0)

    # Translate the point cloud to the origin
    points -= centroid

    # Scale the point cloud to fit in the range [0, 255]
    max_distance = np.max(np.linalg.norm(points, axis=1))
    scaling_factor = 255.0 / max_distance
    points *= scaling_factor

    # Shift values to start from 0
    points -= np.min(points)

    # Update the points of the pc_copy with normalized points
    pc_copy.points = o3d.utility.Vector3dVector(points)

    return pc_copy


# def inverse_normalize_point_cloud(pc: o3d.geometry.PointCloud, original_pc: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
#     # Deep copy to avoid modifications on the original point cloud
#     pc_copy = deepcopy(pc)
#
#     # Convert points to numpy array
#     points = np.asarray(pc_copy.points)
#
#     # Compute the centroid of the original point cloud
#     original_centroid = np.asarray(original_pc.points).mean(axis=0)
#
#     # Scale the points back
#     max_distance_original = np.max(np.linalg.norm(original_pc.points - original_centroid, axis=1))
#     scaling_factor = max_distance_original / 255.0
#     points /= scaling_factor
#
#     # Translate the points back to their original positions
#     points += original_centroid
#
#     # Update the points of the pc_copy with inverse-normalized points
#     pc_copy.points = o3d.utility.Vector3dVector(points)
#
#     return pc_copy


def register_with_icp(source, target):
    # Clone the original_pointcloud using the copy constructor
    source = normalize_point_cloud(source)
    target = normalize_point_cloud(target)

    # Convert to CUDA PointClouds
    source_cuda = o3d.cuda.pybind.geometry.PointCloud()
    target_cuda = o3d.cuda.pybind.geometry.PointCloud()
    source_cuda.points = source.points
    target_cuda.points = target.points

    # To save the point clouds
    source_out = o3d.geometry.PointCloud()
    source_out.points = source.points
    print('source_out.points', np.asarray(source_out.points), np.min(np.asarray(source_out.points)), np.mean(np.asarray(source_out.points)), np.max(np.asarray(source_out.points)))

    target_out = o3d.geometry.PointCloud()
    target_out.points = target.points
    print('target_out.points', np.asarray(target_out.points), np.min(np.asarray(target_out.points)), np.mean(np.asarray(target_out.points)), np.max(np.asarray(target_out.points)))

    # ICP Registration
    reg_p2p = o3d.cuda.pybind.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.cuda.pybind.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, max_iteration=1000)
    result = o3d.cuda.pybind.pipelines.registration.registration_icp(source_cuda, target_cuda, 0.02, np.eye(4), reg_p2p, criteria)

    return result


def fill_depth_holes(depth_image, max_hole_size=5):
    """
    Fill small holes in the depth map.
    :param depth_image: Input depth image with holes.
    :param max_hole_size: Maximum hole size to be filled.
    :return: Depth image with holes filled.
    """
    # Find holes in the depth image
    holes_mask = depth_image == 0

    # Fill holes
    filled_depth = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE,
                                    kernel=np.ones((max_hole_size, max_hole_size), dtype=np.uint8))
    filled_depth[~holes_mask] = depth_image[~holes_mask]

    return filled_depth


def point_cloud_to_depth_image(point_cloud, intrinsic_matrix, depth_shape):
    depth_image = np.zeros(depth_shape, dtype=np.float32)

    # transform the depth map to the range of original depth map, not the proj ones
    # point_cloud = inverse_normalize_point_cloud(point_cloud, ref_point_cloud)
    # print('scaled_new_depth_point_cloud', np.asarray(point_cloud.points))

    for point in point_cloud.points:
        p = np.dot(intrinsic_matrix, point)
        u, v, _ = p / p[2]
        u, v = int(u), int(v)
        # print('u', u, depth_shape[1], 'v', v, depth_shape[0])

        if 0 <= u < depth_shape[1] and 0 <= v < depth_shape[0]:
            depth_value = point[2]
            if depth_image[v, u] == 0 or depth_value < depth_image[v, u]:
                depth_image[v, u] = depth_value

    # Interpolate the depth image to fill zero-valued pixels
    depth_image = fill_depth_holes(depth_image)

    return depth_image


def main():
    old_depth_folder = "data/old_toss_1/depth"
    proj_depth_folder = "data/old_toss_1_contactnet/depth"
    old_mask_folder = "data/old_toss_1/masks"
    proj_mask_folder = "data/old_toss_1_contactnet/masks"
    cam_K_file = "results/old_toss_1/cam_K.txt"
    output_depth_folder = "data/old_toss_1_icp/depth"
    output_mask_folder = "data/old_toss_1_icp/masks"

    # sc_factor = 8.697657471427803

    # Read the file and extract intrinsic matrix values
    with open(cam_K_file, 'r') as f:
        lines = f.readlines()

    lines = extract_floats_from_camk(lines)
    fx, fy, cx, cy = float(lines[0][0]), float(lines[1][1]), float(lines[0][2]), float(lines[1][2])

    # Create the intrinsic matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

    # Make sure output directories exist
    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    old_depth_files = get_image_files_from_folder(old_depth_folder)
    old_mask_files = get_image_files_from_folder(old_mask_folder)
    proj_depth_files = get_image_files_from_folder(proj_depth_folder)
    proj_mask_files = get_image_files_from_folder(proj_mask_folder)

    # Assuming the same number of depth and mask images
    for old_depth_file, old_mask_file, proj_depth_file, proj_mask_file in zip(old_depth_files, old_mask_files, proj_depth_files, proj_mask_files):
        old_depth_image, proj_depth_image, old_mask_image, proj_mask_image = load_depth_and_mask_from_paths(old_depth_file, proj_depth_file, old_mask_file, proj_mask_file)

        # Convert depth image to point cloud
        old_depth_point_cloud, old_depth_values = depth_to_point_cloud(old_depth_image, intrinsic_matrix, old_mask_image)
        print('old_depth_point_cloud', np.asarray(old_depth_point_cloud.points))
        proj_depth_point_cloud, proj_depth_values = depth_to_point_cloud(proj_depth_image, intrinsic_matrix, proj_mask_image)
        print('proj_depth_point_cloud', np.asarray(proj_depth_point_cloud.points))

        # Register with ICP
        result = register_with_icp(proj_depth_point_cloud, old_depth_point_cloud)   # source, target
        print('file', old_depth_file, 'result', result)
        print('result.transformation', result.transformation)

        # Apply transformations from ICP to refine depth
        transformed_point_cloud = proj_depth_point_cloud.transform(result.transformation)

        # # Compute normals for the point cloud (required for surface reconstruction)
        # transformed_point_cloud.estimate_normals()
        # # Use Poisson surface reconstruction to convert point cloud to mesh
        # transformed_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(transformed_point_cloud, depth=8)
        # # Save the resulting mesh as an .obj file
        # o3d.io.write_triangle_mesh("transformed_point_cloud.obj", transformed_mesh)

        # Convert transformed point cloud back to depth image
        new_depth_image_data = point_cloud_to_depth_image(transformed_point_cloud, intrinsic_matrix, np.asarray(proj_depth_image).shape)

        # Update mask based on transformed depth
        new_mask_image_data = (new_depth_image_data > 0).astype(np.uint8) * 255

        # Convert numpy arrays to Open3D images and save
        new_depth_image = o3d.geometry.Image(new_depth_image_data)
        new_mask_image = o3d.geometry.Image(new_mask_image_data)

        basename = os.path.basename(old_depth_file)
        o3d.io.write_image(os.path.join(output_depth_folder, basename), new_depth_image)
        o3d.io.write_image(os.path.join(output_mask_folder, basename), new_mask_image)

        break


if __name__ == "__main__":
    main()
