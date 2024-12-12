import numpy as np
import pybullet as p
import open3d as o3d
import assignment_6_helper as helper
from sklearn.cluster import DBSCAN

def segment_single_object(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=0.015, min_points=10, print_progress=True))
    # num_clusters = labels.max() + 1

    # if num_clusters > 2:
    #     raise ValueError(f"Too many objects")

    object0 = pcd.select_by_index(np.where(labels == 0)[0])
    o3d.visualization.draw_geometries([object0])
    return object0

def check_alignment(normal_i, normal_j, angle_threshold = np.pi / 18):
    alignment = np.dot(normal_i, normal_j)
    if alignment < -np.cos(angle_threshold):  # Normals point towards each other
        return True
    return False

def search_pair(pcd, search_radius = 0.15):
    '''
    returns a Nx4, :3 is the midpoint of pair, 3 is the angle theta of that pair
    '''
    pc_points = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)

    # Antipodal point pairs
    antipodal_pairs = np.zeros((1,5))

    # KDTree for fast neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for i, normal_i in enumerate(pc_normals):
        # Find neighbors for the current point
        [_, idx, _] = kdtree.search_radius_vector_3d(pc_points[i], search_radius)

        for j in idx:
            if i == j:
                continue  # Skip the same point
            
            # Check antipodal condition
            normal_j = pc_normals[j]
            displacement = pc_points[j] - pc_points[i]
            normal_disp = displacement / np.linalg.norm(displacement)

            if check_alignment(normal_i, normal_j) and check_alignment(normal_i, normal_disp):
                this_pair = np.zeros(5)

                dot_product_x = np.dot(normal_i, np.array([1, 0, 0]))
                theta = np.arccos(dot_product_x)

                midpoint = (pc_points[i] + pc_points[j]) / 2
                length = np.linalg.norm(pc_points[j] - pc_points[i])

                this_pair[:3] = midpoint
                this_pair[3] = length
                this_pair[-1] = theta

                antipodal_pairs = np.vstack([antipodal_pairs, this_pair])
    
    # print(antipodal_pairs)
    return antipodal_pairs

def get_cluster(values, eps, min_samples):
    
    angles_reshaped = values.reshape(-1,1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(angles_reshaped)

    # Identify the unique cluster labels (excluding noise, which has label -1)
    unique_labels = np.unique(labels)

    cluster = {}
    
    for label in unique_labels:
        if label != -1:
            cluster_size = np.sum(labels == label)
            cluster_points = values[labels == label]
            cluster_mean = np.mean(cluster_points)

            indices = np.where(labels == label)[0]

            cluster[label] = (cluster_size, cluster_mean, indices)

    return cluster

def get_antipodal(pcd):
    """
    function to compute antipodal grasp given point cloud pcd
    :param pcd: point cloud in open3d format (converted to numpy below)
    :return: gripper pose (4, ) numpy array of gripper pose (x, y, z, theta)
    """
    # convert pcd to numpy arrays of points and normals
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.005)
    # o3d.visualization.draw_geometries([downsampled_pcd])
    
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    object0 = segment_single_object(downsampled_pcd)
    
    pairs = search_pair(object0)
    lengths = pairs[:, 3]

    length_cluster = get_cluster(lengths, eps=0.01, min_samples=2)
    # print(length_cluster)
    min_length_label = min(length_cluster, key=lambda x: length_cluster[x][1])
    indices_of_min_length = length_cluster[min_length_label][2]

    angles = pairs[indices_of_min_length,-1]
    angle_cluster = get_cluster(angles, eps=0.3, min_samples=2)
    most_angle_label = max(angle_cluster, key=lambda x: angle_cluster[x][0])
    theta = angle_cluster[most_angle_label][1]

    

    gripper_pose = np.zeros(4)
    
    # Compute the mean of the points in that cluster
    # gripper orientation - replace 0. with your calculations
    gripper_pose[3] = np.mean(theta)

    # gripper pose: (x, y, z, theta) - replace 0. with your calculations
    gripper_pose[:3] = np.mean(pairs[1:,:3], axis=0)
            
    
    # print(pc_normals[antipodal_pairs[0][0]])
    # print(pc_normals[antipodal_pairs[0][1]])
    # print(pc_points[antipodal_pairs[0][1]] - pc_points[antipodal_pairs[0][0]])
    print(gripper_pose)


    # ------------------------------------------------

    return gripper_pose


def main(n_tries=5):
    # Initialize the world
    world = helper.World()

    # start grasping loop
    # number of tries for grasping
    for i in range(n_tries):
        # get point cloud from cameras in the world
        pcd = world.get_point_cloud()
        # check point cloud to see if there are still objects to remove
        finish_flag = helper.check_pc(pcd)
        if finish_flag:  # if no more objects -- done!
            print('===============')
            print('Scene cleared')
            print('===============')
            break
        # visualize the point cloud from the scene
        helper.draw_pc(pcd)
        # compute antipodal grasp
        gripper_pose = get_antipodal(pcd)
        # send command to robot to execute
        robot_command = world.grasp(gripper_pose)
        # robot drops object to the side
        world.drop_in_bin(robot_command)
        # robot goes to initial configuration and prepares for next grasp
        world.home_arm()
        # go back to the top!

    # terminate simulation environment once you're done!
    p.disconnect()
    return finish_flag


if __name__ == "__main__":
    flag = main()
