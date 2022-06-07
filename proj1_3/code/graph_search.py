from heapq import heappush, heappop  # Recommended.
import numpy as np


from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def min_distance(p1, p2):
    min_distance = np.linalg.norm(p2 - p1)
    return min_distance

def min_metric_distance(p1, p2, map):
    p1_metric = map.index_to_metric_center(p1)
    p2_metric = map.index_to_metric_center(p2)
    min_distance = np.linalg.norm(p2_metric - p1_metric)
    return min_distance

def within_bounds(point, map, margin):
    # check point itself
    if map.is_occupied_index(point):
        #print('point out of bounds')
        return False
    return True

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment
        resolution, xyz resolution in METERS for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in METERS from path to obstacles.
        start,      xyz position in METERS, shape=(3,)
        goal,       xyz position in METERS, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """
    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # print(occ_map.metric_to_index(start))
    # print(occ_map.metric_to_index(goal))

    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start)) # in voxel
    goal_index = tuple(occ_map.metric_to_index(goal)) # in voxel
    step_size = 1 # in voxels
    #occ_map.get_goal_box(goal)

    print('start index', start_index)
    print('goal index', goal_index)
    print('step size', step_size)

    # create priority heap
    pq = []
    # create list of visited nodes
    visited = set()  # defined as set to dodge duplicates
    edge_set = []  # set of parent -> child edges
    path = []  # list of paths taken

    cost = 0
    if astar is True:
        cost = min_distance(np.array(list(start_index)), np.array(list(goal_index)))
    parent = None

    x_step = resolution[0]
    y_step = resolution[1]
    z_step = resolution[2]

    edge = (cost, parent, start_index, cost) # edge defined as location A -> location B, w/ cost X
    heappush(pq, edge)

    # add initial frontier to priority queue
    while goal_index not in visited:
        if len(pq) == 0: # no more frontier nodes
            return None, len(visited)

        (cost, parent, p2, sub_cost) = heappop(pq) # cost of path to p2, path from start to p2, frontier point
        edge_set.append((parent, p2))  # add visited to node to list of parents

        # ignore if p2 has already been visited
        if p2 not in visited:
            # print('now visiting: ', p2)
            visited.add(p2)  # add p2 to visited list

            # goal node successfully found
            #if min_distance(np.array(list(p2)), np.array(list(goal_index))) < 0.01: # to deal with overflow
            if np.linalg.norm(np.array(list(p2)) - np.array(list(goal_index))) < 0.01:
                path.append(p2)
                print("success!")

                # determine final path
                path = [p2]
                while path[-1] != start_index:
                    parent = [a_tuple for (index, a_tuple) in enumerate(edge_set) if a_tuple[1] == path[-1]][0]
                    path.append(parent[0])
                path.reverse()
                #print('path', path)  # in voxel

                #convert path to metric
                path_mod = []
                #path_mod = [np.array(occ_map.index_to_metric_center(i)).astype(float).tolist() for i in enumerate(path)]
                for i in path:
                    segment = (np.array(occ_map.index_to_metric_center(i)).astype(float))
                    path_mod.append(segment.tolist())

                path_mod[0] = start
                path_mod[-1] = goal
                print('path mod', path_mod)
                return np.array(path_mod), len(visited)

            # search for frontier nodes
            for i in [-1*step_size, 0, step_size]:
                for j in [-1*step_size, 0, step_size]:
                    for k in [-1*step_size, 0, step_size]:
                        if i == 0 and j == 0 and k == 0: #if same point
                            pass

                        shift = np.array([i, j, k])
                        frontier_point = np.array(list(p2)) + shift

                        if within_bounds(frontier_point, occ_map, margin): # check if within bounds
                            frontier_cost = sub_cost + np.sqrt(abs(i)*x_step**2 + abs(j)*y_step**2 + abs(k)*z_step**2)

                            astar_cost = 0
                            if astar:
                                #astar_cost = min_distance(np.array(frontier_point), np.array(list(goal_index)))
                                astar_cost = min_metric_distance(np.array(frontier_point), np.array(list(goal_index)), occ_map)

                            # print('cost: ', cost)
                            # print('frontier point', frontier_point)
                            heappush(pq, (frontier_cost+astar_cost, p2, tuple(frontier_point), frontier_cost))

            #exit()

    # Return a tuple (path, nodes_expanded)
    return None, 0
