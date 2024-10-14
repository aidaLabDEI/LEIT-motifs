from base import *
from find_bin_width import *
import numpy as np
import queue, threading, itertools
from multiprocessing import Pool, cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from hash_lsh import RandomProjection, euclidean_hash
import cProfile
from stop import stop3
#import networkx as nx, matplotlib.pyplot as plt, plotly.graph_objects as go

def worker(i, j, subsequences, hash_mat, ordering, k, stop_i, failure_thresh):
        #if i == 0 and j == 1:
         #   pr = cProfile.Profile()
         #   pr.enable()
        if stop_i:
            return
        #top_temp, dist_comp_temp = cycle(i, j, windowed_ts, hash_mat, ordering, k, failure_thresh)
        dist_comp = 0
        top = queue.PriorityQueue()
        window = subsequences.w
        n = subsequences.num_sub
        dimensionality = subsequences.dimensionality
        motif_dimensionality = subsequences.d
        L = subsequences.L
        K = subsequences.K 
        bin_width = subsequences.r
        dimensions = np.arange(dimensionality, dtype=np.int32)
        counter = dict()
        
        hash_mat_curr = hash_mat[:,j,:,:-i] if i != 0 else hash_mat[:,j,:,:]
        # Let's assume that ordering has the lexigraphical order of the dimensions
        for curr_dim in range(dimensionality):
            ordering_dim = ordering[curr_dim,:,j]
            ordered_view = hash_mat_curr[ordering_dim,curr_dim,:]
            # Take the subsequent elements of the ordering and check if their hash is the same
            for idx, elem1 in enumerate(ordered_view):
                for idx2, elem2 in enumerate(ordered_view[idx+1:]):
                    sub_idx1 = ordering_dim[idx]
                    sub_idx2 = ordering_dim[idx+idx2+1]
                    # No trivial match
                    if (abs(sub_idx1 - sub_idx2) <= window):
                        continue
                    # If same hash, increase the counter, see the next
                    if (elem1 == elem2).all():
                        counter.setdefault((sub_idx1, sub_idx2), 0)
                        counter[sub_idx1, sub_idx2] += 1
                    # Else skip because we know that the ordering ensures that all the subsequences are different
                    else:
                        break
    # Get all entries whose counter is above or equal the motif dimensionality
        #v_max = max(list(counter.values()))
        #counter_extr = [pair for pair, v in counter.items() if v >= v_max]
        counter_extr = [pair for pair, v in counter.items() if v >= motif_dimensionality]

    # Find the set of dimensions with the minimal distance
        for maximum_pair in counter_extr:
            coll_0, coll_1 = maximum_pair
            # Iƒ we already seen it in a key of greater length, skip
            
            if not i == 0:
                rows = hash_mat[coll_0,j,:,:-i+1] == hash_mat[coll_1,j,:,:-i+1]
                comp = np.sum(np.all(rows, axis=1))
                if comp >= motif_dimensionality:
                    continue            
            dist_comp += 1
            curr_dist, dim, stop_dist= z_normalized_euclidean_distance(subsequences.sub(coll_0), subsequences.sub(coll_1),
                                                dimensions, subsequences.mean(coll_0), subsequences.std(coll_0),
                                                subsequences.mean(coll_1), subsequences.std(coll_1), motif_dimensionality)
            top.put((-curr_dist, [dist_comp, maximum_pair, [dim], stop_dist]))

        if dist_comp > k :
            top.queue = top.queue[:k]

        
       # if i == 0 and j == 1:
        #    pr.disable()
         #   pr.print_stats(sort='cumtime')

        return top.queue, dist_comp, i, j, counter

def order_hash(hash_mat, l, dimension):
    for curr_dim in range(dimension):
        # Order hash[:,rep,dim,:] using as key the array of the last dimension
        hash_mat_curr = hash_mat[:,curr_dim,:]
        ordering = np.lexsort(hash_mat_curr.T[::-1])
    return ordering, l


def pmotif_findg(time_series, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.8):
    #pr = cProfile.Profile()
    #pr.enable()
  # Data
    dimension = time_series.shape[1]
    n = time_series.shape[0]
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}

    
    failure_thresh = fail_thresh
    dist_comp = 0
  # Hasher
    rp = RandomProjection(window, bin_width, K, L) #[]


    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)
    
    chunks = [(time_series[ranges[0]:ranges[-1]+window], ranges, window, rp) for ranges in np.array_split(np.arange(n - window + 1), num_chunks)]

    shm_hash_mat, hash_mat = create_shared_array((n-window+1, L, dimension, K), dtype=np.int8)
    ordering = np.ndarray((dimension, n - window + 1, L), dtype=np.int32)

    with Pool(processes=int(cpu_count())) as pool:
        results = []
        ord_results = []

        for chunk in chunks:
            result = pool.apply_async(process_chunk, (*chunk, shm_hash_mat.name, hash_mat.shape, L, dimension, K))
            results.append(result)

        for result in results:
            std_temp, mean_temp = result.get()
            std_container.update(std_temp)
            mean_container.update(mean_temp)

        for rep in range(L):
          result = pool.apply_async(order_hash, (hash_mat[:,rep,:,:], rep, dimension))
          ord_results.append(result)
        
        for result in ord_results:
            ordering_temp, rep = result.get()
            ordering[:,:,rep] = ordering_temp

    windowed_ts = WindowedTS(time_series, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)


    stop_val = False
    stop_count = 0
    stop_elem = None
    counter_tot = dict()
    with ProcessPoolExecutor(max_workers= cpu_count() // 2) as executor:
        futures = [executor.submit(worker, i, j, windowed_ts, hash_mat, ordering, k, stop_val, fail_thresh) for i, j in itertools.product(range(K), range(L))]
        for future in as_completed(futures):
            top_temp, dist_comp_temp, i, j, counter = future.result()
            # Sum the counter values in the total counter
            #for key, value in counter.items():
             #   counter_tot.setdefault(key, 0)
              #  counter_tot[key] += value
            if dist_comp_temp == 0: 
                stop_count += 1
                if stop_count >= 20:
                    stop_val = True
                if (stop_val and len(top.queue) >= k):
                    executor.shutdown(wait=True, cancel_futures=True)   
                    break
                continue
            dist_comp += dist_comp_temp
            for element in top_temp:
                add = True
                #Check is there's already an overlapping sequence, in that case keep the best match
                for stored in top.queue:
                    indices_1_0 = element[1][1][0]
                    indices_1_1 = element[1][1][1]
                    indices_2_0 = stored[1][1][0]
                    indices_2_1 = stored[1][1][1]
                    if ((abs(indices_1_0 - indices_2_0) < window) or 
                        (abs(indices_1_0 - indices_2_1) < window) or 
                        (abs(indices_1_1 - indices_2_0) < window) or 
                        (abs(indices_1_1 - indices_2_1) < window)):
                        if element[0] > stored[0]:
                            top.queue.remove(stored)
                        else:
                            add = False
                            continue
                if add: top.put(element)
                if len(top.queue) > k:
                    top.get()
            # If the top element of the queue is the same for 10 iterations return
            if stop_elem == top.queue[0]:
                stop_count += 1
            else:
                stop_count = 0
                stop_elem = top.queue[0]
            if stop_count >= 20:
                stop_val = True
            if (stop_val and len(top.queue) >= k):
                    executor.shutdown(wait=True, cancel_futures=True)   
                    break
                    

    if False:
        # Imagine counter_tot as an edge list with weights, plot the graph with the weights as the edge weights of the top
        # elements
        G = nx.Graph()

        # Add the edges with the weights of the top elements and their neighbors
        for element in top.queue:
            G.add_edge(element[1][1][0], element[1][1][1], weight=counter_tot[element[1][1]])
            keys = list(counter_tot.keys())
            for key in keys:
                if element[1][1][0] in key or element[1][1][1] in key:
                    G.add_edge(key[0], key[1], weight=counter_tot[key])

        # Add the edges between the added nodes, if they have a weight
        keys = list(counter_tot.keys())
        for key in keys:
                if key[0] in G.nodes and key[1] in G.nodes:
                    G.add_edge(key[0], key[1], weight=counter_tot[key])

        # Generate layout positions
        pos = nx.spring_layout(G)

        # Get node degrees for size scaling
        node_degrees = dict(G.degree())

        # Each pair of motifs has a color the others are default
        colors = ['tomato', 'blue', 'yellow', 'orange', 'purple']  
        pair_colors = {}
        default_color = 'orchid'

        # Assign colors to pairs in top.queue
        for idx, element in enumerate(top.queue):
            node1, node2 = element[1][1]
            pair_colors[(node1, node2)] = colors[idx % len(colors)]

        # Create the Plotly figure
        fig = go.Figure()

        # Add nodes with size proportional to degree
        for node, (x, y) in pos.items():
            # Check if node belongs to any pair in top.queue
            node_color = default_color  # Default color for non-pair nodes
            for (n1, n2), color in pair_colors.items():
                if node == n1 or node == n2:
                    node_color = color
                    break
            
            # Add node to figure with specified color and size
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=5 + 0.3* node_degrees[node], color=node_color, opacity=0.7),
                text=str(node_degrees[node]),
                name=str(node)
            ))

        # Show the interactive plot
        fig.write_html("graph.html")
    
    #pr.disable()
    #pr.print_stats(sort='cumtime')
    shm_hash_mat.close()
    shm_hash_mat.unlink()
    return top, dist_comp
