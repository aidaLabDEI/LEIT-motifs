from base import *
import numpy as np, queue, itertools
import multiprocessing
from multiprocessing import Pool, cpu_count
from concurrent.futures import as_completed, ProcessPoolExecutor
from hash_lsh import RandomProjection, euclidean_hash
import cProfile
from stop import stopgraph
import networkx as nx, matplotlib.pyplot as plt, plotly.graph_objects as go
from scipy.sparse import dok_matrix

def worker(i, j, subsequences, hash_mat_name, ordering_name, k, stop_i, failure_thresh):  
        #if i == 0 and j == 1:
         #   pr = cProfile.Profile()
         #   pr.enable()
        if stop_i:
            return
        #top_temp, dist_comp_temp = cycle(i, j, windowed_ts, hash_mat, ordering, k, failure_thresh)
        dist_comp = 0
        top = queue.PriorityQueue(k+1)
        window = subsequences.w
        n = subsequences.num_sub
        dimensionality = subsequences.dimensionality
        motif_dimensionality = subsequences.d
        L = subsequences.L
        K = subsequences.K 
        bin_width = subsequences.r
        ex_time_series = shared_memory.SharedMemory(name=subsequences.subsequences)
        time_series = np.ndarray((n,dimensionality), dtype=np.float32, buffer=ex_time_series.buf)
        dimensions = np.arange(dimensionality, dtype=np.int32)
        counter = dok_matrix((n-window+1, n-window+1), dtype=np.uint8)
        existing_arr = shared_memory.SharedMemory(name=hash_mat_name.name)
        existing_ord = shared_memory.SharedMemory(name=ordering_name.name)
        hash_mat = np.ndarray((n-window+1, dimensionality,K), dtype=np.int8, buffer=existing_arr.buf)
        ordering = np.ndarray((dimensionality, n-window+1), dtype=np.int32, buffer=existing_ord.buf)

        #hash_mat_curr = hash_mat[:,:,:-i] if i != 0 else hash_mat
        # Let's assume that ordering has the lexigraphical order of the dimensions
        for curr_dim in range(dimensionality):
            ordering_dim = ordering[curr_dim,:]
            hash_mat_curr = hash_mat[:,curr_dim,:-i] if i != 0 else hash_mat[:,curr_dim,:]
            # Take the subsequent elements of the ordering and check if their hash is the same
            for idx, elem1 in enumerate(hash_mat_curr):
                for idx2, elem2 in enumerate(hash_mat_curr[idx+1:]):
                    sub_idx1 = ordering_dim[idx]
                    sub_idx2 = ordering_dim[idx+idx2+1]
                    # No trivial match
                    if (abs(sub_idx1 - sub_idx2) <= window):
                        continue
                    # If same hash, increase the counter, see the next
                    if (elem1 == elem2).all():
                        counter[sub_idx1, sub_idx2] += 1
                    # Else skip because we know that the ordering ensures that all the subsequences are different
                    else:
                        break
    # Get all entries whose counter is above or equal the motif dimensionality
        #counter_extr = [pair for pair, v in counter.items() if v >= v_max]
        counter_extr = [pair for pair, v in counter.items() if v >= motif_dimensionality]
        del counter
    # Find the set of dimensions with the minimal distance
        for maximum_pair in counter_extr:
            coll_0, coll_1 = maximum_pair
            # Iƒ we already seen it in a key of greater length, skip
            if i >= 1:
                rows = hash_mat[coll_0,:,:] == hash_mat[coll_1,:,:] if i == 1 else hash_mat[coll_0,:,:-i+1] == hash_mat[coll_1,:,:-i+1]
                comp = np.sum(np.all(rows, axis=1))
                if comp >= motif_dimensionality:
                    continue             
            dist_comp += 1
            curr_dist, dim, stop_dist= z_normalized_euclidean_distanceg(time_series[coll_0:coll_0+window].T, time_series[coll_1:coll_1+window].T,
                                                dimensions, subsequences.mean(coll_0), subsequences.std(coll_0),
                                                subsequences.mean(coll_1), subsequences.std(coll_1), motif_dimensionality)
            top.put((-curr_dist, [dist_comp, maximum_pair, [dim], stop_dist]))
            if top.qsize() > k:
                top.get()

        ex_time_series.close()
        existing_arr.close()
        existing_ord.close()
        del time_series, hash_mat, ordering
       # if i == 0 and j == 1:
        #    pr.disable()
         #   pr.print_stats(sort='cumtime')
        return top.queue, dist_comp, i, j#, counter

def order_hash(hash_mat_name, indices_name, l, dimension, num_s, K):
    for hash_name, indices_n in zip(hash_mat_name, indices_name):
        hash_mat_data = shared_memory.SharedMemory(name=hash_name.name)
        hash_mat = np.ndarray((num_s, dimension, K), dtype=np.int8, buffer=hash_mat_data.buf)
        indices_data = shared_memory.SharedMemory(name=indices_n.name)
        indices = np.ndarray((dimension, num_s), dtype=np.int32, buffer=indices_data.buf)
        for curr_dim in range(dimension):
            indices[curr_dim,:] = np.lexsort(hash_mat[:,curr_dim,:].T[::-1])
            hash_mat[:,curr_dim,:] = hash_mat[indices[curr_dim,:], curr_dim,:]

        # Assign the ordering to the shared memory in one go
        #hash_mat = hash_mat[indices,:]
        
        hash_mat_data.close()
        indices_data.close()
    return l

def pmotif_findg(time_series_name, n, dimension, window, k, motif_dimensionality, bin_width, lsh_threshold, L, K, fail_thresh=0.1):
    #pr = cProfile.Profile()
    #pr.enable()
    time_series_data = shared_memory.SharedMemory(name=time_series_name)
    time_series = np.ndarray((n,dimension), dtype=np.float32, buffer=time_series_data.buf)
  # Data
    top = queue.PriorityQueue(maxsize=k+1)
    std_container = {}
    mean_container = {}
    indices_container = []
    hash_container = []

    # Create shared memory for everything
    for _ in range(L):
        arrn, _ = create_shared_array((n-window+1, dimension, K), dtype=np.int8)
        hash_container.append(arrn)
        arri, _ = create_shared_array((dimension, n-window+1), dtype=np.int32)
        indices_container.append(arri)
    
    failure_thresh = fail_thresh
    dist_comp = 0
  # Hasher
    rp = RandomProjection(window, bin_width, K, L) #[]


    chunk_sz = int(np.sqrt(n))
    num_chunks = max(1, n // chunk_sz)
    
    chunks = [(time_series[ranges[0]:ranges[-1]+window], ranges, window, rp) for ranges in np.array_split(np.arange(n - window + 1), num_chunks)]
    #ordering = np.ndarray((dimension, n - window + 1, L), dtype=np.int32)

    # Hash the subsequences and order them lexigraphically
    with Pool() as pool:
        results = []
        ord_results = []

        for chunk in chunks:
            result = pool.apply_async(process_chunk_graph, (*chunk, hash_container, L, dimension, n, K))
            results.append(result)

        for result in results:
            std_temp, mean_temp = result.get()
            std_container.update(std_temp)
            mean_container.update(mean_temp)

        sizeL = int(np.sqrt(L))
        splitted_hash = np.array_split(hash_container, sizeL)
        splitted_indices = np.array_split(indices_container, sizeL)
        for split, indices in zip(splitted_hash, splitted_indices):
            result = pool.apply_async(order_hash, (split, indices, sizeL, dimension, n - window + 1, K))
            ord_results.append(result)
        
        for result in ord_results:
            rep = result.get()

    windowed_ts = WindowedTS(time_series_name, n, dimension, window, mean_container, std_container, L, K, motif_dimensionality, bin_width)
    stop_val = False
    #counter_tot = dict()

    # Cycle for the hash repetitions and concatenations
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(worker, i, j, windowed_ts, hash_container[j], indices_container[j], k, stop_val, fail_thresh) for i, j in itertools.product(range(K), range(L))]
        for future in as_completed(futures):
            #top_temp, dist_comp_temp, i, j, counter = future.result()
            top_temp, dist_comp_temp, i, j = future.result()

            #counter_tot.update(counter)
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
            # Confirmation sampling version
            '''
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
            '''   
            if not top.empty():
                stop_val = stopgraph(top.queue[0], i, j, fail_thresh, K, L, bin_width, motif_dimensionality)
                #print(stop_val)
                if (stop_val and len(top.queue) >= k):
                        executor.shutdown(wait=True, cancel_futures=True)   
                        break
                 
    # Graph construction for visualization
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
    
   # pr.disable()
    #pr.print_stats(sort='cumtime')
    for arr in hash_container:
        arr.close()
        arr.unlink()
    for arr in indices_container:
        arr.close()
        arr.unlink()
    time_series_data.close()
    return top, dist_comp
