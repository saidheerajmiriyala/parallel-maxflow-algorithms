#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

struct Graph {
    int num_nodes;
    int num_edges;
    int* neighbors; // Array to store the neighbors
    int* indices;   // Array to store the starting index of each node's neighbors
    int* weights;   // Array to store the weights of the edges
};

struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph;
    cudaMallocManaged(&graph, sizeof(struct Graph));
    
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    
    cudaMallocManaged(&(graph->neighbors), num_edges * 2 * sizeof(int));
    cudaMallocManaged(&(graph->indices), (num_nodes + 1) * sizeof(int)); // +1 to store the end index
    cudaMallocManaged(&(graph->weights), num_edges * 2 * sizeof(int));
    
    return graph;
}

void freeGraph(struct Graph* graph) {
    cudaFree(graph->neighbors);
    cudaFree(graph->indices);
    cudaFree(graph->weights);
    cudaFree(graph);
}

typedef struct {
    bool path_exists;
    int* path;
    int length;
} BFSResult;





// Function to load a graph from a file
struct Graph* loadGraph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return NULL;
    }

    int num_edges = 2312497;   //281903 2312497
    int num_nodes = 281903;

    struct Graph* graph = createGraph(num_nodes, num_edges);

    printf("Loading graph from file...\n");

    int* temp_counts = (int*)calloc(num_nodes, sizeof(int));

    // First pass: count the number of neighbors for each node
    for (int i = 0; i < num_edges; ++i) {
        int src, dest;
        if (fscanf(file, "%d %d\n", &src, &dest) != 2) {
            printf("Error: Invalid file format.\n");
            fclose(file);
            free(temp_counts);
            freeGraph(graph);
            return NULL;
        }
        temp_counts[src]++;
        temp_counts[dest]++; // Count for both directions
    }

    // Build the index array
    graph->indices[0] = 0;
    for (int i = 1; i <= num_nodes; ++i) {
        graph->indices[i] = graph->indices[i - 1] + temp_counts[i - 1];
    }

    // Reset file pointer to the beginning of the file for the second pass
    fseek(file, 0, SEEK_SET);

    int* current_pos = (int*)calloc(num_nodes, sizeof(int));

    // Second pass: fill the neighbors array and weights array
    for (int i = 0; i < num_edges; ++i) {
        int src, dest;
        if (fscanf(file, "%d %d\n", &src, &dest) != 2) {
            printf("Error: Invalid file format.\n");
            fclose(file);
            free(temp_counts);
            free(current_pos);
            freeGraph(graph);
            return NULL;
        }

        int index_src = graph->indices[src] + current_pos[src];
        int index_dest = graph->indices[dest] + current_pos[dest];

        graph->neighbors[index_src] = dest;
        graph->neighbors[index_dest] = src;

        int weight = 1;
        graph->weights[index_src] = weight;
        graph->weights[index_dest] = weight; // Same weight for undirected graph

        current_pos[src]++;
        current_pos[dest]++;
    }

    printf("Graph loaded successfully.\n");

    free(temp_counts);
    free(current_pos);
    fclose(file);
    return graph;
}




















__global__ void ddefault(int* lvl, int* parent, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_nodes; i += stride) {
        lvl[i] = -1;
        parent[i] = -1;
    }
}

__global__ void bfs_kernel(int frontier_size, int* indices, int* neighbors, int* weights,  int* parent, bool* found, int* lvl, int* frontier, int current_level, int dest,int* next_frontier, int* next_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (*found) {
        return;
    }

    for (int i = tid; i < frontier_size; i += stride) {
        int u = frontier[i];
        for (int j = indices[u]; j < indices[u + 1]; j++) {
            int v = neighbors[j];
            if (lvl[v] == -1 && weights[j] > 0) {
                int old = atomicCAS(&lvl[v], -1, current_level + 1);
                if (old == -1) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = v;
                    parent[v] = u;
                    if (v == dest) {
                        *found = true;
                        return;
                    }
                }
            }
        }
    }
}

void bfs(int* indices, int* neighbors, int* weights, int src, int dest, BFSResult* result, int num_nodes, int num_edges) {
    //clock_t start_time = clock();
    int* lvl;
    cudaMallocManaged(&lvl, num_nodes * sizeof(int));

    int* parent;
    cudaMallocManaged(&parent, num_nodes * sizeof(int));
    ddefault<<<(num_nodes + 255) / 256, 256>>>(lvl, parent, num_nodes);
    cudaDeviceSynchronize();

    lvl[src] = 0;

    int* frontier;
    cudaMallocManaged(&frontier, num_nodes * sizeof(int));

    int* next_frontier;
    cudaMallocManaged(&next_frontier, num_nodes * sizeof(int));



    int frontier_size = 0;
        int* next_frontier_size;
    cudaMallocManaged(&next_frontier_size,sizeof(int));

    frontier[frontier_size++] = src;
    int current_level = 0;
    bool* found;
    cudaMallocManaged(&found, sizeof(bool));
    *found = false;

    while (frontier_size > 0 && !(*found)) {
        *next_frontier_size = 0;

        bfs_kernel<<<256, 256>>>(frontier_size, indices, neighbors, weights,  parent, found, lvl, frontier, current_level, dest,next_frontier, next_frontier_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        if (*found) {
            break;
        }




        cudaMemcpy(frontier, next_frontier, *next_frontier_size * sizeof(int),cudaMemcpyDeviceToDevice);
        frontier_size = *next_frontier_size;
        current_level++;
    }

    if (*found) {
        result->path_exists = true;
        result->length = 0;

        int temp = dest;
        while (temp != -1) {
            result->length++;
            temp = parent[temp];
        }

        result->path = (int*)malloc(result->length * sizeof(int));
        temp = dest;
        for (int i = result->length - 1; i >= 0; i--) {
            result->path[i] = temp;
            temp = parent[temp];
        }
    } else {
        result->path_exists = false;
        result->path = NULL;
        result->length = 0;
    }

    cudaFree(lvl);
    cudaFree(parent);
    cudaFree(frontier);
    cudaFree(next_frontier);
   
    cudaFree(found);

   // clock_t end_time = clock();
   // double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  //  printf("BFS completed in %.2f seconds.\n", time_spent);
}

void printPath(BFSResult result) {
    if (result.path_exists) {
        printf("Path from source to destination:\n");
        for (int i = 0; i < result.length; i++) {
            printf("%d ", result.path[i]);
        }
        printf("\n");
    } else {
        printf("No path exists from the source to the destination.\n");
    }
}



int getEdgeIndex(int* indices,int* neighbors, int src, int dest) {
    for (int i = indices[src]; i < indices[src + 1]; i++) {
        if (neighbors[i] == dest) {
            return i;
        }
    }
    return -1;
}








































int edmondsKarp(int* indices,int* neighbors, int* weights, int src, int dest,int num_nodes,int num_edges) {
    int min_capacity;
    int max_flow = 0;

    // Continue looping until no path from source to sink exists in the residual graph
    while (true) {
        // Find an augmenting path using BFS
        BFSResult bfs_result;
        bfs(indices,neighbors,weights, src, dest, &bfs_result,num_nodes,num_edges);
        min_capacity = INT_MAX;

        // If no path exists, break the loop
        if (!bfs_result.path_exists) {

            break;
        }

        // Print the BFS result 
        printf("BFS Result: Path exists = %d, Length = %d, Path = ", bfs_result.path_exists, bfs_result.length);
        for (int i = 0; i < bfs_result.length; ++i) {
            printf("%d ", bfs_result.path[i]);
        }
        printf("\n");

        for (int i = 0; i < bfs_result.length - 1; ++i) {
            int u = bfs_result.path[i];
            int v = bfs_result.path[i + 1];

            // Find the corresponding edge in the residual graph
            int edge_index = getEdgeIndex(indices,neighbors, u, v);
            
            
            if (edge_index == -1) {
                printf("Error: Edge not found in the residual graph\n");
                return -1;
            }
            min_capacity = (min_capacity < weights[edge_index]) ? min_capacity : weights[edge_index];
        }

        printf("min_capacity: %d\n", min_capacity);

        // Update the flow along the augmenting path and the residual capacities of the edges
        for (int i = 0; i < bfs_result.length - 1; ++i) {
            int u = bfs_result.path[i];
            int v = bfs_result.path[i + 1];

            // Find the index of the forward edge in the graph
            int forward_edge_index = getEdgeIndex(indices,neighbors, u, v);

            if (forward_edge_index == -1) {
                printf("Error: Forward edge not found in the graph\n");
                return -1;
            }

            weights[forward_edge_index] -= min_capacity;

            // Find the index of the corresponding backward edge in the graph
            int backward_edge_index = getEdgeIndex(indices,neighbors, v, u);
            
            // if (backward_edge_index == -1) {
            //      //printf("Error: Backward edge not found \n");
            //     // addReverseEdge(indices,&neighbors,&weights, v, u, 0, num_nodes, num_edges);
            //      //printf("done adding\n");
            //     backward_edge_index = getEdgeIndex(indices,neighbors, v, u);
            // }
             //printf("backward edge index: %d\n", backward_edge_index);

            weights[backward_edge_index] += min_capacity;
        }

        max_flow += min_capacity;
    }

    return max_flow;
}



































int main() {
    const char* filename = "t9_281903,2312497.txt";
    struct Graph* graph = loadGraph(filename);
    if (!graph) {
        return 1;
    }
    int* indices;
    int* neighbors;
    int* weights;
      int num_edges = 2312497;   //281903 2312497
    int num_nodes = 281903;
    cudaMallocManaged(&indices, (num_nodes + 1) * sizeof(int));
    cudaMallocManaged(&neighbors, num_edges * 2 *sizeof(int));
    cudaMallocManaged(&weights, num_edges * 2 * sizeof(int));

    cudaMemcpy(indices, graph->indices, (num_nodes + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(neighbors, graph->neighbors, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(weights, graph->weights, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);

    int src = 0;
    int dest = num_nodes - 1;
struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    
    int k = edmondsKarp(indices,neighbors,weights,src,dest,num_nodes,num_edges);
    
    printf("maxflow = %d\n", k);
 
                clock_gettime(CLOCK_MONOTONIC, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;

    if (nanoseconds < 0) {
        seconds -= 1;
        nanoseconds += 1000000000;
    }

    double elapsed = seconds + nanoseconds * 1e-9;

    printf("Elapsed time: %.9f seconds.\n", elapsed);
    freeGraph(graph);
    cudaFree(indices);
    cudaFree(neighbors);
    cudaFree(weights);
  
    return 0;
}
