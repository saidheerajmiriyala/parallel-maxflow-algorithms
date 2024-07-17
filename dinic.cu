#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <cuda.h>

typedef struct {
    bool path_exists;
    int* path;
    int length;
} BFSResult;

struct Graph {
    int num_nodes;
    int num_edges;
    int* neighbors;
    int* indices;
    int* weights;
};

struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph;
    cudaMallocManaged(&graph, sizeof(struct Graph));
    
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    
    cudaMallocManaged(&(graph->neighbors), num_edges * 2 * sizeof(int));
    cudaMallocManaged(&(graph->indices), (num_nodes + 1) * sizeof(int));
    cudaMallocManaged(&(graph->weights), num_edges * 2 * sizeof(int));
    
    return graph;
}

void freeGraph(struct Graph* graph) {
    cudaFree(graph->neighbors);
    cudaFree(graph->indices);
    cudaFree(graph->weights);
    cudaFree(graph);
}



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














__host__ __device__ int getEdgeIndex(int* indices, int* neighbors, int src, int dest) {
    for (int i = indices[src]; i < indices[src + 1]; i++) {
        if (neighbors[i] == dest) {
            return i;
        }
    }
    return -1;
}

__global__ void ddefault(int* lvl, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_nodes; i += stride) {
        lvl[i] = -1;
    }
}

__global__ void bfs_kernel(int frontier_size, int* indices, int* neighbors, int* weights, bool* found, int* level, int* frontier, int current_level, int dest, int* next_frontier, int* next_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < frontier_size; i += stride) {
        int u = frontier[i];
        for (int j = indices[u]; j < indices[u + 1]; j++) {
            int v = neighbors[j];
            if (level[v] == -1 && weights[j] > 0) {
                int old = atomicCAS(&level[v], -1, current_level + 1);
                if (old == -1) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = v;
                    if (v == dest) {
                        *found = true;
                        // printf("hello\n");
                    }
                }
            }
        }
    }
}

void bfs(int* indices, int* neighbors, int* weights, int* level, int src, int dest, BFSResult* result, int num_nodes, int num_edges, bool* found) {
    ddefault<<<(num_nodes + 255) / 256, 256>>>(level, num_nodes);
    cudaDeviceSynchronize();

    level[src] = 0;

    int* frontier;
    cudaMallocManaged(&frontier, num_nodes * sizeof(int));

    int* next_frontier;
    cudaMallocManaged(&next_frontier, num_nodes * sizeof(int));

    // bool* frontier_visited;
    // cudaMallocManaged(&frontier_visited, num_nodes * sizeof(bool));
    // memset(frontier_visited, 0, num_nodes * sizeof(bool));

    int frontier_size = 0;
    int* next_frontier_size;
    cudaMallocManaged(&next_frontier_size,sizeof(int));

    frontier[frontier_size++] = src;
    int current_level = 0;

    *found = false; // Initialize found to false

    while (frontier_size > 0 && !(*found)) {
        *next_frontier_size = 0;

        bfs_kernel<<<256, 256>>>(frontier_size, indices, neighbors, weights,  found, level, frontier, current_level, dest, next_frontier, next_frontier_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // for (int i = 0; i < num_nodes; i++) {
        //     if (frontier_visited[i]) {
        //         next_frontier[next_frontier_size++] = i;
        //     }
        // }
        // printf("current frontier size %d\n", frontier_size);
        // memset(frontier_visited, 0, num_nodes * sizeof(bool));

        cudaMemcpy(frontier, next_frontier, *next_frontier_size * sizeof(int),cudaMemcpyDeviceToDevice);
        frontier_size = *next_frontier_size;
        current_level++;
    }
    // printf("current frontier size %d\n", frontier_size);

    cudaFree(frontier);
    cudaFree(next_frontier);
   
}

void dfs(int* indices, int* neighbors, int* weights, int src, int dest, BFSResult* result, int* level, int num_nodes, int num_edges) {
    // printf("dfs entered\n");

    int* lvl;
    int* parent;
    cudaMallocManaged(&lvl, num_nodes * sizeof(int));
    cudaMallocManaged(&parent, num_nodes * sizeof(int));
    for (int i = 0; i < num_nodes; i++) {
        lvl[i] = -1;
        parent[i] = -1;
    }
    lvl[src] = 0;
    int* stack;
    cudaMallocManaged(&stack, num_nodes * sizeof(int));
    
    int stack_size = 0;

    stack[stack_size++] = src;
    bool found = false;

    while (stack_size > 0 && !found) {
        int u = stack[--stack_size];

        for (int j = indices[u]; j < indices[u + 1]; j++) {
            int v = neighbors[j];
            int weight = weights[j];

            if (level[v] == level[u] + 1 && weight != 0 && lvl[v] == -1) {
                stack[stack_size++] = v;
                lvl[v] = lvl[u] + 1;
                parent[v] = u;
                if (v == dest) {
                    found = true;
                    break;
                }
            }
        }
    }

    if (found) {
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
    // printf("dfs done\n");
    cudaFree(lvl);
    cudaFree(parent);
    cudaFree(stack);
}

int dinic(int* indices, int* neighbors, int* weights, int src, int dest, int num_nodes, int num_edges) {
    if (src == dest) {
        return -1;
    }

    bool* found;
    cudaMallocManaged(&found, sizeof(bool));

    int max_flow = 0;
    int* level;
    cudaMallocManaged(&level, num_nodes * sizeof(int));

    while (true) {
        BFSResult result;
        bfs(indices, neighbors, weights, level, src, dest, &result, num_nodes, num_edges, found);
        if (!(*found)) {
            break;
        }

        int min_capacity;
        while (true) {
            BFSResult Dfs_result;
            // printf("hellooo\n");
            dfs(indices, neighbors, weights, src, dest, &Dfs_result, level, num_nodes, num_edges);
            min_capacity = INT_MAX;

            if (!Dfs_result.path_exists) {
                break;
            }

            printf("DFS Result: Path exists = %d, Length = %d, Path = ", Dfs_result.path_exists, Dfs_result.length);
            for (int i = 0; i < Dfs_result.length; ++i) {
                printf("%d ", Dfs_result.path[i]);
            }
            printf("\n");

            for (int i = 0; i < Dfs_result.length - 1; ++i) {
                int u = Dfs_result.path[i];
                int v = Dfs_result.path[i + 1];

                int edge_index = getEdgeIndex(indices, neighbors, u, v);

                if (edge_index == -1) {
                    printf("Error: Edge not found in the residual graph\n");
                    return -1;
                }
                min_capacity = (min_capacity < weights[edge_index]) ? min_capacity : weights[edge_index];
            }

            printf("min_capacity: %d\n", min_capacity);

            for (int i = 0; i < Dfs_result.length - 1; ++i) {
                int u = Dfs_result.path[i];
                int v = Dfs_result.path[i + 1];

                int forward_edge_index = getEdgeIndex(indices, neighbors, u, v);

                if (forward_edge_index == -1) {
                    printf("Error: Forward edge not found in the graph\n");
                    return -1;
                }

                weights[forward_edge_index] -= min_capacity;

                int backward_edge_index = getEdgeIndex(indices, neighbors, v, u);
                weights[backward_edge_index] += min_capacity;
            }

            max_flow += min_capacity;
        }
    }

    cudaFree(level);
    cudaFree(found);
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
    cudaMallocManaged(&neighbors, num_edges * 2 * sizeof(int));
    cudaMallocManaged(&weights, num_edges * 2 * sizeof(int));
    cudaMemcpy(indices, graph->indices, (num_nodes + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(neighbors, graph->neighbors, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(weights, graph->weights, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int max_flow = dinic(indices, neighbors, weights, 0, num_nodes - 1, num_nodes, num_edges);
    printf("\nMax flow: %d\n", max_flow);
                clock_gettime(CLOCK_MONOTONIC, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;

    if (nanoseconds < 0) {
        seconds -= 1;
        nanoseconds += 1000000000;
    }

    double elapsed = seconds + nanoseconds * 1e-9;

    printf("Elapsed time: %.9f seconds.\n", elapsed);

    cudaFree(indices);
    cudaFree(neighbors);
    cudaFree(weights);
    freeGraph(graph);

    return 0;
}