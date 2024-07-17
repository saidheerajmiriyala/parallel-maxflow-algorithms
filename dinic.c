#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

#define MAX_SIZE 281903
#define INT_MAX 2147483647


typedef struct {
    bool path_exists;
    int* path;
    int length;
} BFSResult;


struct Graph {
    int num_nodes;
    int num_edges;
    int* neighbors; // Array to store the neighbors
    int* indices;   // Array to store the starting index of each node's neighbors
    int* weights;   // Array to store the weights of the edges
};

// Function to create a graph
struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->neighbors = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    graph->indices = (int*)malloc((num_nodes + 1) * sizeof(int));
    graph->weights = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    return graph;
}

// Function to free the graph
void freeGraph(struct Graph* graph) {
    free(graph->neighbors);
    free(graph->indices);
    free(graph->weights);
    free(graph);
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














// Function to get the edge index
int getEdgeIndex(struct Graph* graph, int src, int dest) {
    for (int i = graph->indices[src]; i < graph->indices[src + 1]; i++) {
        if (graph->neighbors[i] == dest) {
            return i;
        }
    }
    return -1;
}




bool bfsLevelGraph(struct Graph* graph,int* level, int src, int dest) {
    
    for (int i = 0; i < graph->num_nodes; i++) {
        level[i] = -1;
        
    }
    level[src] = 0;

    int* frontier = (int*)malloc(graph->num_nodes * sizeof(int));
    int* next_frontier = (int*)malloc(graph->num_nodes * sizeof(int));
    int frontier_size = 0;
    int next_frontier_size = 0;

    frontier[frontier_size++] = src;
    int current_level = 0;
    bool found = false;

    while (frontier_size > 0 && !found) {
        next_frontier_size = 0;

        for (int i = 0; i < frontier_size; i++) {
            int u = frontier[i];

            for (int j = graph->indices[u]; j < graph->indices[u + 1]; j++) {
                int v = graph->neighbors[j];
                int weight = graph->weights[j];

                if (weight != 0 && level[v] == -1) {
                    next_frontier[next_frontier_size++] = v;
                    level[v] = current_level + 1;
                    if (v == dest) {
                        found = true;

                    }
                }
            }

        }

        int* temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;

        frontier_size = next_frontier_size;
        current_level++;
    }



    free(frontier);
    free(next_frontier);
    return found;
}





// Function to perform DFS
void dfs(struct Graph* graph, int src, int dest, BFSResult* result, int* level) {
    int* lvl = (int*)malloc(graph->num_nodes * sizeof(int));
    int* parent = (int*)malloc(graph->num_nodes * sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        lvl[i] = -1;
        parent[i] = -1;
    }
    lvl[src] = 0;

    int* stack = (int*)malloc(graph->num_nodes * sizeof(int));
    int stack_size = 0;

    stack[stack_size++] = src;
    bool found = false;

    while (stack_size > 0 && !found) {
        int u = stack[--stack_size];

        for (int j = graph->indices[u]; j < graph->indices[u + 1]; j++) {
            int v = graph->neighbors[j];
            int weight = graph->weights[j];

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

    free(lvl);
    free(parent);
    free(stack);
}





// Function to implement Dinic's Algorithm
int dinic(struct Graph* graph, int src, int dest) {
    if (src == dest) {
        return -1;
    }

    int* level = (int*)malloc(graph->num_nodes * sizeof(int));


    int max_flow = 0;

    while (bfsLevelGraph(graph, level, src, dest)) {
        int min_capacity;
        while (true) {
            BFSResult Dfs_result;
            dfs(graph, src, dest, &Dfs_result, level);
            min_capacity = INT_MAX;

            if (!Dfs_result.path_exists) {
                // printf("hello\n");
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

                int edge_index = getEdgeIndex(graph, u, v);

                if (edge_index == -1) {
                    printf("Error: Edge not found in the residual graph\n");
                    return -1;
                }
                min_capacity = (min_capacity < graph->weights[edge_index]) ? min_capacity : graph->weights[edge_index];
            }

            printf("min_capacity: %d\n", min_capacity);

            for (int i = 0; i < Dfs_result.length - 1; ++i) {
                int u = Dfs_result.path[i];
                int v = Dfs_result.path[i + 1];

                int forward_edge_index = getEdgeIndex(graph, u, v);

                if (forward_edge_index == -1) {
                    printf("Error: Forward edge not found in the graph\n");
                    return -1;
                }

                graph->weights[forward_edge_index] -= min_capacity;

                int backward_edge_index = getEdgeIndex(graph, v, u);


                graph->weights[backward_edge_index] += min_capacity;
            }

            max_flow += min_capacity;
        }















        }

    free(level);
   
    return max_flow;
}




























int main() {
    const char* filename = "t9_281903,2312497.txt";
    struct Graph* graph = loadGraph(filename);
    if (graph) {
        struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
        int max_flow = dinic(graph, 0, graph->num_nodes - 1);
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

        freeGraph(graph);

    }
    return 0;
}
