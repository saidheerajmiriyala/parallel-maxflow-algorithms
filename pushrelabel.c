#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

// Graph structure
struct Graph {
    int num_nodes;
    int num_edges;
    int* neighbors;
    int* indices;
    int* weights;
    int* heights;
    int* excessflows;
};

// Function to create a graph
struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->neighbors = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    graph->indices = (int*)malloc((num_nodes + 1) * sizeof(int));
    graph->weights = (int*)malloc(num_edges * 2 * sizeof(int)); // Double for undirected graph
    graph->heights = (int*)calloc(num_nodes, sizeof(int));
    graph->excessflows = (int*)calloc(num_nodes, sizeof(int));
    return graph;
}

// Function to free the graph
void freeGraph(struct Graph* graph) {
    free(graph->neighbors);
    free(graph->indices);
    free(graph->weights);
    free(graph->heights);
    free(graph->excessflows);
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

// Function to initialize preflow
void preflow(struct Graph* graph, int src) {
    graph->heights[src] = graph->num_nodes;
    graph->excessflows[src] = 0;
    for (int i = graph->indices[src]; i < graph->indices[src + 1]; i++) {
        graph->excessflows[src] += graph->weights[i];
    }

    for (int i = graph->indices[src]; i < graph->indices[src + 1]; i++) {
        int neighbor = graph->neighbors[i];
        int capacity = graph->weights[i];

        if (capacity > 0) {
            graph->weights[i] = 0;
            int k = getEdgeIndex(graph, neighbor, src);

            if (k != -1) {
                graph->weights[k] += capacity;
            }

            graph->excessflows[neighbor] += capacity;
            graph->excessflows[src] -= capacity;
        }
    }
}

// Function to relabel a node
void relabel(struct Graph* graph, int u) {
    int minHeight = INT_MAX;

    for (int i = graph->indices[u]; i < graph->indices[u + 1]; i++) {
        int v = graph->neighbors[i];

        if (graph->weights[i] > 0) {
            if (graph->heights[v] < minHeight) {
                minHeight = graph->heights[v];
            }
        }
    }

    if (minHeight < INT_MAX) {
        graph->heights[u] = minHeight + 1;
    }
}

// Function to push flow from a node
int push(struct Graph* graph, int u) {
    int f = -1;
    for (int i = graph->indices[u]; i < graph->indices[u + 1]; i++) {
        int v = graph->neighbors[i];
        
        if (graph->weights[i] > 0 && graph->heights[u] == graph->heights[v] + 1) {
            int delta = graph->excessflows[u] < graph->weights[i] ? graph->excessflows[u] : graph->weights[i];

            graph->weights[i] -= delta;
            int k = getEdgeIndex(graph, v, u);
            if (k != -1) {
                graph->weights[k] += delta;
            }

            graph->excessflows[u] -= delta;
            graph->excessflows[v] += delta;
            f = 0;

            return 0;
        }
    }

    return -1;
    // return f;
}

// Function to find an active node
int find_active_node(struct Graph* graph, int src, int dest) {
    for (int i = 0; i < graph->num_nodes; i++) {
        if (i == src || i == dest) {
            continue;
        }

        if (graph->excessflows[i] > 0 && graph->heights[i]< graph->num_nodes) {
            return i;
        }
    }
    return -1;
}

// Function to count active nodes
int count_active_nodes(struct Graph* graph, int src, int dest) {
    int count = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        if (i != src && i != dest && graph->excessflows[i] > 0) {
            count++;
        }
    }
    return count;
}

void bfs(struct Graph* graph, int dest, int src) {
    int* lvl = (int*)malloc(graph->num_nodes * sizeof(int));
    int* parent = (int*)malloc(graph->num_nodes * sizeof(int));
    int* heights = (int*)malloc(graph->num_nodes * sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        lvl[i] = -1;
        parent[i] = -1;
        heights[i] = graph->num_nodes;
        

    }
    lvl[dest] = 0;
    heights[dest] = 0;

    int* frontier = (int*)malloc(graph->num_nodes * sizeof(int));
    int* next_frontier = (int*)malloc(graph->num_nodes * sizeof(int));
    int frontier_size = 0;
    int next_frontier_size = 0;

    frontier[frontier_size++] = dest;
    int current_level = 0;


    while (frontier_size > 0) {

        next_frontier_size = 0;

        for (int i = 0; i < frontier_size; i++) {
            int u = frontier[i];
 
            for (int j = graph->indices[u]; j < graph->indices[u + 1]; j++) {
                int v = graph->neighbors[j];
                int weight = graph->weights[getEdgeIndex(graph, v, u)];

                if (weight > 0 && lvl[v] == -1 && heights[v] == graph->num_nodes) {
                    next_frontier[next_frontier_size++] = v;
                    lvl[v] = current_level + 1;
                    parent[v] = u;
                    heights[v] = heights[u] + 1;

                }
            }
            
        }

        int* temp = frontier;
        frontier = next_frontier;
        next_frontier = temp;

        frontier_size = next_frontier_size;
        current_level++;
    }
    for(int i = 0; i < graph->num_nodes; i++){             
                    if(i !=src){graph->heights[i]=heights[i];}
    }



    
    free(lvl);
    free(parent);
    free(frontier);
    free(heights);
    free(next_frontier);
} 


















int main() {
    const char* filename = "t9_281903,2312497.txt";
    struct Graph* graph = loadGraph(filename);
    if (!graph) {
        return -1;
    }

    int src = 0;
    int dest = graph->num_nodes-1;

    preflow(graph, src);
    int count = 0;
  

    int* height_count = (int*)calloc(graph->num_nodes * 2, sizeof(int));
    for (int i = 0; i < graph->num_nodes; i++) {
        height_count[graph->heights[i]]++;
    }
    int cc=0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (true) {
        int active_nodes = count_active_nodes(graph, src, dest);

        printf("Iteration %d: Number of active nodes =%d , flow = %d\n", count,active_nodes, graph->excessflows[dest]);
        
        int k = find_active_node(graph, src, dest);
        if (k == -1) {
            printf("Max flow from %d to %d is %d\n", src, dest, graph->excessflows[dest]);
            break;
        }

        int u = k;
        int result = push(graph, u);
        if (result == 0) {
            
            count++;
    
        }

        if (result == -1) {
            relabel(graph, u);
            count++;
        }
        if ( count % 100 == 0){
            bfs(graph,dest,src);
            cc++;
        }



    }


        clock_gettime(CLOCK_MONOTONIC, &end);

    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;

    if (nanoseconds < 0) {
        seconds -= 1;
        nanoseconds += 1000000000;
    }

    double elapsed = seconds + nanoseconds * 1e-9;

    printf("Elapsed time: %.9f seconds.\n", elapsed);
    printf("number of times relabbling occured %d\n",cc);
    // free(height_count);
    freeGraph(graph);
    return 0;
}
