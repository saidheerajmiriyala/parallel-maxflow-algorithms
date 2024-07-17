#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

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


// Function to create a graph using CUDA managed memory
struct Graph* createGraph(int num_nodes, int num_edges) {
    struct Graph* graph;
    cudaMallocManaged(&graph, sizeof(struct Graph));
    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;

    cudaMallocManaged(&graph->neighbors, num_edges * 2 * sizeof(int)); // Double for undirected graph
    cudaMallocManaged(&graph->indices, (num_nodes + 1) * sizeof(int));
    cudaMallocManaged(&graph->weights, num_edges * 2 * sizeof(int)); // Double for undirected graph
    cudaMallocManaged(&graph->heights, num_nodes * sizeof(int));
    cudaMallocManaged(&graph->excessflows, num_nodes * sizeof(int));

    // Initialize heights and excessflows to 0
    for (int i = 0; i < num_nodes; i++) {
        graph->heights[i] = 0;
        graph->excessflows[i] = 0;
    }

    return graph;
}


// Function to free the graph using CUDA managed memory
void freeGraph(struct Graph* graph) {
    cudaFree(graph->neighbors);
    cudaFree(graph->indices);
    cudaFree(graph->weights);
    cudaFree(graph->heights);
    cudaFree(graph->excessflows);
    cudaFree(graph);
}


// Function to load a graph from a file
struct Graph* loadGraph(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return NULL;
    }

    // Modify these to match your graph file
    //int num_edges = 4519330;
    //int num_nodes = 1191805;
    // int num_edges = 2987624;
    // int num_nodes = 1134890;
//1134890 2987624
//1191805 4519330    15126 824617  
int num_edges = 2760388;
    int num_nodes = 1957027;
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

        int weight = rand() % 10 + 1;
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
__host__ __device__ int getEdgeIndex(int* indices,int* neighbors,int src, int dest) {
    for (int i = indices[src]; i < indices[src + 1]; i++) {
        if (neighbors[i] == dest) {
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
            int k = getEdgeIndex(graph->indices,graph->neighbors, neighbor, src);

            if (k != -1) {
                graph->weights[k] += capacity;
            }

            graph->excessflows[neighbor] += capacity;
            graph->excessflows[src] -= capacity;
        }
    }
}


__device__ void relabel(int* indices, int* heights,  int* weights, int* neighbors, int u) {

    int minHeight = INT_MAX;

    for (int i = indices[u]; i < indices[u + 1]; i++) {
        int v = neighbors[i];

        if (weights[i] > 0) {
            if (heights[v] < minHeight) {
                minHeight = heights[v];
            }
      
        }
    }

    if (minHeight < INT_MAX) {
        atomicExch(&heights[u], minHeight + 1);
    }
}



__device__ int push(int* indices, int* heights, int* excessflows, int* weights, int* neighbors, int num_edges, int num_nodes, int u) {

    for (int i = indices[u]; i < indices[u + 1]; i++) {
        int v = neighbors[i];

        if (weights[i] > 0 && heights[u] == heights[v] + 1) {
            int delta = min(excessflows[u], weights[i]);

            atomicSub(&weights[i], delta);

            int k = getEdgeIndex(indices,neighbors, v, u); ////
            if (k != -1) {
                atomicAdd(&weights[k], delta);
            }

            atomicSub(&excessflows[u], delta);
            atomicAdd(&excessflows[v], delta);

            return 0;
        }
    }

    return -1;
}




__global__ void push_relabel_kernel(int* indices, int* heights, int* excessflows, int* weights, int* neighbors, int num_edges, int num_nodes,int src , int dest , int* activenodesarray, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // printf("thread id = %d\n",tid);
    for (int i = tid; i < size; i += stride) {
        int u = activenodesarray[i];
        int result = push(indices, heights,  excessflows,  weights, neighbors,  num_edges, num_nodes, u);
        if (result == 0) {
            
        //    printf("push from thread %d\n",tid);
    
        }

        if (result == -1) {
            // printf("relabel from thread %d\n",tid);

            relabel(indices, heights,  weights, neighbors,  u);

            
        }



    }
    return;
}



int get_active_nodes(int* heights,int* excessflows,int num_nodes, int src, int dest, int *activenodesarray){
    int c =0;
    for (int i = 0; i < num_nodes; i++) {       
        if(i !=src && i!=dest && excessflows[i] > 0 && heights[i] < num_nodes){
            activenodesarray[c] = i;
            c++;
        }
        
    }
    return c;
}











































__global__ void ddefault(int* lvl, int* parent,int* heightts, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_nodes; i += stride) {
        lvl[i] = -1;
        parent[i] = -1;
        heightts[i] = num_nodes;
    }
}



__global__ void bfs_kernel(int* heights, int* excessflows, int num_nodes,int frontier_size, int* indices, int* neighbors, int* weights, bool* frontier_visited, int* parent, int* lvl, int* frontier, int current_level, int* heightts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < frontier_size; i += stride) {
        int u = frontier[i];
        for (int j = indices[u]; j < indices[u + 1]; j++) {
            int v = neighbors[j];
            int weight = weights[getEdgeIndex(indices, neighbors , v, u)];
            if (lvl[v] == -1 && weight > 0 && heightts[v] == num_nodes) {
                int old = atomicCAS(&lvl[v], -1, current_level + 1);
                if (old == -1) {
                    frontier_visited[v] = true;
                    parent[v] = u;
                    atomicCAS(&heightts[v], num_nodes, heightts[u] + 1);

                }
            }
        }
    }
}





void bfs(int* indices, int* neighbors,int* heights, int* excessflows, int* weights, int dest, int src, int num_nodes, int num_edges) {
    //clock_t start_time = clock();
    int* lvl;
    cudaMallocManaged(&lvl, num_nodes * sizeof(int));

    int* parent;
    cudaMallocManaged(&parent, num_nodes * sizeof(int));
    // int* heights = (int*)malloc(graph->num_nodes * sizeof(int));
    int* heightts;
    cudaMallocManaged(&heightts, num_nodes * sizeof(int));
    ddefault<<<(num_nodes + 255) / 256, 256>>>(lvl, parent,heightts, num_nodes);
    cudaDeviceSynchronize();

    lvl[dest] = 0;
    heightts[dest] = 0;
    // printf("heightts at dest %d\n\n\n\n", heightts[dest]);

    int* frontier;
    cudaMallocManaged(&frontier, num_nodes * sizeof(int));

    int* next_frontier;
    cudaMallocManaged(&next_frontier, num_nodes * sizeof(int));

    bool* frontier_visited;
    cudaMallocManaged(&frontier_visited, num_nodes * sizeof(bool));
    memset(frontier_visited, 0, num_nodes * sizeof(bool)); // Initialize frontier_visited

    int frontier_size = 0;
    int next_frontier_size = 0;

    frontier[frontier_size++] = dest;
    int current_level = 0;
    

    while (true) {
        next_frontier_size = 0;
       // printf("number of nodes current frontier = %d\n", frontier_size);
        bfs_kernel<<<32, 128>>>(heights, excessflows, num_nodes,frontier_size, indices, neighbors, weights, frontier_visited, parent, lvl, frontier, current_level, heightts);

        cudaDeviceSynchronize();
        if(frontier_size == 0){
           // printf("number of nodes current frontier = %d\n", frontier_size);
            break;
        }

        next_frontier_size = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (frontier_visited[i]) {
                next_frontier[next_frontier_size++] = i;
            }
        }

        memset(frontier_visited, 0, num_nodes * sizeof(bool));

        memcpy(frontier, next_frontier, next_frontier_size * sizeof(int));
        frontier_size = next_frontier_size;
        current_level++;
    }


   // printf("frontier size = %d\n",frontier_size);
    for(int i = 0; i < num_nodes; i++){
                   
                    if(i !=src){heights[i]=heightts[i];}
    }






    cudaFree(lvl);
    cudaFree(parent);
    cudaFree(frontier);
    cudaFree(next_frontier);
    cudaFree(frontier_visited);
    cudaFree(heightts);


}





























int main(){
    const char* filename = "mm.txt";
    struct Graph* graph = loadGraph(filename);
    if (!graph) {
        return -1;
    }

    int* indices;
    int* neighbors;
    int* weights;
    int* heights;
    int* excessflows;
    int num_edges = 2760388;
    int num_nodes = 1957027;
    cudaMallocManaged(&indices, (num_nodes + 1) * sizeof(int));
    cudaMallocManaged(&neighbors, num_edges * 2 *sizeof(int));
    cudaMallocManaged(&weights, num_edges * 2 * sizeof(int));
    cudaMallocManaged(&heights, num_edges * 2 * sizeof(int));




















    int src = 0;
    int dest = 15000;
  
    preflow(graph, src);
    int count = 0;
  
    int cc=0;
    cudaMallocManaged(&excessflows, num_edges * 2 * sizeof(int));
    cudaMemcpy(indices, graph->indices, (num_nodes + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(neighbors, graph->neighbors, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(weights, graph->weights, num_edges * 2 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(heights, graph->heights, (num_nodes) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(excessflows, graph->excessflows, (num_nodes) * sizeof(int), cudaMemcpyDeviceToDevice);











    int* activenodesarray;
    cudaMallocManaged(&activenodesarray, graph->num_nodes * sizeof(int));
    int c = 0;
    while(true){
        
        c = get_active_nodes(heights, excessflows,num_nodes,src,dest,activenodesarray);
        printf("Iteration %d: Number of active nodes = %d, flow = %d\n", count, c , excessflows[dest]);
        if( c == 0){
            printf("flow = %d\n", excessflows[dest]);
            break;
        }

      push_relabel_kernel<<<1,32>>>(indices, heights, excessflows,  weights,  neighbors,num_edges, num_nodes,src,dest,activenodesarray,c);

      cudaDeviceSynchronize();
        

        count++;
                if(count % 100 == 0){

            bfs(indices,  neighbors, heights, excessflows, weights,  dest, src, num_nodes, num_edges);

            cc++;
        }
        
        



    }
    printf("number of times relabbling occured %d\n",cc);
    
    return 0;



}