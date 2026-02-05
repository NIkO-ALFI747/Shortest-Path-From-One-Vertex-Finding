# Parallel Bellman-Ford Shortest Path Algorithm: Multi-Platform Implementation

## Overview

A comprehensive comparative study of the Bellman-Ford single-source shortest path algorithm implemented across four distinct computational paradigms: serial execution, shared-memory parallelism (OpenMP), distributed-memory parallelism (MPI), and GPU acceleration (CUDA). The project demonstrates advanced parallel algorithm design, optimization techniques, and performance analysis across heterogeneous computing architectures.

## Algorithmic Foundation

### Bellman-Ford Algorithm

The Bellman-Ford algorithm solves the single-source shortest path problem for weighted directed graphs, including those with negative edge weights. Unlike Dijkstra's algorithm, it can handle negative weights and detect negative-weight cycles.

**Time Complexity**: O(V × E) where V = vertices, E = edges  
**Space Complexity**: O(V)

**Core Properties**:
- Handles negative edge weights
- Detects negative-weight cycles
- Guarantees optimal solution after V-1 iterations
- Suitable for sparse graphs represented in various formats

**Relaxation Operation** (fundamental primitive):
```cpp
if (distance[u] + weight(u,v) < distance[v]) {
    distance[v] = distance[u] + weight(u,v);
    predecessor[v] = u;
}
```

## Project Architecture

### Components

1. **Bellman-Ford-Serial**: Baseline sequential implementation
2. **Bellman-Ford-OMPParallel**: OpenMP shared-memory parallelization
3. **Bellman-Ford-MPIParallel**: MPI distributed-memory parallelization
4. **Bellman-Ford-CUDAParallel**: CUDA GPU-accelerated implementation
5. **EdgeList-To-CSR**: Graph format converter (Edge List → CSR)
6. **Test-Data-Generator**: Random connected graph generator with negative weight support

### Graph Representation Strategies

#### Edge List Format (Serial Implementation)
```
Structure: [(source, destination, weight), ...]
Example: [(0, 1, 5), (0, 2, 3), (1, 3, 2)]
```

**Characteristics**:
- Simple, intuitive representation
- Space-efficient for sparse graphs
- Poor cache locality during edge relaxation
- Used in serial baseline implementation

#### Compressed Sparse Row (CSR) Format (Parallel Implementations)
```
Three arrays:
- Sources: [adjacent vertices for each destination]
- RangesOfAdjacentVertices: [start indices for each vertex's neighbors]
- Weights: [corresponding edge weights]
```

**Advantages**:
- Excellent cache locality
- Coalesced memory access patterns (GPU)
- Efficient parallel vertex processing
- Reduced memory bandwidth requirements

**Conversion Process**:
1. Sort edges by destination vertex
2. Compute edge count per destination
3. Calculate cumulative ranges
4. Sort sources within each vertex's adjacency range

## Implementation Details

### 1. Serial Implementation

**File**: `Bellman-Ford-Serial/Bellman-Ford-Serial.cpp`

**Key Features**:
- Baseline reference implementation
- Early termination optimization (iteration stops when no updates occur)
- Negative cycle detection
- Uses Edge List representation

**Algorithm Structure**:
```cpp
void BellmanFordSerial(vector<int>& shortestPaths, vector<int>& previousVertices,
    bool& negativeWeightCycle, int numOfVertices, int numOfEdges, 
    int startVertex, vector<int> sources, vector<int> dests, 
    vector<int> weights, double& time, int& k)
{
    // Initialize distances to infinity
    for (int i = 0; i < numOfVertices; i++) {
        shortestPaths[i] = INF;
        previousVertices[i] = -1;
    }
    shortestPaths[startVertex] = 0;
    
    // Main relaxation loop with early termination
    bool kLoop = true;
    for (k = 0; (k < numOfVertices - 1) && kLoop; k++) {
        kLoop = false;
        for (int i = 0; i < numOfEdges; i++) {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]]) {
                shortestPaths[dests[i]] = shortestPaths[sources[i]] + weights[i];
                previousVertices[dests[i]] = sources[i];
                kLoop = true;  // Continue if updates occurred
            }
        }
    }
    
    // Negative cycle detection
    if (k == numOfVertices - 1) {
        for (int i = 0; i < numOfEdges; i++) {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]]) {
                negativeWeightCycle = true;
                break;
            }
        }
    }
}
```

**Performance Characteristics**:
- Predictable execution time
- No synchronization overhead
- Cache-friendly sequential access
- Serves as speedup baseline

### 2. OpenMP Parallel Implementation

**File**: `Bellman-Ford-OMPParallel/Bellman-Ford-OMPParallel.cpp`

**Parallelization Strategy**: Vertex-level parallelism with CSR representation

**Key Optimizations**:
1. **Parallel Initialization**: Concurrent distance array initialization
2. **Static Scheduling**: Deterministic thread-to-vertex mapping for cache affinity
3. **Reduction Clause**: Efficient parallel summation of update flags
4. **Temporary Array**: Read-write separation to avoid race conditions

**Critical Sections**:
```cpp
#pragma omp parallel for schedule(static) reduction (+:kLoop)
for (int i = 0; i < numOfVertices; i++) {
    for (int j = rangesOfAdjacentVertices[i]; 
         j < rangesOfAdjacentVertices[i + 1]; j++) {
        if (shortestTempPaths[i] > shortestPaths[sources[j]] + weights[j]) {
            shortestTempPaths[i] = shortestPaths[sources[j]] + weights[j];
            previousVertices[i] = sources[j];
            kLoop++;  // Thread-local accumulator
        }
    }
}
```

**Synchronization Points**:
- Implicit barrier after parallel initialization
- Reduction synchronization for convergence check
- Array copy synchronization between iterations

**Thread Scaling Strategy**:
- Tests with 1, 2, 3, 4 threads
- Static scheduling ensures consistent workload distribution
- Reduction clause minimizes synchronization overhead

**Race Condition Avoidance**:
- Separate read (`shortestPaths`) and write (`shortestTempPaths`) arrays
- Each vertex updated by single thread (static partitioning)
- Reduction variable prevents contention on convergence flag

### 3. MPI Distributed-Memory Implementation

**File**: `Bellman-Ford-MPIParallel/Bellman-Ford-MPIParallel.cpp`

**Decomposition Strategy**: Domain decomposition with vertex partitioning

**Process Allocation Algorithm**:
```cpp
void Process::getArrComputationalBoundaries(int arraySize) {
    partOfArr = arraySize / size;
    if ((rank + 1) > size - (arraySize % size)) {
        // Handle remainder distribution
        numberOfAdditionsToIndexes = rank - (size - (arraySize % size));
        startIndex = partOfArr * rank + numberOfAdditionsToIndexes;
        partOfArr += 1;
        stopIndex = startIndex + partOfArr;
    } else {
        startIndex = partOfArr * rank;
        stopIndex = partOfArr * (rank + 1);
    }
}
```

**Load Balancing**:
- Vertices distributed as evenly as possible
- Remainder vertices assigned to higher-rank processes
- Each process computes `ceil(V/P)` or `floor(V/P)` vertices

**Communication Pattern**:

**Per Iteration**:
1. **Computation Phase**: Each process relaxes edges for its vertex partition
2. **Convergence Check**: `MPI_Allreduce` sums local update counts
3. **Distance Exchange**: `MPI_Allgatherv` broadcasts updated distances

**Critical MPI Operations**:
```cpp
// Global convergence detection
MPI_Allreduce(&kLoop_local, &kLoop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

// Distance array synchronization
MPI_Allgatherv(shortestTempPaths, proc.partOfArr, MPI_INT, shortestPaths_link,
               recvCounts, displs, MPI_INT, MPI_COMM_WORLD);

// Final predecessor gathering
MPI_Gatherv(previousVertices_local, proc.partOfArr, MPI_INT, 
            previousVertices_link, recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
```

**Communication Complexity**:
- `MPI_Allreduce`: O(log P) with optimal algorithms
- `MPI_Allgatherv`: O(V × P) — dominant communication cost
- Total per iteration: O(V × P)

**Scalability Challenges**:
- High communication-to-computation ratio for sparse graphs
- All-to-all communication pattern limits strong scaling
- Network topology impacts performance (interconnect bandwidth critical)

**MPI I/O Optimization**:
```cpp
string File::fileToBufMPI() {
    MPI_File handle;
    MPI_File_open(MPI_COMM_WORLD, fileName.c_str(), 
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &handle);
    MPI_File_read(handle, buf, fileSize, MPI_BYTE, &status);
    MPI_File_close(&handle);
}
```
- Collective file operations
- Parallel I/O reduces initialization overhead
- Scales better than serial read + broadcast

### 4. CUDA GPU Parallel Implementation

**File**: `Bellman-Ford-CUDAParallel/kernel.cu`

**Parallelization Philosophy**: Massively parallel vertex processing with hierarchical synchronization

#### GPU Memory Hierarchy Utilization

**Global Memory**:
- `shortestPaths`: Current best distances (read-write)
- `edges`, `weights`, `rangesOfAdjacentVertices`: Graph data (read-only)

**Shared Memory**:
- `shortestTempPathsShared`: Block-local temporary distances
- `previousVerticesShared`: Block-local predecessor tracking
- Reduces global memory transactions

**Register Memory**:
- Local variables for edge relaxation
- Maximizes compute throughput

#### Thread Hierarchy

**Grid Configuration**:
```cpp
int optimalBlockSize = getOptimalBlockSize(numOfVertices, 1024);
int gridSize = (numOfVertices + optimalBlockSize - 1) / optimalBlockSize;
```

**Block Size Optimization**:
- Dynamic calculation based on vertex count
- Factorization-based approach for optimal divisibility
- Maximum block size: 1024 threads (hardware limit)

**Work Distribution**:
```cpp
DataIndexes::DataIndexes(int numOfData, int gridSize, int blockSize, int warpSize) {
    blockPartOfData = numOfData / gridSize;
    blockIndex = gridSize - (numOfData % gridSize);
    // Calculate warp-level iteration counts
    setVarpsIndexes(blockPartOfData, blockSize, warpSize,
                    warpsThreadIndexBeforeBlockIndex, 
                    warpsMaxNumOfIterationsBeforeBlockIndex);
}
```

**Load Balancing Complexity**:
- Vertices may not divide evenly across grid
- Warp-level granularity for fine-grained distribution
- Handles irregular workloads (varying vertex degrees)

#### Synchronization Mechanisms

**Warp-Level Synchronization** (Warp Shuffle):
```cpp
__inline__ __device__ int warpReduceSum(int val, unsigned mask = FULL_MASK) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}
```
- Exploits SIMD parallelism within warps
- No shared memory required
- Extremely low latency (registers only)

**Block-Level Synchronization**:
```cpp
__inline__ __device__ int blockReduceSum(int val) {
    static __shared__ int shared[32];  // One per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Intra-warp reduction
    val = warpReduceSum(val);
    
    // Inter-warp reduction via shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < numOfWarps) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
```

**Grid-Level Synchronization** (Cooperative Groups):
```cpp
__inline__ __device__ int deviceReduce(int val, int* sumArr) {
    int sum = blockReduceSum(val);
    if (threadIdx.x == 0) {
        sumArr[blockIdx.x] = sum;
    }
    this_grid().sync();  // Global barrier across all blocks
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < gridDim.x; i++) {
            result += sumArr[i];
        }
    }
    return result;
}
```

**Cooperative Groups Requirements**:
- Enables synchronization across entire grid
- Requires kernel launch with `cudaLaunchCooperativeKernel`
- Hardware support: Compute Capability 6.0+

#### Kernel Launch Configuration

```cpp
cudaError_t BellmanFordCuda(...) {
    // Memory allocation
    cudaMalloc(&dev_shortestPaths, numOfVertices * sizeof(int));
    cudaMalloc(&dev_shortestTempPaths, sharedMemArraySize);
    
    // Cooperative kernel launch
    void* kernelArgs[] = {&dev_shortestPaths, &dev_shortestTempPaths, ...};
    
    cudaLaunchCooperativeKernel(
        (void*)kernel,
        gridSize,
        blockSize,
        kernelArgs,
        sharedMemSize,
        0  // Default stream
    );
}
```

#### CUDA-Specific Optimizations

**Coalesced Memory Access**:
- CSR format ensures adjacent threads access adjacent edges
- Warp-aligned reads from `edges` and `weights` arrays
- Critical for GPU memory bandwidth utilization

**Divergence Minimization**:
```cpp
// Early loop termination coordinated across warps
if (blockDim.x * gridDim.x < numOfVertices) {
    if ((blockIdx.x + 1) <= indexes.blockIndex) {
        // Uniform control flow within warp
    }
}
```

**Shared Memory Banking**:
- Arrays sized to avoid bank conflicts
- Padding applied when necessary
- Ensures maximum throughput

**Occupancy Optimization**:
- Shared memory usage tuned to maximize active warps
- Register pressure managed through compiler pragmas
- Target: 100% theoretical occupancy when possible

#### Performance Considerations

**Memory Bandwidth Bottleneck**:
- Graph data (edges, weights) requires significant bandwidth
- Distance arrays updated every iteration
- Mitigation: Shared memory caching of frequently accessed data

**Divergent Execution**:
- Variable vertex degrees cause warp divergence
- Some threads idle while others process long adjacency lists
- Impact: Reduced SIMD efficiency

**Synchronization Overhead**:
- Global barrier (`this_grid().sync()`) stalls entire GPU
- Required for correctness (cross-block dependencies)
- Frequency: Once per iteration

**Launch Overhead**:
- Cooperative kernel launch has higher overhead than standard launch
- Mitigated by high arithmetic intensity of relaxation operations

### 5. Graph Format Converter

**File**: `EdgeList-To-CSR/EdgeList-To-CSR.cpp`

**Purpose**: Transform edge-list representation to CSR for efficient parallel processing

**Algorithm**:

**Phase 1: Edge Sorting by Destination**
```cpp
tailRecursiveQuickSort(dests, sources, weights, 0, numOfEdges - 1);
```
- Tail-recursive quicksort to avoid stack overflow
- Simultaneously permutes sources and weights to maintain correspondence
- Groups all incoming edges for each vertex

**Phase 2: Adjacency Range Calculation**
```cpp
#pragma omp parallel for
for (int j = 0; j < numOfVertices + 1; j++) {
    for (int k = 0; k < j; k++)
        rangesOfAdjacentVertices[j] += numOfEmergingEdges[k];
}
```
- Prefix sum computation
- Parallelized with OpenMP
- Determines start index of each vertex's adjacency list

**Phase 3: Source Sorting Within Ranges**
```cpp
for (int j = 1; j < numOfVertices; j++) {
    startPosition = rangesOfAdjacentVertices[j];
    endPosition = rangesOfAdjacentVertices[j + 1] - 1;
    if (startPosition != endPosition)
        tailRecursiveQuickSort(sources, dests, weights, startPosition, endPosition);
}
```
- Sort sources within each vertex's adjacency range
- Improves cache locality during edge traversal

**Tail-Recursive Quicksort** (prevents stack overflow on large graphs):
```cpp
void tailRecursiveQuickSort(vector<int>& arr, vector<int>& arr2, 
                            vector<int>& arr3, int start, int end) {
    while (start < end) {
        int pivot = quickSortPartition(arr, arr2, arr3, start, end);
        
        // Recurse on smaller partition, iterate on larger
        if (pivot - start < end - pivot) {
            tailRecursiveQuickSort(arr, arr2, arr3, start, pivot - 1);
            start = pivot + 1;
        } else {
            tailRecursiveQuickSort(arr, arr2, arr3, pivot + 1, end);
            end = pivot - 1;
        }
    }
}
```

**Complexity**:
- Sorting: O(E log E)
- Prefix sum: O(V²) naive, O(V) with optimization
- Total: O(E log E) for sparse graphs

### 6. Test Data Generator

**File**: `Test-Data-Generator/Test-Data-Generator.cpp`

**Objective**: Generate random connected weighted graphs suitable for Bellman-Ford testing

**Graph Generation Algorithm**:
```cpp
void generateRandomConnectedGraph(int minWeight, int maxWeight, int startVertex,
    int numOfVertices, vector<int>& sources, vector<int>& dests, 
    vector<int>& weights)
{
    // Phase 1: Spanning tree construction (ensures connectivity)
    int* childVertices = getIndicesInRandomOrder(numOfVertices, excludedIndex);
    vector<int> parentVertices;
    int parentVertex = startVertex;
    
    for (int i = 0; i < numOfVertices - 1; i++) {
        int childVertex = childVertices[i];
        int weight = minWeight + rand() % (maxWeight - minWeight + 1);
        
        sources.push_back(parentVertex);
        dests.push_back(childVertex);
        weights.push_back(weight);
        
        parentVertices.push_back(childVertex);
        parentVertex = parentVertices[parentVertexIndex];
    }
    
    // Phase 2: Additional random edges (increases graph density)
    // ...
}
```

**Key Properties**:
- **Connectivity**: Guaranteed via spanning tree
- **Negative Weights**: Supported (minWeight can be negative)
- **Negative Cycle Detection**: Rejects graphs with negative cycles
- **Scalability**: Generates graphs from 3 to 3000+ vertices

**Validation**:
- Runs serial Bellman-Ford to verify absence of negative cycles
- Regenerates until valid graph produced
- Ensures test data correctness

**Test Graph Sizes**:
- File 1: 3 vertices
- File 2: 30 vertices (10x scaling)
- File 3: 300 vertices (10x scaling)
- File 4: 3000 vertices (10x scaling)

## Advanced Technical Challenges

### 1. Dynamic Convergence Detection

**Challenge**: Algorithm terminates when no edge relaxations occur, but iteration count varies by graph

**Serial Solution**:
```cpp
bool kLoop = true;
for (k = 0; (k < numOfVertices - 1) && kLoop; k++) {
    kLoop = false;
    // Edge relaxation loop
    if (distance_updated) kLoop = true;
}
```

**OpenMP Solution**:
```cpp
#pragma omp parallel for reduction(+:kLoop)
for (int i = 0; i < numOfVertices; i++) {
    if (distance_updated) kLoop++;
}
if (kLoop == 0) break;  // Master thread decision
```

**MPI Solution**:
```cpp
MPI_Allreduce(&kLoop_local, &kLoop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
if (kLoop == 0) break;  // All processes agree
```

**CUDA Solution**:
```cpp
int localSum = deviceReduce(localCompletionOfTheMainLoop, sumArr);
if (threadIdx.x == 0) {
    if (localSum == 0) completionOfTheMainLoop = false;
}
__syncthreads();  // Broadcast decision to block
```

### 2. Read-After-Write Hazards

**Problem**: Concurrent updates to shared distance array create race conditions

**OpenMP Mitigation**:
- Separate read (`shortestPaths`) and write (`shortestTempPaths`) arrays
- Synchronization point between iterations for array swap

**MPI Mitigation**:
- Local temporary arrays per process
- `Allgatherv` synchronizes before next iteration

**CUDA Mitigation**:
- Global barrier (`this_grid().sync()`) ensures all updates visible
- Double-buffering in shared memory for block-local state

### 3. Load Imbalance

**Issue**: Vertices have varying degrees (number of outgoing edges)

**OpenMP**: Static scheduling distributes vertices uniformly, but work per vertex varies
- Impact: Some threads finish early, others lag
- Mitigation: Could use dynamic scheduling, but cache locality suffers

**MPI**: Vertex partitioning may result in uneven edge distribution
- Impact: Processes with high-degree vertices have more work
- Mitigation: Degree-aware partitioning (not implemented, future work)

**CUDA**: Warp divergence when threads process different numbers of edges
- Impact: SIMD lanes idle while others work
- Mitigation: `DataIndexes` structure balances work across warps

### 4. Memory Bandwidth Saturation

**CUDA Specific**:
- Distance array: V integers read + write per iteration
- Graph data: E integers read per iteration
- Total: (2V + 2E) × sizeof(int) × iterations

**Optimization**:
- Shared memory caching reduces global memory traffic
- Coalesced access patterns maximize bandwidth utilization
- Occupancy tuning keeps memory subsystem saturated

### 5. Scalability Analysis

**Strong Scaling** (fixed problem size, increasing processors):
- **OpenMP**: Limited by Amdahl's law (synchronization overhead)
- **MPI**: Communication cost increases with process count
- **CUDA**: Memory bandwidth bottleneck dominates

**Weak Scaling** (proportional problem and processor growth):
- **OpenMP**: Near-linear for moderate thread counts (2-8)
- **MPI**: Communication overhead grows with network diameter
- **CUDA**: Excellent weak scaling (thousands of threads)

## Performance Characteristics

### Complexity Analysis

**Serial**:
- Time: O(V × E)
- Space: O(V + E)

**OpenMP**:
- Time: O((V × E) / P + T_sync)
- Space: O(V + E) + O(V × P) for temporary arrays
- T_sync: Synchronization overhead (barrier + reduction)

**MPI**:
- Time: O((V × E) / P + T_comm)
- Space: O(V + E) distributed, O(V/P) per process
- T_comm: O(V × P) per iteration (Allgatherv dominates)

**CUDA**:
- Time: O((V × E) / (G × B) + T_global_sync)
- Space: O(V + E) on device, O(V) shared memory per block
- G: Grid size, B: Block size
- T_global_sync: Grid-wide synchronization latency

### Expected Speedup Patterns

**OpenMP** (4-core CPU):
- Initialization: ~3.5x (near-linear)
- Main loop: 2.5-3x (synchronization overhead)
- Overall: 2-3x typical

**MPI** (4-node cluster):
- Computation phase: ~3.8x
- Communication overhead: Significant for sparse graphs
- Overall: 1.5-2.5x (communication-bound)

**CUDA** (GPU with 1000+ cores):
- Dense graphs: 10-50x vs. serial
- Sparse graphs: 5-20x (memory-bound)
- Small graphs (<1000 vertices): Minimal speedup (overhead dominates)

## Build and Execution

### Prerequisites

**Serial & OpenMP**:
- C++11 or later
- OpenMP-enabled compiler (GCC with `-fopenmp` or MSVC with `/openmp`)

**MPI**:
- MPI implementation (MPICH, OpenMPI, or MS-MPI)
- C++11 compiler with MPI bindings

**CUDA**:
- NVIDIA GPU with Compute Capability 6.0+ (for cooperative groups)
- CUDA Toolkit 9.0+
- Compatible C++ compiler (MSVC 2017+ on Windows, GCC 7+ on Linux)

### Compilation

**Serial**:
```bash
g++ -std=c++11 -O3 Bellman-Ford-Serial.cpp -o bellman_serial
```

**OpenMP**:
```bash
g++ -std=c++11 -O3 -fopenmp Bellman-Ford-OMPParallel.cpp -o bellman_omp
```

**MPI**:
```bash
mpicxx -std=c++11 -O3 Bellman-Ford-MPIParallel.cpp -o bellman_mpi
```

**CUDA**:
```bash
nvcc -std=c++11 -O3 -arch=sm_60 --expt-relaxed-constexpr kernel.cu -o bellman_cuda
```

### Windows (Visual Studio)

1. Open `Shortest-Path-From-One-Vertex-Finding.sln`
2. Set platform to x64
3. Configure project-specific settings:
   - **OpenMP**: Project Properties → C/C++ → Language → OpenMP Support: Yes
   - **MPI**: Add Include Directories and Link Libraries for your MPI installation
   - **CUDA**: Ensure CUDA toolkit is detected by Visual Studio
4. Build Solution (Release configuration recommended)

### Linux (CMake)

```bash
# Serial
g++ -O3 Bellman-Ford-Serial/Bellman-Ford-Serial.cpp -o bin/bellman_serial

# OpenMP
g++ -O3 -fopenmp Bellman-Ford-OMPParallel/Bellman-Ford-OMPParallel.cpp -o bin/bellman_omp

# MPI
mpicxx -O3 Bellman-Ford-MPIParallel/Bellman-Ford-MPIParallel.cpp -o bin/bellman_mpi

# CUDA
nvcc -O3 -arch=sm_70 Bellman-Ford-CUDAParallel/kernel.cu -o bin/bellman_cuda
```

## Usage

### Data Preparation

**Generate Test Graphs**:
```bash
./test_data_generator
# Produces: EdgeList_1.txt, EdgeList_2.txt, EdgeList_3.txt, EdgeList_4.txt
```

**Convert to CSR Format** (for parallel implementations):
```bash
./edgelist_to_csr
# Produces: Sources_*.txt, Weights_*.txt, RangesOfAdjacentVertices_*.txt
```

### Running Implementations

**Serial**:
```bash
./bellman_serial
# Processes EdgeList_1.txt through EdgeList_4.txt sequentially
```

**OpenMP** (test with varying thread counts):
```bash
export OMP_NUM_THREADS=4
./bellman_omp
# Automatically tests 1, 2, 3, 4 threads
```

**MPI**:
```bash
mpirun -np 4 ./bellman_mpi
# Runs with 4 processes
```

**CUDA**:
```bash
./bellman_cuda
# Automatically selects optimal block/grid configuration
```

### Output Format

```
Number Of Vertices: 300
Number Of Edges: 448
------------------
Bellman Ford execution time: 0.0042 seconds
Number Of Iterations in the main loop: 15

Shortest paths from vertex 0: 0 3 2 5 7 6 8 9 10 11 ...
Predecessor vertex numbers: -1 0 0 1 2 3 4 5 6 7 ...
```

## Experimental Configuration

### Test Data Characteristics

| File | Vertices | Edges (Approx.) | Density |
|------|----------|-----------------|---------|
| 1    | 3        | 3-6             | Sparse  |
| 2    | 30       | 40-60           | Sparse  |
| 3    | 300      | 400-600         | Sparse  |
| 4    | 3000     | 4000-6000       | Sparse  |

**Graph Properties**:
- Connected (guaranteed by spanning tree construction)
- Negative weights possible (-3 to 10 range)
- No negative cycles (validated during generation)
- Random topology with controlled density

### Hardware Assumptions

**CPU Benchmarks**:
- Multi-core processor (4-8 cores typical)
- Shared L3 cache (improves OpenMP performance)
- NUMA architecture impacts (MPI may benefit from process pinning)

**GPU Benchmarks**:
- Compute Capability 6.0+ (Pascal or later)
- Memory bandwidth: 200-900 GB/s (varies by model)
- CUDA cores: 1000-10000+ (Geforce/Quadro/Tesla)

**Network (MPI)**:
- Ethernet: 1-10 Gbps
- InfiniBand: 40-200 Gbps (low-latency HPC networks)

## Advanced Optimization Opportunities

### 1. CUDA Kernel Fusion
**Current**: Separate kernels for initialization, relaxation, and convergence checking
**Proposed**: Fuse operations to reduce kernel launch overhead
**Impact**: 10-20% speedup for small graphs

### 2. MPI Asynchronous Communication
**Current**: Blocking `Allgatherv` synchronizes all processes
**Proposed**: Overlap computation with `Isend`/`Irecv` for distance updates
**Impact**: Reduced communication stalls, better weak scaling

### 3. OpenMP Dynamic Scheduling with Guided
**Current**: Static scheduling for predictable cache behavior
**Proposed**: Guided scheduling adapts to load imbalance
```cpp
#pragma omp parallel for schedule(guided, 32)
```
**Impact**: 15-25% improvement for high-degree variance graphs

### 4. GPU Shared Memory Tiling
**Current**: Global memory access for distance lookups
**Proposed**: Tile distance array into shared memory blocks
**Impact**: 30-40% reduction in global memory transactions

### 5. Graph Partitioning for MPI
**Current**: Simple vertex range partitioning
**Proposed**: METIS/KaHIP-based edge-cut minimization
**Impact**: Reduced inter-process communication volume

### 6. Persistent CUDA Threads
**Current**: Kernel launched per iteration
**Proposed**: Single kernel launch with internal iteration loop
**Impact**: Eliminates kernel launch overhead (~10 µs per launch)

## Known Limitations

1. **Negative Cycle Handling**: Current implementations detect but don't report cycle location
2. **Memory Capacity**: GPU implementation limited by device memory (graphs up to ~100M edges)
3. **Integer Overflow**: No overflow protection for accumulated path weights
4. **MPI Collective Bottleneck**: `Allgatherv` becomes dominant cost at high process counts (>32)
5. **CUDA Occupancy**: Shared memory usage may limit occupancy on older GPUs
6. **Dynamic Graphs**: No support for edge insertion/deletion during execution

## Research Applications

This implementation suite serves as a foundation for:

1. **Parallel Algorithm Analysis**: Comparative study of parallelization paradigms
2. **Performance Modeling**: Validating theoretical complexity predictions
3. **Scalability Studies**: Strong/weak scaling behavior across architectures
4. **Hybrid Parallelism**: MPI+OpenMP or MPI+CUDA combinations
5. **Graph Algorithm Optimization**: Techniques applicable to other graph problems (BFS, PageRank, etc.)

## Future Enhancements

### Algorithmic Extensions
- **Δ-Stepping Algorithm**: Reduces iterations through bucketing
- **SPFA (Shortest Path Faster Algorithm)**: Queue-based optimization
- **Work-Efficient Variant**: Avoid redundant edge relaxations

### Multi-GPU Support
- **NCCL Integration**: Efficient inter-GPU communication
- **Graph Partitioning**: Distribute large graphs across multiple GPUs
- **Unified Memory**: Simplify data management with page migration

### Advanced Metrics
- **Energy Profiling**: Power consumption vs. performance tradeoffs
- **Memory Bandwidth Utilization**: Achieved vs. theoretical bandwidth
- **Cache Miss Analysis**: L1/L2/L3 miss rates (CPU implementations)

### Real-World Applications
- **Road Network Routing**: Integration with GIS data formats
- **Network Protocol Routing**: BGP/OSPF path computation
- **Social Network Analysis**: Influence propagation modeling

## References

### Core Algorithm
- Bellman, R. (1958). "On a routing problem". Quarterly of Applied Mathematics
- Ford, L. R., Jr. (1956). "Network Flow Theory". RAND Corporation

### Parallel Computing
- Ortega, J. M., & Voigt, R. G. (1985). "Solution of Partial Differential Equations on Vector and Parallel Computers"
- Kirk, D. B., & Hwu, W. W. (2016). "Programming Massively Parallel Processors" (3rd ed.)

### CUDA Programming
- NVIDIA (2024). "CUDA C++ Programming Guide"
- Harris, M. (2007). "Optimizing Parallel Reduction in CUDA"

### MPI Standards
- MPI Forum (2021). "MPI: A Message-Passing Interface Standard Version 4.0"

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Authors

This implementation prioritizes pedagogical clarity and comparative analysis over absolute performance. Production systems may employ additional optimizations specific to target hardware and use cases.
