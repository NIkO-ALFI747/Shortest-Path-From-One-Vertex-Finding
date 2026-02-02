#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

using namespace std;
using namespace cooperative_groups;

int getOptimalBlockSize(int numOfThreads, int optimalMaxBlockSize);

struct File {
    string fileName;
    File() {}
    File(int fileNameNo) { autoSetFileName(fileNameNo); }
    File(string fileName) { this->fileName = fileName; }
    void autoSetFileName(int fileNameNo) { this->fileName = "file_" + to_string(fileNameNo) + ".txt"; }
    size_t getFileSize();
    string fileToBuf();
};

struct Array {
    static vector<int> fillArrayFromString(string str, char separator);
    static void printVector(vector <int>& vec, int size);
    static void printArray(int* array, int length);
};

struct FileVector {
    vector <File> file;
    FileVector(int numOfFiles, string fileName);
    vector<int> fileToArr(size_t fileIndex);
};

struct DataIndexes {
    int blockIndex;
    int warpsThreadIndexBeforeBlockIndex;
    int warpsMaxNumOfIterationsBeforeBlockIndex;
    int warpsThreadIndexAfterBlockIndex;
    int warpsMaxNumOfIterationsAfterBlockIndex;
    DataIndexes(int numOfData, int gridSize, int blockSize, int warpSize);
    void setVarpsIndexes(int blockPartOfData, int blockSize, int warpSize, int& warpsThreadIndex, int& warpsMaxNumOfIterations);
};

__inline__ __device__ int smallWarpReduceSum(int val);

__inline__ __device__ int warpReduceSum(int val, unsigned mask);

__inline__ __device__ int blockReduceSum(int val);

__inline__ __device__ int deviceReduce(int val, int* sumArr);

__inline__ __device__ void deviceBellmanFord(int threadId, int indexForArr1, int indexForArr2, int indexForGlobalMem, int numOfIterations,
    int minIterations, int* shortestPaths, int* shortestTempPaths, int* previousVertices, int startVertex, const int* edges, const int* weights,
    const int* rangesOfAdjacentVertices, int numOfVertices, int* sumArr, int* numOfGlobalIterations);

__global__ void kernel(int* shortestPaths, int* shortestTempPaths, int* previousVertices, int* sumArr, const int* edges,
    const int* weights, const int* rangesOfAdjacentVertices, int numOfVertices, int startVertex, int numOfSharedArrays, DataIndexes indexes, int* numOfIterations);

cudaError_t BellmanFordCuda(int* shortestPaths, int* previousVertices, float* gpuTime, vector <int> edges,
    vector <int> weights, vector <int> rangesOfAdjacentVertices, int startVertex, int& numOfIterations);

int main(int argc, char** argv) {
    int startVertex = 0;
    size_t numOfVertices;
    int printMode = 0;
    int numOfFiles = 4;
    FileVector edgesFiles(numOfFiles, "Sources_");
    FileVector weightsFiles(numOfFiles, "Weights_");
    FileVector rangesOfAdjacentVerticesFiles(numOfFiles, "RangesOfAdjacentVertices_");
    vector <int> rangesOfAdjacentVertices, edges, weights;
    int* shortestPaths, * previousVertices, numOfIterations = 0;
    float gpuTime = 0.0;
    for (int i = 0; i < numOfFiles; i++) {
        edges = edgesFiles.fileToArr((size_t)i);
        weights = weightsFiles.fileToArr((size_t)i);
        rangesOfAdjacentVertices = rangesOfAdjacentVerticesFiles.fileToArr((size_t)i);
        numOfVertices = rangesOfAdjacentVertices.size() - 1;
        printf("\nNumber Of Vertices: %d\n", numOfVertices);
        printf("Number Of Edges: %d\n", edges.size());
        shortestPaths = new int[numOfVertices];
        previousVertices = new int[numOfVertices];
        BellmanFordCuda(shortestPaths, previousVertices, &gpuTime, edges, weights, rangesOfAdjacentVertices, startVertex, numOfIterations);
        if (printMode) {
            printf("\nShortest paths from vertex %d: ", startVertex);
            Array::printArray(shortestPaths, 10);
            printf("Predecessor vertex numbers for each vertex: ");
            Array::printArray(previousVertices, 10);
            printf("----------");
        }
        printf("\nTime: %f (ms)\n", gpuTime);
        printf("Num Of Iterations: %d\n", numOfIterations);
        delete[] shortestPaths;
        delete[] previousVertices;
    }
    return 0;
}

int getOptimalBlockSize(int numOfThreads, int optimalMaxBlockSize)
{
    int blockSize = 0;
    int gridSize = 0;
    int i = 2;
    int numOfIterations = static_cast<int>(sqrt(numOfThreads)) + 2;
    while (i < numOfIterations) {
        if (numOfThreads % i == 0) {
            if (gridSize == 0) {
                gridSize = i;
                blockSize = numOfThreads / i;
            }
            if (numOfThreads / i >= optimalMaxBlockSize) blockSize = numOfThreads / i;
            int tmp = 1;
            numOfThreads /= i;
            while (numOfThreads % i == 0) {
                if (numOfThreads / i >= optimalMaxBlockSize) blockSize = numOfThreads / i;
                tmp += 1;
                numOfThreads /= i;
            }
            numOfIterations = static_cast<int>(sqrt(numOfThreads)) + 2;
        }
        i += 1;
    }
    return blockSize;
}

size_t File::getFileSize() {
    FILE* file;
    fopen_s(&file, this->fileName.c_str(), "r");
    if (!file) return 0;
    size_t fileSize;
    fpos_t position;
    if (fgetpos(file, &position)) {
        fclose(file);
        return 0;
    }
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    if (fsetpos(file, &position)) {
        fclose(file);
        return 0;
    }
    fclose(file);
    return fileSize;
}

string File::fileToBuf() {
    size_t fileSize = this->getFileSize();
    char* buf = new char[fileSize];
    ifstream inFile(this->fileName, ios::binary);
    if (!inFile) return "Failed to open file!";
    inFile.read(buf, fileSize);
    inFile.close();
    string str(buf, fileSize);
    delete[] buf;
    return str;
}

vector<int> Array::fillArrayFromString(string str, char separator) {
    vector<int> arr;
    string number = "";
    for (int i = 0; i < str.size(); i++) {
        if ((str[i] != separator) && (str[i] != '\n')) {
            number += str[i];
        }
        if (str[i] == ' ') {
            arr.push_back(atoi(number.c_str()));
            number = "";
        }
    }
    return arr;
}

void Array::printVector(vector <int>& vec, int size = 0) {
    if (size == 0) size = vec.size();
    for (int i = 0; i < size && i < vec.size(); i++)
        cout << vec[i] << " ";
    cout << "\n";
}

void Array::printArray(int* array, int length) {
    for (int i = 0; i < length; i++)
        cout << array[i] << " ";
    cout << "\n";
}

FileVector::FileVector(int numOfFiles, string fileName) {
    file = vector<File>(numOfFiles);
    for (int i = 0; i < numOfFiles; i++) {
        file[i].fileName = fileName + to_string(i + 1) + ".txt";
    }
}

vector<int> FileVector::fileToArr(size_t fileIndex) {
    vector<int> arr;
    string arrStr = file[fileIndex].fileToBuf();
    arr = Array::fillArrayFromString(arrStr, ' ');
    return arr;
}

DataIndexes::DataIndexes(int numOfData, int gridSize, int blockSize, int warpSize)
{
    int blockPartOfData = numOfData / gridSize;
    blockIndex = gridSize - (numOfData % gridSize);
    setVarpsIndexes(blockPartOfData, blockSize, warpSize,
        warpsThreadIndexBeforeBlockIndex, warpsMaxNumOfIterationsBeforeBlockIndex
    );
    setVarpsIndexes(blockPartOfData + 1, blockSize, warpSize,
        warpsThreadIndexAfterBlockIndex, warpsMaxNumOfIterationsAfterBlockIndex
    );
}

void DataIndexes::setVarpsIndexes(int blockPartOfData, int blockSize, int warpSize, int& warpsThreadIndex, int& warpsMaxNumOfIterations)
{
    int warpsPartOfData = blockPartOfData;
    warpsMaxNumOfIterations = warpsPartOfData / blockSize;
    if (warpsPartOfData % blockSize > 0) warpsMaxNumOfIterations++;
    warpsThreadIndex = blockSize - (warpsPartOfData % blockSize);
}

__inline__ __device__ int smallWarpReduceSum(int val)
{
    if ((blockDim.x & (blockDim.x - 1)) == 0) {
        for (int offset = (blockDim.x % warpSize) / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    else {
        for (int i = (blockDim.x % warpSize) - 1; i > 0; i -= 1) {
            if (threadIdx.x % warpSize == 0) {
                val += __shfl_down_sync(FULL_MASK, val, 1);
            }
            else {
                val = __shfl_down_sync(FULL_MASK, val, 1);
            }
        }
    }
    return val;
}

__inline__ __device__ int warpReduceSum(int val, unsigned mask = FULL_MASK)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__inline__ __device__ int blockReduceSum(int val)
{
    static __shared__ int shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int numOfSumInBlock = blockDim.x / warpSize;
    if (blockDim.x % warpSize != 0) {
        numOfSumInBlock++;
        if ((wid + 1) != numOfSumInBlock) {
            val = warpReduceSum(val);
        }
        else {
            val = smallWarpReduceSum(val);
        }
    }
    else {
        val = warpReduceSum(val);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < numOfSumInBlock) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__inline__ __device__ int deviceReduce(int val, int* sumArr)
{
    int sum = 0;
    if (blockDim.x > warpSize) {
        sum = blockReduceSum(val);
    }
    else {
        unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < blockDim.x);
        sum = warpReduceSum(val, mask);
    }
    if (threadIdx.x == 0) {
        sumArr[blockIdx.x] = sum;
    }
    this_grid().sync();
    int result = 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < gridDim.x; i++) {
            result += sumArr[i];
        }
    }
    return result;
}

__inline__ __device__ void deviceBellmanFord(int threadId, int indexForArr1, int indexForArr2, int indexForGlobalMem, int numOfIterations,
    int minIterations, int* shortestPaths, int* shortestTempPaths, int* previousVertices, int startVertex, const int* edges, const int* weights,
    const int* rangesOfAdjacentVertices, int numOfVertices, int* sumArr, int* numOfGlobalIterations)
{
    int INF = 2000000000;
    int i, k, j;
    for (i = 0; i < minIterations; i += blockDim.x) {
        __syncthreads();
        shortestTempPaths[indexForArr1 + i] = INF;
        previousVertices[indexForArr2 + i] = -1;
        shortestPaths[indexForGlobalMem + i] = INF;
    }
    for (; i < numOfIterations; i += blockDim.x) {
        shortestTempPaths[indexForArr1 + i] = INF;
        previousVertices[indexForArr2 + i] = -1;
        shortestPaths[indexForGlobalMem + i] = INF;
    }
    __shared__ bool completionOfTheMainLoop;
    int localCompletionOfTheMainLoop;
    if (threadId % numOfVertices == startVertex) {
        shortestPaths[startVertex] = 0;
        shortestTempPaths[startVertex] = 0;
    }
    if (threadIdx.x == 0) {
        completionOfTheMainLoop = true;
    }
    int newShortestPath, edge;
    int maxNumOfGlobalIterations = numOfVertices - 1, numOfJCycleIterations;
    this_grid().sync();
    for (i = 0; (i < maxNumOfGlobalIterations) && completionOfTheMainLoop; ++i) {
        localCompletionOfTheMainLoop = 0;
        for (k = 0; k < minIterations; k += blockDim.x) {
            __syncthreads();
            numOfJCycleIterations = rangesOfAdjacentVertices[indexForGlobalMem + k + 1];
            for (j = rangesOfAdjacentVertices[indexForGlobalMem + k]; j < numOfJCycleIterations; ++j) {
                edge = edges[j];
                newShortestPath = shortestPaths[edge] + weights[j];
                if (shortestTempPaths[indexForArr1 + k] > newShortestPath) {
                    shortestTempPaths[indexForArr1 + k] = newShortestPath;
                    previousVertices[indexForArr2 + k] = edge;
                    ++localCompletionOfTheMainLoop;
                }
            }
        }
        for (; k < numOfIterations; k += blockDim.x) {
            numOfJCycleIterations = rangesOfAdjacentVertices[indexForGlobalMem + k + 1];
            for (j = rangesOfAdjacentVertices[indexForGlobalMem + k]; j < numOfJCycleIterations; ++j) {
                edge = edges[j];
                newShortestPath = shortestPaths[edge] + weights[j];
                if (shortestTempPaths[indexForArr1 + k] > newShortestPath) {
                    shortestTempPaths[indexForArr1 + k] = newShortestPath;
                    previousVertices[indexForArr2 + k] = edge;
                    ++localCompletionOfTheMainLoop;
                }
            }
        }
        for (k = 0; k < minIterations; k += blockDim.x) {
            __syncthreads();
            shortestPaths[indexForGlobalMem + k] = shortestTempPaths[indexForArr1 + k];
        }
        for (; k < numOfIterations; k += blockDim.x) {
            shortestPaths[indexForGlobalMem + k] = shortestTempPaths[indexForArr1 + k];
        }
        localCompletionOfTheMainLoop = deviceReduce(localCompletionOfTheMainLoop, sumArr);
        if (threadIdx.x == 0) {
            if (localCompletionOfTheMainLoop == 0) completionOfTheMainLoop = false;
        }
        __syncthreads();
    }
    if (threadId == 1) *numOfGlobalIterations = i;
}

__global__ void kernel(int* shortestPaths, int* shortestTempPaths, int* previousVertices, int* sumArr, const int* edges, const int* weights, const int* rangesOfAdjacentVertices, int numOfVertices, int startVertex, int numOfSharedArrays, DataIndexes indexes, int* numOfIterations)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int indexForGlobalMem = threadIdx.x + blockIdx.x * (numOfVertices / gridDim.x);
    int numOfLocalIterations = blockDim.x, minIterations = blockDim.x;
    if (blockDim.x * gridDim.x < numOfVertices) {
        if ((blockIdx.x + 1) <= indexes.blockIndex) {
            if ((threadIdx.x + 1) > blockDim.x - indexes.warpsThreadIndexBeforeBlockIndex) {
                numOfLocalIterations *= (indexes.warpsMaxNumOfIterationsBeforeBlockIndex - 1);
            }
            else {
                numOfLocalIterations *= indexes.warpsMaxNumOfIterationsBeforeBlockIndex;
            }
            minIterations *= (indexes.warpsMaxNumOfIterationsBeforeBlockIndex - 1);
        }
        else {
            if ((threadIdx.x + 1) > blockDim.x - indexes.warpsThreadIndexAfterBlockIndex) {
                numOfLocalIterations *= (indexes.warpsMaxNumOfIterationsAfterBlockIndex - 1);
            }
            else {
                numOfLocalIterations *= indexes.warpsMaxNumOfIterationsAfterBlockIndex;
            }
            minIterations *= (indexes.warpsMaxNumOfIterationsAfterBlockIndex - 1);
        }
    }

    if (numOfSharedArrays == 2) {
        int indexForSharedMem = threadIdx.x, i;
        extern __shared__ int sharedMemory[];
        int* shortestTempPathsShared = &sharedMemory[0];
        int* previousVerticesShared = &sharedMemory[numOfVertices / gridDim.x];
        deviceBellmanFord(threadId, indexForSharedMem, indexForSharedMem, indexForGlobalMem, numOfLocalIterations,
            minIterations, shortestPaths, shortestTempPathsShared, previousVerticesShared, startVertex, edges, weights,
            rangesOfAdjacentVertices, numOfVertices, sumArr, numOfIterations);
        for (i = 0; i < minIterations; i += blockDim.x) {
            __syncthreads();
            previousVertices[indexForGlobalMem + i] = previousVerticesShared[indexForSharedMem + i];
        }
        for (; i < numOfLocalIterations; i += blockDim.x) {
            previousVertices[indexForGlobalMem + i] = previousVerticesShared[indexForSharedMem + i];
        }
    }
    else if (numOfSharedArrays == 1) {
        int indexForSharedMem = threadIdx.x;
        extern __shared__ int sharedMemory[];
        int* shortestTempPathsShared = &sharedMemory[0];
        deviceBellmanFord(threadId, indexForSharedMem, indexForGlobalMem, indexForGlobalMem, numOfLocalIterations,
            minIterations, shortestPaths, shortestTempPathsShared, previousVertices, startVertex, edges, weights,
            rangesOfAdjacentVertices, numOfVertices, sumArr, numOfIterations);
    }
    else {
        deviceBellmanFord(threadId, indexForGlobalMem, indexForGlobalMem, indexForGlobalMem, numOfLocalIterations,
            minIterations, shortestPaths, shortestTempPaths, previousVertices, startVertex, edges, weights,
            rangesOfAdjacentVertices, numOfVertices, sumArr, numOfIterations);
    }
}

cudaError_t BellmanFordCuda(int* shortestPaths, int* previousVertices, float* gpuTime, vector<int> edges, vector<int> weights, vector<int> rangesOfAdjacentVertices, int startVertex, int& numOfIterations)
{
    int* dev_edges = 0;
    int* dev_weights = 0;
    int* dev_rangesOfAdjacentVertices = 0;
    int* dev_shortestPathsToVertices = 0;
    int* dev_shortestTempPaths = 0;
    int* dev_previousVertices = 0;
    int* dev_sumArr = 0;
    int* dev_numOfIterations = 0;
    cudaError_t cudaStatus;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int cudaVersion;
    cudaError_t cudaError = cudaRuntimeGetVersion(&cudaVersion);
    if (cudaError != cudaSuccess) {
        std::cerr << "Failed to get CUDA version: " << cudaGetErrorString(cudaError) << std::endl;
    }
    int major = cudaVersion / 1000;
    int minor = (cudaVersion % 1000) / 10;
    int ptxasNumOfRegistersUsedByThread = 52;
    int ptxasConstSharedMemoryUsedByBlock = 144;
    int constSharedMemoryUsedByBlock = sizeof(bool) + (32 * sizeof(int));
    int numOfVertices = rangesOfAdjacentVertices.size() - 1;
    int minNumOfBlocksPerMultiprocessorForMaxLoadBySharedMem = devProp.sharedMemPerMultiprocessor / (devProp.sharedMemPerBlock - devProp.reservedSharedMemPerBlock);
    if (devProp.sharedMemPerMultiprocessor % (devProp.sharedMemPerBlock - devProp.reservedSharedMemPerBlock) > 0)
        minNumOfBlocksPerMultiprocessorForMaxLoadBySharedMem += 1;
    int minNumOfBlocksPerMultiprocessorForMaxLoadByRegs = devProp.regsPerMultiprocessor / devProp.regsPerBlock;
    if (devProp.regsPerMultiprocessor % devProp.regsPerBlock > 0)
        minNumOfBlocksPerMultiprocessorForMaxLoadByRegs += 1;
    int numOfThreadsPerMultiprocessorForMaxLoadByRegs = devProp.regsPerMultiprocessor / ptxasNumOfRegistersUsedByThread;
    if (devProp.regsPerMultiprocessor % ptxasNumOfRegistersUsedByThread > 0)
        numOfThreadsPerMultiprocessorForMaxLoadByRegs += 1;
    int minNumOfBlocksPerMultiprocessorForMaxLoad = max(minNumOfBlocksPerMultiprocessorForMaxLoadBySharedMem, minNumOfBlocksPerMultiprocessorForMaxLoadByRegs);
    int numOfThreadsPerMultiprocessorForMaxLoad = min(numOfThreadsPerMultiprocessorForMaxLoadByRegs, devProp.maxThreadsPerMultiProcessor);
    int minGridSizeForMaxLoad = devProp.multiProcessorCount * minNumOfBlocksPerMultiprocessorForMaxLoad;
    int maxBlockSizeForMaxLoad = min(devProp.maxThreadsPerBlock, numOfThreadsPerMultiprocessorForMaxLoad / minNumOfBlocksPerMultiprocessorForMaxLoad);
    int maxNumOfActiveThreads = devProp.multiProcessorCount * numOfThreadsPerMultiprocessorForMaxLoad;
    int optimalMaxBlockSize = 256;
    int optimalMaxNumOfWarpsPerBlock = optimalMaxBlockSize / devProp.warpSize;
    int blockSize;
    int gridSize;
    if (numOfVertices <= optimalMaxBlockSize) {
        blockSize = numOfVertices;
        gridSize = 1;
    }
    else if (numOfVertices <= maxNumOfActiveThreads) {
        int numOfWarpsPerData = numOfVertices / devProp.warpSize;

        blockSize = getOptimalBlockSize(numOfVertices, optimalMaxBlockSize);
        gridSize = numOfVertices / blockSize;
    }
    else if (numOfVertices > maxNumOfActiveThreads) {
        blockSize = optimalMaxBlockSize; // 256
        gridSize = maxNumOfActiveThreads / blockSize; // 64
    }
    int maxNumOfActiveBlocksPerMultiprocessor = gridSize / devProp.multiProcessorCount;
    if (gridSize % devProp.multiProcessorCount > 0) maxNumOfActiveBlocksPerMultiprocessor++;
    int sharedMemoryPerBlock = min(
        devProp.sharedMemPerBlock - ptxasConstSharedMemoryUsedByBlock,
        (devProp.sharedMemPerMultiprocessor / maxNumOfActiveBlocksPerMultiprocessor) - ptxasConstSharedMemoryUsedByBlock
    );
    int intSharedMemoryPerBlock = sharedMemoryPerBlock / sizeof(int);
    int numOfIntDataPerBlock = numOfVertices / gridSize;
    int numOfSharedArrays = 0;
    if (numOfIntDataPerBlock < (intSharedMemoryPerBlock / 2))
        numOfSharedArrays = 2;
    else if (numOfVertices < intSharedMemoryPerBlock)
        numOfSharedArrays = 1;
    int dynamicSharedMemory = numOfSharedArrays * numOfIntDataPerBlock * sizeof(int);
    printf("\nCUDA version: %d.%d\n", major, minor);
    printf("computeCapability: %d.%d\n", devProp.major, devProp.minor);
    printf("maxThreadsPerBlock: %d\n", devProp.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("maxThreadsDim: %d,%d,%d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("maxBlocksPerMultiProcessor: %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("maxGridSize: %d,%d,%d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("multiProcessorCount: %d\n", devProp.multiProcessorCount);
    printf("warpSize: %d\n", devProp.warpSize);
    printf("regsPerBlock: %d\n", devProp.regsPerBlock);
    printf("regsPerMultiprocessor: %d\n", devProp.regsPerMultiprocessor);
    printf("sharedMemPerBlock: %d\n", devProp.sharedMemPerBlock);
    printf("sharedMemPerMultiprocessor: %d\n", devProp.sharedMemPerMultiprocessor);
    printf("reservedSharedMemPerBlock: %d\n", devProp.reservedSharedMemPerBlock);
    printf("totalConstMem: %d\n", devProp.totalConstMem);
    printf("\nblockSize: %d\n", blockSize);
    printf("gridSize: %d\n", gridSize);
    printf("numOfSharedArrays: %d\n", numOfSharedArrays);
    printf("dynamicSharedMemory: %d\n", dynamicSharedMemory);
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);
    if (supportsCoopLaunch != 1) throw runtime_error("Cooperative Launch is not supported on this machine configuration.");
    cudaStatus = cudaMalloc((void**)&dev_shortestPathsToVertices, numOfVertices * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_shortestTempPaths, numOfVertices * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_previousVertices, numOfVertices * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_edges, edges.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_weights, weights.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_rangesOfAdjacentVertices, rangesOfAdjacentVertices.size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_sumArr, gridSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_numOfIterations, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_edges, &edges[0], edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_weights, &weights[0], weights.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_rangesOfAdjacentVertices, &rangesOfAdjacentVertices[0], rangesOfAdjacentVertices.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    DataIndexes indexes(numOfVertices, gridSize, blockSize, devProp.warpSize);
    dim3 blocks(gridSize);
    dim3 threads(blockSize);
    void* args[] = {
        &dev_shortestPathsToVertices,
        &dev_shortestTempPaths,
        &dev_previousVertices,
        &dev_sumArr,
        &dev_edges,
        &dev_weights,
        &dev_rangesOfAdjacentVertices,
        &numOfVertices,
        &startVertex,
        &numOfSharedArrays,
        &indexes,
        &dev_numOfIterations
    };
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, 0);
    cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, args, dynamicSharedMemory);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(gpuTime, startTime, stopTime);
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaStatus = cudaMemcpy(shortestPaths, dev_shortestPathsToVertices, numOfVertices * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(previousVertices, dev_previousVertices, numOfVertices * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(&numOfIterations, dev_numOfIterations, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_shortestPathsToVertices);
    cudaFree(dev_shortestTempPaths);
    cudaFree(dev_previousVertices);
    cudaFree(dev_sumArr);
    cudaFree(dev_edges);
    cudaFree(dev_weights);
    cudaFree(dev_rangesOfAdjacentVertices);
    cudaFree(dev_numOfIterations);
    return cudaStatus;
}