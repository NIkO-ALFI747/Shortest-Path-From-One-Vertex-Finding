#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <mpi.h>

using namespace std;

struct File {
    string fileName;
    File() {}
    File(int fileNameNo) { autoSetFileName(fileNameNo); }
    File(string fileName) { this->fileName = fileName; }
    void autoSetFileName(int fileNameNo) { this->fileName = "file_" + to_string(fileNameNo) + ".txt"; }
    void setFileName(string fileName) { this->fileName = fileName; }
    size_t getFileSize();
    void bufToFile(string str);
    string fileToBuf();
    string fileToBufMPI();
};

struct Array {
    static vector<int> fillArrayFromString(string str, char separator);
    static vector<vector<int>> fill2DArrayFromString(string str, char separator, int numOfCols);
    static void printVector(vector <int>& vec, int size);
    static void printArray(int* array, int length);
};

struct FileVector {
    vector <File> file;
    FileVector(int numOfFiles, string fileName);
    vector<int> fileToArr(size_t fileIndex);
    vector<vector<int>> fileTo2DArr(size_t fileIndex);
};

struct Process {
    int rank, size, partOfArr, startIndex, stopIndex;
    Process(int rank, int size) { this->rank = rank; this->size = size; }
    void getArrComputationalBoundaries(int arraySize);
};

void BellmanFordParallelMPI(vector <int>& shortestPaths, vector <int>& previousVertices, int numOfVertices, int startVertex,
    vector<int> sources, vector<int> rangesOfAdjacentVertices, vector<int> weights, double& time, int& k, Process proc);

int main(int argc, char** argv)
{
    size_t numOfVertices, numOfEdges;
    int startVertex = 0;
    int printMode = 0;
    int endPrintIndex = 9;
    int numOfFiles = 4;
    double time;
    int k;
    FileVector sourcesFiles(numOfFiles, "Sources_");
    FileVector weightsFiles(numOfFiles, "Weights_");
    FileVector rangesOfAdjacentVerticesFiles(numOfFiles, "RangesOfAdjacentVertices_");
    vector <int> rangesOfAdjacentVertices, sources, weights;
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Process proc(rank, size);
    if (rank == 0) cout << "\nNumber Of Processes: " << size << "\n";
    for (int i = 0; i < numOfFiles; i++) {
        sources = sourcesFiles.fileToArr((size_t)i);
        if (sources.empty()) {
            if (rank == 0) cout << "CRITICAL ERROR: Could not load " << sourcesFiles.file[i].fileName << endl;
            MPI_Finalize();
            return 1;
        }
        weights = weightsFiles.fileToArr((size_t)i);
        rangesOfAdjacentVertices = rangesOfAdjacentVerticesFiles.fileToArr((size_t)i);
        numOfVertices = rangesOfAdjacentVertices.size() - 1;
        numOfEdges = sources.size();
        if (rank == 0) {
            cout << "\nNumber Of Vertices: " << numOfVertices;
            cout << "\nNumber Of Edges: " << numOfEdges;
        }
        vector<int> shortestPaths(numOfVertices), previousVertices(numOfVertices);
        MPI_Barrier(MPI_COMM_WORLD);
        BellmanFordParallelMPI(shortestPaths, previousVertices, numOfVertices,
            startVertex, sources, rangesOfAdjacentVertices, weights, time, k, proc);
        if (rank == 0) {
            printf("\n------------------\nBellman Ford parallel mpi execution time: %f seconds\n", time);
            cout << "Number Of Iterations in the main loop: " << k + 1 << "\n";
            if (printMode) {
                cout << "\nShortest paths from vertex " << startVertex << ": ";
                Array::printVector(shortestPaths, endPrintIndex + 1);
                printf("Predecessor vertex numbers for each vertex: ");
                Array::printVector(previousVertices, endPrintIndex + 1);
            }
        }
    }
    MPI_Finalize();
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

void File::bufToFile(string str) {
    bool fileNotEmpty = 0;
    ifstream inFile(fileName);
    if (inFile.is_open()) {
        char symbol;
        inFile.read(&symbol, 1);
        fileNotEmpty = inFile.good();
        inFile.close();
    }
    fstream file;
    file.open(fileName, fileNotEmpty ? (ios::in | ios::out | ios::trunc) : (ios::in | ios::out | ios::app));
    file << str;
    file.close();
}

string File::fileToBuf() {
    size_t fileSize = getFileSize();
    char* buf = new char[fileSize];
    ifstream inFile(fileName, ios::binary);
    if (!inFile) return "Failed to open file!";
    inFile.read(buf, fileSize);
    inFile.close();
    string str(buf, fileSize);
    delete[] buf;
    return str;
}

string File::fileToBufMPI()
{
    size_t fileSize = getFileSize();
    char* buf = new char[fileSize];
    MPI_File handle;
    MPI_Status status;
    if (MPI_File_open(MPI_COMM_SELF, fileName.c_str(), MPI_MODE_RDONLY,
        MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
        printf("Failure in opening the file.\n");
    }
    MPI_File_read(handle, buf, fileSize, MPI_CHAR, &status);
    MPI_File_close(&handle);
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

vector<vector<int>> Array::fill2DArrayFromString(string str, char separator, int numOfCols = 3)
{
    string number = "";
    int row = 0, column = 0;
    for (int i = 0; i < str.size(); i++) {
        if (str[i] == '\n') row++;
    }
    vector<vector<int>> arr(row, vector<int>(numOfCols));
    row = 0;
    for (int i = 0; i < str.size(); i++) {
        if ((str[i] != separator) && (str[i] != '\n')) {
            number += str[i];
        }
        if (str[i] == ' ') {
            arr[row][column] = atoi(number.c_str());
            number = "";
            column++;
            if (column > (numOfCols - 1)) {
                column = 0;
                row++;
            }
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
    string arrStr = file[fileIndex].fileToBufMPI();
    arr = Array::fillArrayFromString(arrStr, ' ');
    return arr;
}

vector<vector<int>> FileVector::fileTo2DArr(size_t fileIndex)
{
    vector<vector<int>> arr;
    string arrStr = file[fileIndex].fileToBuf();
    arr = Array::fill2DArrayFromString(arrStr, ' ');
    return arr;
}

void Process::getArrComputationalBoundaries(int arraySize)
{
    int numberOfAdditionsToIndexes;
    partOfArr = arraySize / size;
    if ((rank + 1) > size - (arraySize % size)) {
        numberOfAdditionsToIndexes = rank - (size - (arraySize % size));
        startIndex = partOfArr * rank + numberOfAdditionsToIndexes;
        partOfArr += 1;
        stopIndex = startIndex + partOfArr;
    }
    else {
        startIndex = partOfArr * rank;
        stopIndex = partOfArr * (rank + 1);
    }
}

void BellmanFordParallelMPI(vector<int>& shortestPaths, vector<int>& previousVertices, int numOfVertices, int startVertex,
    vector<int> sources, vector<int> rangesOfAdjacentVertices, vector<int> weights, double& time, int& k, Process proc)
{
    proc.getArrComputationalBoundaries(numOfVertices);
    int* shortestPaths_link = &shortestPaths[0];
    int* previousVertices_link = &previousVertices[0];
    int* shortestTempPaths = new int[proc.partOfArr];
    int* previousVertices_local = new int[proc.partOfArr];
    int* recvCounts = new int[proc.size];
    int* displs = new int[proc.size];
    Process allProcs(0, proc.size);
    for (int i = 0; i < proc.size; i++) {
        allProcs.rank = i;
        allProcs.getArrComputationalBoundaries(numOfVertices);
        recvCounts[i] = allProcs.partOfArr;
        displs[i] = allProcs.startIndex;
    }
    double mpiTime = MPI_Wtime();
    const int INF = 1e9;
    for (int i = 0; i < proc.partOfArr; i++) {
        shortestTempPaths[i] = INF;
        previousVertices_local[i] = -1;
    }
    for (int i = 0; i < numOfVertices; i++) {
        shortestPaths_link[i] = INF;
    }
    shortestPaths_link[startVertex] = 0;
    if ((startVertex >= proc.startIndex) && (startVertex < proc.stopIndex))
        shortestTempPaths[startVertex - proc.startIndex] = 0;
    int kLoop, kLoop_local = 0;
    for (k = 0; k < numOfVertices - 1; k++)
    {
        kLoop_local = 0;
        for (int i = 0; i < proc.partOfArr; i++)
        {
            for (int j = rangesOfAdjacentVertices[i + proc.startIndex];
                j < rangesOfAdjacentVertices[(i + proc.startIndex) + 1]; j++) {
                if (shortestTempPaths[i] > shortestPaths_link[sources[j]] + weights[j])
                {
                    shortestTempPaths[i] = shortestPaths_link[sources[j]] + weights[j];
                    previousVertices_local[i] = sources[j];
                    kLoop_local++;
                }
            }
        }
        MPI_Allreduce(&kLoop_local, &kLoop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (kLoop == 0) break;
        MPI_Allgatherv(shortestTempPaths, proc.partOfArr, MPI_INT, shortestPaths_link,
            recvCounts, displs, MPI_INT, MPI_COMM_WORLD);
    }
    MPI_Gatherv(previousVertices_local, proc.partOfArr, MPI_INT, previousVertices_link,
        recvCounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    time = MPI_Wtime() - mpiTime;
    delete[] displs;
    delete[] recvCounts;
    delete[] previousVertices_local;
    delete[] shortestTempPaths;
}