#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <omp.h>

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

void BellmanFordParallelOmp(vector <int>& shortestPaths, vector <int>& previousVertices, int numOfVertices, int numOfEdges,
    int startVertex, vector<int> sources, vector<int> rangesOfAdjacentVertices, vector<int> weights, double& time, int& k);

int main()
{
    size_t numOfVertices, numOfEdges;
    int startVertex = 0;
    int printMode = 0;
    int endPrintIndex = 9;
    int numOfFiles = 4;
    int numOfThreads = 1;
    double time;
    int k;
    FileVector sourcesFiles(numOfFiles, "Sources_");
    FileVector weightsFiles(numOfFiles, "Weights_");
    FileVector rangesOfAdjacentVerticesFiles(numOfFiles, "RangesOfAdjacentVertices_");
    vector <int> rangesOfAdjacentVertices, sources, weights;
    for (int t = 1; t <= 4; t++) {
        omp_set_num_threads(t);
        cout << "\n\nNumber Of Threads: " << t << "\n";
        for (int i = 0; i < numOfFiles; i++) {
            sources = sourcesFiles.fileToArr((size_t)i);
            weights = weightsFiles.fileToArr((size_t)i);
            rangesOfAdjacentVertices = rangesOfAdjacentVerticesFiles.fileToArr((size_t)i);
            numOfVertices = rangesOfAdjacentVertices.size() - 1;
            numOfEdges = sources.size();
            cout << "\nNumber Of Vertices: " << numOfVertices;
            cout << "\nNumber Of Edges: " << numOfEdges;
            vector<int> shortestPaths(numOfVertices), previousVertices(numOfVertices);
            BellmanFordParallelOmp(shortestPaths, previousVertices, numOfVertices, numOfEdges,
                startVertex, sources, rangesOfAdjacentVertices, weights, time, k);
            printf("\n------------------\nBellman Ford parallel omp execution time: %.4f seconds\n", time);
            cout << "Number Of Iterations in the main loop: " << k + 1 << "\n";
            if (printMode) {
                cout << "\nShortest paths from vertex " << startVertex << ": ";
                Array::printVector(shortestPaths, endPrintIndex + 1);
                printf("Predecessor vertex numbers for each vertex: ");
                Array::printVector(previousVertices, endPrintIndex + 1);
            }
        }
    }

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
    string arrStr = file[fileIndex].fileToBuf();
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

void BellmanFordParallelOmp(vector<int>& shortestPaths, vector<int>& previousVertices, int numOfVertices, int numOfEdges,
    int startVertex, vector<int> sources, vector<int> rangesOfAdjacentVertices, vector<int> weights, double& time, int& k)
{
    double ompStartTime, ompEndTime;
    vector<int> shortestTempPaths(numOfVertices);
    ompStartTime = omp_get_wtime();
    const int INF = 1e9;
#pragma omp parallel for
    for (int i = 0; i < numOfVertices; i++) {
        shortestPaths[i] = INF;
        shortestTempPaths[i] = INF;
        previousVertices[i] = -1;
    }
    shortestPaths[startVertex] = 0;
    shortestTempPaths[startVertex] = 0;
    int kLoop = 0;
    for (k = 0; k < numOfVertices - 1; k++)
    {
        kLoop = 0;
#pragma omp parallel for schedule(static) reduction (+:kLoop)
        for (int i = 0; i < numOfVertices; i++)
        {
            for (int j = rangesOfAdjacentVertices[i]; j < rangesOfAdjacentVertices[i + 1]; j++) {
                if (shortestTempPaths[i] > shortestPaths[sources[j]] + weights[j])
                {
                    shortestTempPaths[i] = shortestPaths[sources[j]] + weights[j];
                    previousVertices[i] = sources[j];
                    kLoop++;
                }
            }
        }
        if (kLoop == 0) break;
#pragma omp parallel for
        for (int i = 0; i < numOfVertices; i++) {
            shortestPaths[i] = shortestTempPaths[i];
        }
    }
    ompEndTime = omp_get_wtime();
    time = ompEndTime - ompStartTime;
}
