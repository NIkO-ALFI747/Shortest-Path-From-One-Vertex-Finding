#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;
using namespace chrono;

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

void BellmanFordSerialDebug(vector <int>& shortestPaths, vector <int>& previousVertices, bool& negativeWeightCycle,
    int numOfVertices, int numOfEdges, int startVertex, vector<int> sources, vector<int> dests, vector<int> weights, int endPrintIndex, int& k);

void BellmanFordSerial(vector <int>& shortestPaths, vector <int>& previousVertices, bool& negativeWeightCycle,
    int numOfVertices, int numOfEdges, int startVertex, vector<int> sources, vector<int> dests, vector<int> weights, double& time, int& k);

int main()
{
    int vertexCountFactor = 3;
    size_t numOfVertices, numOfEdges;
    int startVertex = 0;
    int printMode = 0;
    int debugMode = 0;
    int endPrintIndex = 9;
    int numOfFiles = 4;
    bool negativeWeightCycle;
    double time;
    int k;
    FileVector edgeListFiles(numOfFiles, "EdgeList_");
    vector<vector<int>> edgeList;
    for (int i = 0; i < numOfFiles; i++) {
        edgeList = edgeListFiles.fileTo2DArr((size_t)i);
        numOfEdges = edgeList.size();
        numOfVertices = vertexCountFactor * pow(10, i);
        cout << "\nNumber Of Vertices: " << numOfVertices << "\nNumber Of Edges: " << numOfEdges << "\n";
        vector<int> shortestPaths(numOfVertices), previousVertices(numOfVertices);
        vector <int> sources(numOfEdges), dests(numOfEdges), weights(numOfEdges);
        for (int j = 0; j < numOfEdges; j++) {
            sources[j] = edgeList[j][0];
            dests[j] = edgeList[j][1];
            weights[j] = edgeList[j][2];
        }
        if (debugMode && printMode) {
            cout << endl;
            BellmanFordSerialDebug(shortestPaths, previousVertices, negativeWeightCycle,
                numOfVertices, numOfEdges, startVertex, sources, dests, weights, endPrintIndex, k);
        }
        else {
            BellmanFordSerial(shortestPaths, previousVertices, negativeWeightCycle,
                numOfVertices, numOfEdges, startVertex, sources, dests, weights, time, k);
            printf("------------------\nBellman Ford serial execution time: %.4f seconds\n", time);
        }
        cout << "Number Of Iterations in the main loop: " << k << "\n";
        if (printMode) {
            cout << "\nShortest paths from vertex " << startVertex << ": ";
            Array::printVector(shortestPaths, endPrintIndex + 1);
            printf("Predecessor vertex numbers for each vertex: ");
            Array::printVector(previousVertices, endPrintIndex + 1);
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

void BellmanFordSerialDebug(vector<int>& shortestPaths, vector<int>& previousVertices, bool& negativeWeightCycle,
    int numOfVertices, int numOfEdges, int startVertex, vector<int> sources, vector<int> dests, vector<int> weights, int endPrintIndex, int& k)
{
    const int INF = 1e9;
    for (int i = 0; i < numOfVertices; i++) {
        shortestPaths[i] = INF;
        previousVertices[i] = -1;
    }
    shortestPaths[startVertex] = 0;
    bool kLoop = true;
    for (k = 0; (k < numOfVertices - 1) && kLoop; k++)
    {
        kLoop = false;
        for (int i = 0; i < numOfEdges; i++)
        {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]])
            {
                shortestPaths[dests[i]] = shortestPaths[sources[i]] + weights[i];
                previousVertices[dests[i]] = sources[i];
                kLoop = true;
            }
        }
        cout << "k: " << k << ", shortest paths: ";
        Array::printVector(shortestPaths, endPrintIndex + 1);
    }
    negativeWeightCycle = false;
    if (k == numOfVertices - 1) {
        for (int i = 0; i < numOfEdges; i++)
        {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]])
            {
                negativeWeightCycle = true;
                break;
            }
        }
    }
}

void BellmanFordSerial(vector <int>& shortestPaths, vector <int>& previousVertices, bool& negativeWeightCycle,
    int numOfVertices, int numOfEdges, int startVertex, vector<int> sources, vector<int> dests, vector<int> weights, double& time, int& k) {
    auto startTime = high_resolution_clock::now();
    const int INF = 1e9;
    for (int i = 0; i < numOfVertices; i++) {
        shortestPaths[i] = INF;
        previousVertices[i] = -1;
    }
    shortestPaths[startVertex] = 0;
    bool kLoop = true;
    for (k = 0; (k < numOfVertices - 1) && kLoop; k++)
    {
        kLoop = false;
        for (int i = 0; i < numOfEdges; i++)
        {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]])
            {
                shortestPaths[dests[i]] = shortestPaths[sources[i]] + weights[i];
                previousVertices[dests[i]] = sources[i];
                kLoop = true;
            }
        }
    }
    auto endTime = high_resolution_clock::now();
    time = duration_cast<milliseconds>(endTime - startTime).count() / 1000.0;
    negativeWeightCycle = false;
    if (k == numOfVertices - 1) {
        for (int i = 0; i < numOfEdges; i++)
        {
            if (shortestPaths[sources[i]] + weights[i] < shortestPaths[dests[i]])
            {
                negativeWeightCycle = true;
                break;
            }
        }
    }
}