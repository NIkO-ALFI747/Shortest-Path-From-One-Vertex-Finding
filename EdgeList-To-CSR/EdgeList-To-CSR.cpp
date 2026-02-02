#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <omp.h>

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

int quickSortPartition(vector <int>& arr, vector <int>& arr2, vector <int>& arr3, int start, int end);
void tailRecursiveQuickSort(vector <int>& arr, vector <int>& arr2, vector <int>& arr3, int start, int end);

int main()
{
    int vertexCountFactor = 3;
    size_t numOfVertices, numOfEdges;
    int startVertex = 0;
    int printMode = 1;
    int endPrintIndex = 9;
    int numOfFiles = 4;
    int numOfThreads = 8;
    FileVector edgeListFiles(numOfFiles, "EdgeList_");
    vector<vector<int>> edgeList;
    omp_set_num_threads(numOfThreads);
    for (int i = 0; i < numOfFiles; i++) {
        edgeList = edgeListFiles.fileTo2DArr((size_t)i);
        numOfEdges = edgeList.size();
        numOfVertices = vertexCountFactor * pow(10, i);
        cout << "\nNumber Of Vertices: " << numOfVertices << "\nNumber Of Edges: " << numOfEdges << "\n";
        vector <int> sources(numOfEdges), dests(numOfEdges), weights(numOfEdges);
        vector <int> numOfEmergingEdges(numOfVertices + 1), rangesOfAdjacentVertices(numOfVertices + 1);
#pragma omp parallel for
        for (int j = 0; j < numOfEdges; j++) {
            sources[j] = edgeList[j][0];
            dests[j] = edgeList[j][1];
            weights[j] = edgeList[j][2];
        }
#pragma omp parallel for
        for (int j = 0; j < numOfVertices + 1; j++) {
            numOfEmergingEdges[j] = 0;
            rangesOfAdjacentVertices[j] = 0;
        }
        if (printMode) {
            cout << "\nDest vertices before sort: ";
            Array::printVector(dests, endPrintIndex + 1);
        }
        int startPosition = 0;
        int endPosition = numOfEdges - 1;
        tailRecursiveQuickSort(dests, sources, weights, startPosition, endPosition);
        for (int j = 0; j < numOfEdges; j++) {
            numOfEmergingEdges[dests[j]]++;
        }
#pragma omp parallel for
        for (int j = 0; j < numOfVertices + 1; j++) {
            for (int k = 0; k < j; k++)
                rangesOfAdjacentVertices[j] += numOfEmergingEdges[k];
        }
        for (int j = 1; j < numOfVertices; j++) {
            startPosition = rangesOfAdjacentVertices[j];
            endPosition = rangesOfAdjacentVertices[j + 1] - 1;
            if (startPosition != endPosition)
                tailRecursiveQuickSort(sources, dests, weights, startPosition, endPosition);
        }
        if (printMode) {
            cout << "Dest vertices after sort: ";
            Array::printVector(dests, endPrintIndex + 1);
            cout << "Sources: ";
            Array::printVector(sources, endPrintIndex + 1);
            cout << "Weights: ";
            Array::printVector(weights, endPrintIndex + 1);
            cout << "RangesOfAdjacentVertices: ";
            Array::printVector(rangesOfAdjacentVertices, endPrintIndex + 1);
        }
        string strEdges = "", strAdjacentVertices = "", strWeights = "";
        for (int j = 0; j < numOfEdges; j++) {
            strEdges += to_string(sources[j]) + " ";
            strWeights += to_string(weights[j]) + " ";
        }
        for (int j = 0; j < numOfVertices + 1; j++)
            strAdjacentVertices += to_string(rangesOfAdjacentVertices[j]) + " ";
        File file("Sources_" + to_string(i + 1) + ".txt");
        file.bufToFile(strEdges);
        file.setFileName("Weights_" + to_string(i + 1) + ".txt");
        file.bufToFile(strWeights);
        file.setFileName("RangesOfAdjacentVertices_" + to_string(i + 1) + ".txt");
        file.bufToFile(strAdjacentVertices);
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

int quickSortPartition(vector <int>& arr, vector <int>& arr2, vector <int>& arr3, int start, int end)
{
    int pivot = arr[end];
    int pIndex = start;
    for (int i = start; i < end; i++)
    {
        if (arr[i] <= pivot)
        {
            swap(arr[i], arr[pIndex]);
            swap(arr2[i], arr2[pIndex]);
            swap(arr3[i], arr3[pIndex]);
            pIndex++;
        }
    }
    swap(arr[pIndex], arr[end]);
    swap(arr2[pIndex], arr2[end]);
    swap(arr3[pIndex], arr3[end]);
    return pIndex;
}

void tailRecursiveQuickSort(vector <int>& arr, vector <int>& arr2, vector <int>& arr3, int start, int end)
{
    while (start < end)
    {
        int pivot = quickSortPartition(arr, arr2, arr3, start, end);
        if (pivot - start < end - pivot)
        {
            tailRecursiveQuickSort(arr, arr2, arr3, start, pivot - 1);
            start = pivot + 1;
        }
        else
        {
            tailRecursiveQuickSort(arr, arr2, arr3, pivot + 1, end);
            end = pivot - 1;
        }
    }
}