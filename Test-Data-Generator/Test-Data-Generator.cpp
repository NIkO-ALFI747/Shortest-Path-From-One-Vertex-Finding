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

int* getIndicesInRandomOrder(int size, int excludedIndex);

void generateRandomConnectedGraph(int minWeight, int maxWeight, int startVertex, int numOfVertices,
    vector<int>& sources, vector<int>& dests, vector<int>& weights);

void BellmanFordSerial(vector <int>& shortestPaths, vector <int>& previousVertices, bool& negativeWeightCycle,
    int numOfVertices, int numOfEdges, int startVertex, vector<int> sources, vector<int> dests, vector<int> weights, double& time, int& k);

int main()
{
    int vertexCountFactor = 3;
    size_t numOfVertices, numOfEdges;
    int startVertex = 0;
    int printMode = 1;
    int endPrintIndex = 9;
    int numOfFiles = 4;
    srand(time(NULL));
    bool negativeWeightCycle;
    double time = 0;
    int k;
    int j = 0;
    for (int i = 0; (i < numOfFiles) && (j < 10);) {
        numOfVertices = vertexCountFactor * pow(10, i);
        vector<int> shortestPaths(numOfVertices), previousVertices(numOfVertices);
        vector<int> sources, dests, weights;
        generateRandomConnectedGraph(-3, 10, startVertex, numOfVertices, sources, dests, weights);
        numOfEdges = sources.size();
        cout << "\nSearch for negative weight cycle... " << j << "\n";
        BellmanFordSerial(shortestPaths, previousVertices, negativeWeightCycle,
            numOfVertices, numOfEdges, startVertex, sources, dests, weights, time, k);
        printf("------------------\nBellman Ford serial execution time: %.4f seconds\n", time);
        j++;
        if (!negativeWeightCycle) {
            cout << "Graph has been generated!\nFile: " << i << "\n";
            cout << "Number Of Vertices: " << numOfVertices << "\n";
            if (printMode) {
                for (int k = 0; k < numOfEdges && k <= endPrintIndex; k++) {
                    cout << k << " -> ";
                    cout << sources[k] << " ";
                    cout << dests[k] << " ";
                    cout << weights[k] << " ";
                    cout << "\n";
                }
            }
            string strEdgeList = "";
            for (int j = 0; j < numOfEdges; j++) {
                strEdgeList += to_string(sources[j]) + " " +
                    to_string(dests[j]) + " " +
                    to_string(weights[j]) + " \n";
            }
            File file("EdgeList_" + to_string(i + 1) + ".txt");
            file.bufToFile(strEdgeList);
            j = 0;
            i++;
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

int* getIndicesInRandomOrder(int size, int excludedIndex)
{
    int* indices = new int[size - 1];
    if (excludedIndex < size - 1) {
        for (int i = 0; i < excludedIndex; i++)
            indices[i] = i;
        for (int i = excludedIndex; i < size - 1; i++)
            indices[i] = i + 1;
    }
    for (int i = size - 2; i > 0; --i)
    {
        int j = rand() % (i + 1);
        swap(indices[i], indices[j]);
    }
    return indices;
}

void generateRandomConnectedGraph(int minWeight, int maxWeight, int startVertex, int numOfVertices,
    vector<int>& sources, vector<int>& dests, vector<int>& weights) {
    int excludedIndex = startVertex;
    int* childVertices = getIndicesInRandomOrder(numOfVertices, excludedIndex);
    vector <int> parentVertices;
    int parentVertex = startVertex;
    int parentVertexIndex = 0;
    parentVertices.push_back(parentVertex);
    int additionalParentVertex;
    for (int i = 0; i < numOfVertices - 1; i++) {
        sources.push_back(parentVertex);
        dests.push_back(childVertices[i]);
        weights.push_back(rand() % (maxWeight - minWeight + 1) + minWeight);
        additionalParentVertex = rand() % (numOfVertices - 2);
        for (int j = 0; j < 2; j++)
            if ((additionalParentVertex == parentVertex) || (additionalParentVertex == childVertices[i])) additionalParentVertex += 1;
        if (i % 2 == 1) {
            sources.push_back(additionalParentVertex);
            dests.push_back(childVertices[i]);
            weights.push_back(rand() % (maxWeight - minWeight + 1) + minWeight);
        }
        parentVertexIndex++;
        parentVertices.push_back(childVertices[i]);
        parentVertex = parentVertices[rand() % (parentVertexIndex + 1)];
    }
    delete[] childVertices;
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
