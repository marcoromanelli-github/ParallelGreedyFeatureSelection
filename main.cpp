#include <iostream>
#include "C_code/gfs.h"
#include <vector>

using namespace std;
using namespace gfs_manager_space;

int main() {
    int rows = 50;
    int cols = 60;
    vector<vector<int> > data;


    for (int i = 0; i < rows; ++i) {
        vector<int> v_;
        v_.reserve(cols);
        for (int i = 0; i < cols; ++i) {
            v_.push_back(rand() % 100);
        }
        data.push_back(v_);
    }

    vector<int> labels;
    for (int i = 0; i < rows; ++i) {
        labels.push_back(rand() % 10);
    }


    gfsManager a = gfsManager(data, labels);

    vector<int> res = a.greedyAlgorithm(5, 2, "shannon");

    return 0;
}
