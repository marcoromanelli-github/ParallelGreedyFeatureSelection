//
// Created by Marco Romanelli on 11/4/20.
//

#include "gfs.h"

using namespace gfs_manager_space;

gfsManager::gfsManager(vector<vector<int> > dataset, vector<int> labels_input) {
    for (int i = 0; i < labels_input.size(); ++i) {
        this->labels.push_back(to_string(labels_input.at(i)));
    }

    this->labels_set = vectorToSet(this->labels);

    if (!isMatrix(dataset)) {
        throw std::invalid_argument("no samples available or columns number mismatch");
    } else {
        this->rows = dataset.size();
        int cols = dataset[0].size();
        for (int col_idx = 0; col_idx < cols; ++col_idx) {
            vector<string> column;
            column.reserve(cols);
            for (int i = 0; i < this->rows; ++i) {
                column.push_back(to_string(dataset.at(i).at(col_idx)));
            }
            this->F.insert(make_pair(col_idx, column));
        }
    }
}

gfsManager::~gfsManager() {
}

vector<int> gfsManager::greedyAlgorithm(int feat_card, int njobs, string strategy) {
    unsigned int nasynch = getNasynch(njobs);

    if (feat_card < 1 || feat_card >= this->F.size()) {
        throw std::invalid_argument("required number of feature to select must be an integer greater than 0 and "
                                    "less than the total number of available features");
    }
    for (int step = 0; step < feat_card; ++step) {   //  feature selection step: step; this is sequential
        gfsPickNextFeature(nasynch, strategy, step);
    }
    return this->S_index;
}

void gfsManager::gfsPickNextFeature(int nasynch, string strategy, int step) {
    vector<pair<int, float> > res;
    res.reserve(this->F.size());

    int out_steps = this->F.size() % nasynch;

    vector<future<pair<int, float> > > futures;

    map<int, vector<string> >::iterator it_tmp_0 = this->F.begin();
    advance(it_tmp_0, out_steps);

    for (map<int, vector<string> >::iterator it = this->F.begin(); it != it_tmp_0; it++) {
        futures.emplace_back(std::async(std::launch::async, computeEntropy, std::cref(it->first), std::cref(it->second),
                                        std::cref(this->S), this->labels, this->labels_set, strategy));
    }

    for (auto &&f: futures) {
        res.push_back(f.get());
    }

    futures.clear();
    for (map<int, vector<string> >::iterator it = it_tmp_0; it != this->F.end(); advance(it, nasynch)) {
        futures.emplace_back(
                std::async(std::launch::async, computeEntropy, std::cref(it->first), std::cref(it->second),
                           std::cref(this->S), this->labels, this->labels_set, strategy));
    }

    for (auto &&f: futures) {
        res.push_back(f.get());
    }

//    vector<future<pair<int, float> > > futures;
//    for (map<int, vector<string> >::iterator it = this->F.begin(); it != this->F.end(); ++it) {
//        futures.emplace_back(
//                std::async(std::launch::async, computeEntropy, std::cref(it->first), std::cref(it->second),
//                           std::cref(this->S), this->labels, this->labels_set, strategy));
//    }
//
//    for (auto &&f: futures) {
//        res.push_back(f.get());
//    }

    int res_final = findMin(res);
    this->S_index.push_back(res_final);
    this->S = newFeature(this->S, F[res_final]);
    this->F.erase(res_final);
    cout << "Step " << step << ": selected feature ---> " << res_final << endl;
}

int findMin(vector<pair<int, float> > res) {
    float min = numeric_limits<float>::infinity();
    int id;
    for (int i = 0; i < res.size(); ++i) {
        int k = (res[i].first);
        float v = (res[i].second);
        if (v < min) {
            min = v;
            id = k;
        }
    }
    return id;
}

unsigned int getNasynch(int njobs) {
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (njobs == -1) {
        return numThreads;
    } else if (njobs < -1) {
        return 1;
    } else if (1 <= njobs <= numThreads) {
        return (unsigned int) njobs;
    } else {
        return numThreads;
    }
}

pair<int, float> computeEntropy(int feature_id,
                                const vector<string> &feature,
                                const vector<string> &S,
                                const vector<string> &labels,
                                const set<string> &labels_set,
                                const string &strategy) {
    vector<string> S_t = newFeature(S, feature);
    pair<int, float> res_map;
    float res;
    if (strategy == "renyi") {
        res = renyiMinEntropy(S_t, labels, vectorToSet(S_t), labels_set);
    } else if (strategy == "shannon") {
        res = shannonEntropy(S_t, labels, vectorToSet(S_t), labels_set);
    } else {
        throw std::invalid_argument("required strategy not known");
    }
    res_map.first = feature_id;
    res_map.second = res;
    return res_map;
}

vector<string> newFeature(vector<string> S_base, vector<string> feature_array) {
    vector<string> new_feature;
    int feature_array_size = feature_array.size();
    int S_base_size = S_base.size();
    new_feature.reserve(feature_array_size);

    if (S_base.empty()) {
        return feature_array;
    } else {
        if ((S_base_size != feature_array_size) || (feature_array.empty())) {
            throw std::invalid_argument("arrays' dimensions mismatch");
        } else {
            for (int i = 0; i < S_base_size; ++i) {
                new_feature.push_back(S_base.at(i) + "_" + feature_array.at(i));
            }
        }
        return new_feature;
    }
}

float shannonEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set) {
    map<string, float> P_XjointY_map = computeJointProb(X, Y);
    map<string, float> P_Y_map = computeProb(Y);
//    printMap(P_XjointY_map);
//    cout << "///";
//    printMap(P_Y_map);
//    cout << "///";
    float sum_ext = 0;
    for (string x_el : X_set) {
        string x = x_el;

        float sum_int = 0;

        for (string y_el : Y_set) {
            string y = y_el;
            float p_y = P_Y_map[y];
            string string_tmp = x + "_" + y;
            if (isKey(P_XjointY_map, string_tmp)) {
                float p_xjointy = P_XjointY_map[string_tmp];
//                cout << string_tmp << "  p_xjointy  " << p_xjointy << " p_y  " << p_y << endl;
                sum_int += p_xjointy * log2(p_xjointy / p_y);
            }
        }

        sum_ext += sum_int;
    }
//    cout << -sum_ext << endl << endl;
    return -sum_ext;
}

float renyiMinEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set) {
    map<string, float> P_XjointY_map = computeJointProb(X, Y);
    float sum_ext = 0;
    for (string y_el : Y_set) {
        string y = y_el;

        float max_int = -numeric_limits<float>::infinity();;

        for (string x_el : X_set) {
            string x = x_el;
            string string_tmp = x + "_" + y;
            if (isKey(P_XjointY_map, string_tmp)) {
                float p_xjointy = P_XjointY_map[string_tmp];
                if (p_xjointy > max_int) {
                    max_int = p_xjointy;
                }
            }
        }
        sum_ext += max_int;
    }
    return -log2(sum_ext);
}

map<string, float> computeProb(vector<string> vec) {
    float number_of_samples = vec.size();

    map<string, int> joint_freq;

    for (int i = 0; i < vec.size(); ++i) {
        string current_element = vec.at(i);
        if (joint_freq.find(current_element) == joint_freq.end()) {
            joint_freq.insert(make_pair(current_element, 1));
        } else {
            joint_freq[current_element] += 1;
        }
    }

    map<string, float> joint_prob;
    map<string, int>::iterator it;
    for (it = joint_freq.begin(); it != joint_freq.end(); it++) {
        joint_prob[it->first] = it->second / number_of_samples;
    }

    return joint_prob;
}

map<string, float> computeJointProb(vector<string> vec_0, vector<string> vec_1) {
    vector<string> vec_tmp = newFeature(vec_0, vec_1);
    return computeProb(vec_tmp);
}
