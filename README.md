[![Documentation](https://img.shields.io/badge/Documentation-yes-blue)](https://img.shields.io/badge/Documentation-yes-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://img.shields.io/badge/License-MIT-yellow.svg)

# ParallelGreedyFeatureSelection

Parrallel C++ based version of [this](https://github.com/marcoromanelli-github/GreedyFeatureSelection). 

### Getting started
In order to try it out, 
download the repository, move into the relative folder and compile the files launching this command from CLI
```console
foo$bar g++ main.cpp C_code/gfs.cpp -o prog -lpthread
```
then launch the script via
```console
foo$bar ./prog
```
In particula, the line
```C++
vector<int> res = a.greedyAlgorithm(5, 2, "shannon");
```
means that we are calling the algorithm with Shannon entropy (alternatively we can call pass "renyi" for the Rényi min-entropy), asking for the 5 
most informative features and spawning 2 new async processes at a time. We have empirically experienced better perfonces when limitating this number instead 
of leaving it up to the system. The max value corresponds to the number of possible threads and is obtained by passing the argiment -1.

#### Todo
- [ ] Extend this to Python bindings
- [ ] Heavy testing
- [ ] Improve the interface and the documentation
