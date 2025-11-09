#pragma once

#include <fcntl.h>
// #include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

const size_t K = 10;  // top-k knns
const size_t M = 32;  // neighbor's number, should be times of 16
const size_t EF_CONSTRUCTION =
    120;  // maximum number of candidate neighbors considered during index construction.
const size_t EF_SEARCH      = 64;  // maximum number of candidates retained during the search phase.

const size_t SUBVECTOR_NUM  = 32;     // PQ subvector num, should be times of 16 when using Flash
const size_t CLUSTER_NUM    = 16;     // cluster numbers of each subvector
const size_t MAX_ITERATIONS = 12;     // k-means iteration times
const size_t SAMPLE_NUM     = 10000;  // sample number in generating codebooks

const size_t PRINCIPAL_DIM  = 96;  // Rest dimiensions after running PCA

#define RERANK
