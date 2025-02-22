#pragma once

#include "core.h"

int64_t time_cost(const std::chrono::system_clock::time_point &st, const std::chrono::system_clock::time_point &en) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
}

void progressBar(int current, int total, std::chrono::steady_clock::time_point startTime, int barWidth = 50) {
    using namespace std::chrono;
    
    float progress = (float)current / total;
    int pos = barWidth * progress;

    // Calculate elapsed time
    auto now = steady_clock::now();
    auto elapsed = duration_cast<seconds>(now - startTime);

    // Format elapsed time as HH:MM:SS
    int hours = elapsed.count() / 3600;
    int minutes = (elapsed.count() % 3600) / 60;
    int seconds = elapsed.count() % 60;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
              << std::setw(3) << std::setfill(' ') << int(progress * 100.0) << "% "
              << std::setw(2) << std::setfill('0') << hours << ":"
              << std::setw(2) << std::setfill('0') << minutes << ":"
              << std::setw(2) << std::setfill('0') << seconds << "\r\n"[current == total];
    std::cout.flush();
}

// Read .fvecs and .ivecs
template <typename T>
void ReadData(const std::string& file_path,
              std::vector<std::vector<T>>& results, uint32_t &num, uint32_t &dim) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (uint32_t)(fsize / (dim + 1) / 4);
    results.resize(num);
    for (uint32_t i = 0; i < num; ++i) {
        results[i].resize(dim);
    }

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);
        in.read((char *)results[i].data(), dim * 4);
    }
    in.close();

    std::cout << "num: " << num << std::endl;
    std::cout << "dim: " << dim << std::endl;
}

// Write .fvecs and .ivecs
template <typename T>
void WriteData(const std::string& file_path,
               std::vector<std::vector<T>>& results) {
    std::filesystem::path fsPath(file_path);
    fsPath.remove_filename();
    std::filesystem::create_directories(fsPath);
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }

    uint32_t num = results.size();
    for (uint32_t i = 0; i < num; ++i) {
        uint32_t dim = results[i].size();
        out.write(reinterpret_cast<const char*>(&dim), 4);
        out.write(reinterpret_cast<const char*>(results[i].data()), dim * 4);
    }
    out.close();
}

static int tmp = 0;
template <typename T>
void WriteTemp(std::vector<std::vector<T>>& results,
               const std::string& file_path="../statistics/temp.txt") {
    std::ofstream out(file_path, std::ios::app);

    for (int i = 0; i < results.size(); ++i) {
        for (int j = 0; j < results[i].size(); ++j) {
            out << std::setw(8) << results[i][j] << " ";
        }
        out << "\n";
    }
    out << "\n";
    out.close();
}
