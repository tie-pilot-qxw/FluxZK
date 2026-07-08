#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

struct BenchArgs {
    std::vector<int> ks;
    int samples;
    int warmups;
};

inline int parse_positive_int(const std::string &value, const char *name) {
    char *end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (*value.c_str() == '\0' || *end != '\0' || parsed <= 0) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

inline int parse_nonnegative_int(const std::string &value, const char *name) {
    char *end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (*value.c_str() == '\0' || *end != '\0' || parsed < 0) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

inline std::string consume_option_value(int &i, int argc, char **argv, const std::string &arg, const char *name) {
    std::string prefix = std::string(name) + "=";
    if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
    }
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
    }
    return argv[++i];
}

inline std::vector<int> parse_k_list(const std::string &value) {
    std::vector<int> ks;
    size_t begin = 0;
    while (begin <= value.size()) {
        size_t end = value.find(',', begin);
        std::string token = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!token.empty()) {
            ks.push_back(parse_positive_int(token, "--ks"));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (ks.empty()) {
        throw std::runtime_error("empty --ks list");
    }
    return ks;
}

inline std::vector<int> make_k_range(int min_k, int max_k, int step) {
    if (min_k > max_k) {
        throw std::runtime_error("--min-k must be <= --max-k");
    }
    std::vector<int> ks;
    for (int k = min_k; k <= max_k; k += step) {
        ks.push_back(k);
    }
    return ks;
}

inline BenchArgs parse_bench_args(
    int argc,
    char **argv,
    int default_min_k,
    int default_max_k,
    int default_step,
    int default_samples,
    int default_warmups
) {
    int min_k = default_min_k;
    int max_k = default_max_k;
    int step = default_step;
    int samples = default_samples;
    int warmups = default_warmups;
    std::vector<int> ks;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--ks" || arg.rfind("--ks=", 0) == 0) {
            ks = parse_k_list(consume_option_value(i, argc, argv, arg, "--ks"));
        } else if (arg == "--min-k" || arg.rfind("--min-k=", 0) == 0) {
            min_k = parse_positive_int(consume_option_value(i, argc, argv, arg, "--min-k"), "--min-k");
        } else if (arg == "--max-k" || arg.rfind("--max-k=", 0) == 0) {
            max_k = parse_positive_int(consume_option_value(i, argc, argv, arg, "--max-k"), "--max-k");
        } else if (arg == "--step" || arg.rfind("--step=", 0) == 0) {
            step = parse_positive_int(consume_option_value(i, argc, argv, arg, "--step"), "--step");
        } else if (arg == "--samples" || arg.rfind("--samples=", 0) == 0) {
            samples = parse_positive_int(consume_option_value(i, argc, argv, arg, "--samples"), "--samples");
        } else if (arg == "--warmups" || arg.rfind("--warmups=", 0) == 0) {
            warmups = parse_nonnegative_int(consume_option_value(i, argc, argv, arg, "--warmups"), "--warmups");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (ks.empty()) {
        ks = make_k_range(min_k, max_k, step);
    }

    return BenchArgs{ks, samples, warmups};
}
