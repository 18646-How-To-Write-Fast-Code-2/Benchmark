#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

auto mean_ms = [](const vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
};

static void print_stats(const string& name, const vector<double>& times_ms) {
    if (times_ms.empty()) return;
    vector<double> t = times_ms;
    sort(t.begin(), t.end());
    double sum = accumulate(t.begin(), t.end(), 0.0);
    double mean = sum / t.size();
    double median;
    size_t n = t.size();
    if (n % 2 == 0) {
        median = 0.5 * (t[n / 2 - 1] + t[n / 2]);
    } else {
        median = t[n / 2];
    }
    double sq_sum = 0.0;
    for (double v : t) {
        double d = v - mean;
        sq_sum += d * d;
    }
    double stddev = std::sqrt(sq_sum / n);
    double minv = t.front();
    double maxv = t.back();
    cout << "  " << name << ":" << endl;
    cout << "    Mean:   " << setw(8) << fixed << setprecision(3) << mean << " ms" << endl;
    cout << "    Median: " << setw(8) << fixed << setprecision(3) << median << " ms" << endl;
    cout << "    Std:    " << setw(8) << fixed << setprecision(3) << stddev << " ms" << endl;
    cout << "    Min:    " << setw(8) << fixed << setprecision(3) << minv << " ms" << endl;
    cout << "    Max:    " << setw(8) << fixed << setprecision(3) << maxv << " ms" << endl;
    cout << endl;
}

struct BenchmarkResult {
    string image_path;
    int rows, cols;
    size_t num_keypoints;
    int desc_rows, desc_cols;
    vector<double> detect_times_ms;
    vector<double> compute_times_ms;
    vector<double> combined_times_ms;
    size_t img_bytes;
    size_t desc_bytes;
};

BenchmarkResult benchmark_sift(const string& img_path, int n_runs = 50) {
    BenchmarkResult result;
    result.image_path = img_path;

    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Could not load image: " << img_path << endl;
        result.num_keypoints = 0;
        return result;
    }

    result.rows = img.rows;
    result.cols = img.cols;
    result.img_bytes = img.total() * img.elemSize();

    Ptr<SIFT> sift = SIFT::create();

    // Benchmark keypoint detection only
    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> keypoints;
        auto start = chrono::high_resolution_clock::now();
        sift->detect(img, keypoints);
        auto end   = chrono::high_resolution_clock::now();
        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        result.detect_times_ms.push_back(duration_ms);
    }

    // Benchmark descriptor computation only (reuse keypoints)
    vector<KeyPoint> keypoints;
    sift->detect(img, keypoints);
    Mat descriptors;
    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> keypoints_out = keypoints;
        auto start = chrono::high_resolution_clock::now();
        sift->compute(img, keypoints_out, descriptors);
        auto end   = chrono::high_resolution_clock::now();
        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        result.compute_times_ms.push_back(duration_ms);
    }

    // Benchmark combined detect + compute
    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> kp;
        Mat desc;
        auto start = chrono::high_resolution_clock::now();
        sift->detectAndCompute(img, noArray(), kp, desc);
        auto end   = chrono::high_resolution_clock::now();
        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        result.combined_times_ms.push_back(duration_ms);
    }

    result.num_keypoints = keypoints.size();
    result.desc_rows = descriptors.rows;
    result.desc_cols = descriptors.cols;
    result.desc_bytes = descriptors.total() * descriptors.elemSize();

    return result;
}

void print_results(const BenchmarkResult& r) {
    cout << "\nImage: " << r.image_path << endl;
    cout << "Image size: " << r.cols << "x" << r.rows
         << " (" << r.rows * r.cols << " pixels)" << endl;
    cout << "Keypoints detected: " << r.num_keypoints << endl;
    cout << "Descriptor shape: " << r.desc_rows << " x " << r.desc_cols
         << "  (num_keypoints x 128)" << endl;
    cout << string(60, '-') << endl;

    print_stats("Keypoint Detection", r.detect_times_ms);
    print_stats("Descriptor Computation", r.compute_times_ms);
    print_stats("Combined (detect+compute)", r.combined_times_ms);

    double mean_combined_s = mean_ms(r.combined_times_ms) / 1000.0;
    double mean_compute_s  = mean_ms(r.compute_times_ms) / 1000.0;

    cout << "  Images/sec (combined):      "
         << fixed << setprecision(2)
         << (mean_combined_s > 0.0 ? 1.0 / mean_combined_s : 0.0) << endl;
    cout << "  Keypoints/sec (compute):    "
         << fixed << setprecision(0)
         << (mean_compute_s > 0.0 ? static_cast<double>(r.num_keypoints) / mean_compute_s : 0.0) << endl;
    double megapixels = (r.rows * r.cols) / 1e6;
    cout << "  Megapixels/sec (combined):  "
         << fixed << setprecision(2)
         << (mean_combined_s > 0.0 ? megapixels / mean_combined_s : 0.0) << endl;
}

void print_aggregate(const vector<BenchmarkResult>& all_results) {
    cout << "\n" << string(60, '=') << endl;
    cout << "AGGREGATE RESULTS (" << all_results.size() << " images)" << endl;
    cout << string(60, '=') << endl;

    vector<double> all_detect, all_compute, all_combined;
    size_t total_kp = 0;
    long long total_pixels = 0;

    for (const auto& r : all_results) {
        all_detect.insert(all_detect.end(), r.detect_times_ms.begin(), r.detect_times_ms.end());
        all_compute.insert(all_compute.end(), r.compute_times_ms.begin(), r.compute_times_ms.end());
        all_combined.insert(all_combined.end(), r.combined_times_ms.begin(), r.combined_times_ms.end());
        total_kp += r.num_keypoints;
        total_pixels += (long long)r.rows * r.cols;
    }

    print_stats("Keypoint Detection", all_detect);
    print_stats("Descriptor Computation", all_compute);
    print_stats("Combined (detect+compute)", all_combined);

    cout << "  Total keypoints across dataset: " << total_kp << endl;
    cout << "  Avg keypoints per image:        "
         << fixed << setprecision(0) << static_cast<double>(total_kp) / all_results.size() << endl;
    cout << "  Avg compute time per image:     "
         << fixed << setprecision(3) << mean_ms(all_compute) << " ms" << endl;
}

int main(int argc, char** argv) {
    string dataset_dir = (argc > 1) ? argv[1] : "dataset";
    int n_runs = (argc > 2) ? stoi(argv[2]) : 50;

    // Collect image paths
    vector<string> image_paths;
    vector<string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(dataset_dir)) {
        if (!entry.is_regular_file()) continue;
        string ext = entry.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        for (const auto& valid_ext : extensions) {
            if (ext == valid_ext) {
                image_paths.push_back(entry.path().string());
                break;
            }
        }
    }
    sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        cerr << "No images found in " << dataset_dir << endl;
        return 1;
    }

    cout << "Found " << image_paths.size() << " images in " << dataset_dir << endl;
    cout << "Runs per image: " << n_runs << endl;

    vector<BenchmarkResult> all_results;
    for (const auto& path : image_paths) {
        BenchmarkResult r = benchmark_sift(path, n_runs);
        if (r.num_keypoints > 0) {
            print_results(r);
            all_results.push_back(r);
        }
    }

    if (!all_results.empty()) {
        print_aggregate(all_results);
    }

    return 0;
}
