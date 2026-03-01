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

using namespace cv;
using namespace std;


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

int benchmark_sift(const string& img_path, int n_runs = 50) {
    // Load image in grayscale
    Mat img = imread(img_path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Could not load image: " << img_path << endl;
        return -1;
    }

    cout << "Image: " << img_path << endl;
    cout << "Image size: " << img.cols << "x" << img.rows
         << " (" << img.rows * img.cols << " pixels)" << endl;
    cout << "Runs: " << n_runs << endl;
    cout << string(60, '-') << endl;

    Ptr<SIFT> sift = SIFT::create();

    // Benchmark keypoint detection only
    vector<double> detect_times_ms;
    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> keypoints;

        auto start = chrono::high_resolution_clock::now();
        sift->detect(img, keypoints);
        auto end   = chrono::high_resolution_clock::now();

        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        detect_times_ms.push_back(duration_ms);
    }

    // Benchmark descriptor computation only (reuse keypoints)
    vector<KeyPoint> keypoints;
    sift->detect(img, keypoints);

    vector<double> compute_times_ms;

    Mat descriptors;  // will hold the last descriptors
    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> keypoints_out = keypoints;  // same keypoints each run

        auto start = chrono::high_resolution_clock::now();
        sift->compute(img, keypoints_out, descriptors);
        auto end   = chrono::high_resolution_clock::now();

        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        compute_times_ms.push_back(duration_ms);
    }

    // Benchmark combined detect + compute
    vector<double> combined_times_ms;

    for (int i = 0; i < n_runs; ++i) {
        vector<KeyPoint> kp;
        Mat desc;

        auto start = chrono::high_resolution_clock::now();
        sift->detectAndCompute(img, noArray(), kp, desc);
        auto end   = chrono::high_resolution_clock::now();

        auto duration_ms = chrono::duration_cast<chrono::duration<double, milli>>(end - start).count();
        combined_times_ms.push_back(duration_ms);
    }

    // Results (shapes, etc.)
    cout << "Keypoints detected: " << keypoints.size() << endl;
    cout << "Descriptor shape: " << descriptors.rows << " x " << descriptors.cols
         << "  (num_keypoints x 128)" << endl;
    cout << "Descriptor dtype: CV_32F" << endl;
    cout << endl;

    // Timing stats
    cout << "Timing" << endl;
    print_stats("Keypoint Detection", detect_times_ms);
    print_stats("Descriptor Computation", compute_times_ms);
    print_stats("Combined (detect+compute)", combined_times_ms);

    // Throughput
    double mean_combined_ms = mean_ms(combined_times_ms);
    double mean_combined_s  = mean_combined_ms / 1000.0;

    double mean_compute_ms = mean_ms(compute_times_ms);
    double mean_compute_s  = mean_compute_ms / 1000.0;

    cout << "Throughput" << endl;
    cout << "  Images/sec (combined):      "
         << fixed << setprecision(2)
         << (mean_combined_s > 0.0 ? 1.0 / mean_combined_s : 0.0) << endl;

    cout << "  Keypoints/sec (compute):    "
         << fixed << setprecision(0)
         << (mean_compute_s > 0.0
             ? static_cast<double>(keypoints.size()) / mean_compute_s
             : 0.0)
         << endl;

    double megapixels = (img.rows * img.cols) / 1e6;
    cout << "  Megapixels/sec (combined):  "
         << fixed << setprecision(2)
         << (mean_combined_s > 0.0 ? megapixels / mean_combined_s : 0.0)
         << endl;
    cout << endl;

    // Memory (no process RSS, but same other fields as Python)
    size_t img_bytes  = img.total() * img.elemSize();
    size_t desc_bytes = descriptors.total() * descriptors.elemSize();
    size_t kp_mem_est = keypoints.size() * 48;  // same rough estimate

    cout << "Memory" << endl;
    cout << "  Image memory:       " << img_bytes  / 1024.0 << " KB" << endl;
    cout << "  Descriptor memory:  " << desc_bytes / 1024.0 << " KB" << endl;
    cout << "  Est. keypoint mem:  " << kp_mem_est / 1024.0 << " KB" << endl;
    cout << endl;

    return 0;
}

int main() {
    string img_path = "sample/Eiffel_Tower_1.jpg";
    benchmark_sift(img_path, 1);
    return 0;
}
