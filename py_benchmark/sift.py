import cv2
import numpy as np
import time
import psutil
import os

def benchmark_sift(image_path, n_runs=50):
    """Benchmark OpenCV's SIFT descriptor generation on a single image."""
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    print(f"Image: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} ({img.shape[0]*img.shape[1]} pixels)")
    print(f"Runs: {n_runs}")
    print("-" * 60)
    
    sift = cv2.SIFT_create()
    
    # Benchmark keypoint detection only
    detect_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        keypoints = sift.detect(img, None)
        end = time.perf_counter()
        detect_times.append(end - start)
    
    # Benchmark descriptor computation only (reuse keypoints)
    keypoints = sift.detect(img, None)
    compute_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        keypoints_out, descriptors = sift.compute(img, keypoints)
        end = time.perf_counter()
        compute_times.append(end - start)
    
    # Benchmark combined detect + compute
    combined_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        kp, desc = sift.detectAndCompute(img, None)
        end = time.perf_counter()
        combined_times.append(end - start)
    
    # Results
    detect_times = np.array(detect_times)
    compute_times = np.array(compute_times)
    combined_times = np.array(combined_times)
    
    print(f"Keypoints detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape}  (num_keypoints x 128)")
    print(f"Descriptor dtype: {descriptors.dtype}")
    print()
    
    def print_stats(name, times_ms):
        times_ms = times_ms * 1000  # convert to ms
        print(f"  {name}:")
        print(f"    Mean:   {np.mean(times_ms):8.3f} ms")
        print(f"    Median: {np.median(times_ms):8.3f} ms")
        print(f"    Std:    {np.std(times_ms):8.3f} ms")
        print(f"    Min:    {np.min(times_ms):8.3f} ms")
        print(f"    Max:    {np.max(times_ms):8.3f} ms")
        print()
    
    print("Timing")
    print_stats("Keypoint Detection", detect_times)
    print_stats("Descriptor Computation", compute_times)
    print_stats("Combined (detect+compute)", combined_times)
    
    # Throughput
    mean_combined_s = np.mean(combined_times)
    print(f"Throughput")
    print(f"  Images/sec (combined):      {1.0 / mean_combined_s:.2f}")
    print(f"  Keypoints/sec (compute):    {len(keypoints) / np.mean(compute_times):.0f}")
    print(f"  Megapixels/sec (combined):  {(img.shape[0]*img.shape[1]) / mean_combined_s / 1e6:.2f}")
    print()
    
    # Memory estimate
    desc_mem = descriptors.nbytes
    kp_mem_est = len(keypoints) * 48  # approximate per-keypoint overhead
    print(f"Memory")
    print(f"  Image memory:       {img.nbytes / 1024:.1f} KB")
    print(f"  Descriptor memory:  {desc_mem / 1024:.1f} KB")
    print(f"  Est. keypoint mem:  {kp_mem_est / 1024:.1f} KB")
    print(f"  Process RSS:        {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.1f} MB")
    
    return {
        "image_shape": img.shape,
        "num_keypoints": len(keypoints),
        "descriptor_shape": descriptors.shape,
        "detect_times": detect_times,
        "compute_times": compute_times,
        "combined_times": combined_times,
    }


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/Eiffel_Tower_1.jpg"
    
    # Warm-up run (not measured)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sift = cv2.SIFT_create()
        sift.detectAndCompute(img, None)
    
    results = benchmark_sift(image_path, n_runs=50)