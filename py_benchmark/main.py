import cv2
import numpy as np
import time
import psutil
import os
import glob

def benchmark_sift(image_path, n_runs=50):
    """Benchmark OpenCV's SIFT descriptor generation on a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    sift = cv2.SIFT_create()

    detect_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        keypoints = sift.detect(img, None)
        end = time.perf_counter()
        detect_times.append(end - start)

    keypoints = sift.detect(img, None)
    compute_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        keypoints_out, descriptors = sift.compute(img, keypoints)
        end = time.perf_counter()
        compute_times.append(end - start)

    combined_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        kp, desc = sift.detectAndCompute(img, None)
        end = time.perf_counter()
        combined_times.append(end - start)

    return {
        "image_path": image_path,
        "image_shape": img.shape,
        "num_keypoints": len(keypoints),
        "descriptor_shape": descriptors.shape,
        "detect_times": np.array(detect_times),
        "compute_times": np.array(compute_times),
        "combined_times": np.array(combined_times),
        "img_nbytes": img.nbytes,
        "desc_nbytes": descriptors.nbytes,
    }

def print_stats(name, times_ms):
    times_ms = times_ms * 1000
    print(f"  {name}:")
    print(f"    Mean:   {np.mean(times_ms):8.3f} ms")
    print(f"    Median: {np.median(times_ms):8.3f} ms")
    print(f"    Std:    {np.std(times_ms):8.3f} ms")
    print(f"    Min:    {np.min(times_ms):8.3f} ms")
    print(f"    Max:    {np.max(times_ms):8.3f} ms")
    print()

def print_results(r):
    print(f"\nImage: {r['image_path']}")
    print(f"Image size: {r['image_shape'][1]}x{r['image_shape'][0]} ({r['image_shape'][0]*r['image_shape'][1]} pixels)")
    print(f"Keypoints detected: {r['num_keypoints']}")
    print(f"Descriptor shape: {r['descriptor_shape']}")
    print("-" * 60)
    print_stats("Keypoint Detection", r["detect_times"])
    print_stats("Descriptor Computation", r["compute_times"])
    print_stats("Combined (detect+compute)", r["combined_times"])

    mean_combined_s = np.mean(r["combined_times"])
    print(f"  Images/sec (combined):      {1.0 / mean_combined_s:.2f}")
    print(f"  Keypoints/sec (compute):    {r['num_keypoints'] / np.mean(r['compute_times']):.0f}")
    print(f"  Megapixels/sec (combined):  {(r['image_shape'][0]*r['image_shape'][1]) / mean_combined_s / 1e6:.2f}")

def print_aggregate(all_results):
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({len(all_results)} images)")
    print("=" * 60)

    all_detect = np.concatenate([r["detect_times"] for r in all_results])
    all_compute = np.concatenate([r["compute_times"] for r in all_results])
    all_combined = np.concatenate([r["combined_times"] for r in all_results])

    print_stats("Keypoint Detection", all_detect)
    print_stats("Descriptor Computation", all_compute)
    print_stats("Combined (detect+compute)", all_combined)

    total_kp = sum(r["num_keypoints"] for r in all_results)
    total_pixels = sum(r["image_shape"][0] * r["image_shape"][1] for r in all_results)
    print(f"  Total keypoints across dataset: {total_kp}")
    print(f"  Avg keypoints per image:        {total_kp / len(all_results):.0f}")
    print(f"  Avg compute time per image:     {np.mean(all_compute)*1000:.3f} ms")

if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    image_paths = sorted(
        glob.glob(os.path.join(dataset_dir, "*"))
    )
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

    if not image_paths:
        print(f"No images found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {dataset_dir}")
    print(f"Runs per image: {n_runs}")

    all_results = []
    for path in image_paths:
        try:
            r = benchmark_sift(path, n_runs=n_runs)
            print_results(r)
            all_results.append(r)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if all_results:
        print_aggregate(all_results)
