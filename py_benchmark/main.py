import cv2
import numpy as np
import time
import psutil
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def benchmark_sift_on_array(img, n_runs=50):
    """Benchmark OpenCV's SIFT descriptor generation on a grayscale ndarray."""
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
        "image_shape": img.shape,
        "num_keypoints": len(keypoints),
        "descriptor_shape": descriptors.shape if descriptors is not None else (0, 128),
        "detect_times": np.array(detect_times),
        "compute_times": np.array(compute_times),
        "combined_times": np.array(combined_times),
        "img_nbytes": img.nbytes,
        "desc_nbytes": descriptors.nbytes if descriptors is not None else 0,
    }


def benchmark_sift(image_path, n_runs=50):
    """Benchmark OpenCV's SIFT descriptor generation on a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    result = benchmark_sift_on_array(img, n_runs=n_runs)
    result["image_path"] = image_path
    return result


def benchmark_across_scales(image_path, scales=(0.10, 0.25, 0.50, 0.75, 1.00), n_runs=50):
    """Run benchmark at multiple resolution scales of a single image."""
    img_full = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h_full, w_full = img_full.shape
    scale_results = []
    for scale in scales:
        new_w = max(1, int(w_full * scale))
        new_h = max(1, int(h_full * scale))
        img_scaled = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Scale {scale*100:.0f}%: {new_w}x{new_h} ({new_w*new_h} pixels)")
        r = benchmark_sift_on_array(img_scaled, n_runs=n_runs)
        r["scale"] = scale
        r["total_pixels"] = new_w * new_h
        r["width"] = new_w
        r["height"] = new_h
        scale_results.append(r)

    return scale_results


def plot_time_vs_size(scale_results, output_path):
    """Line plot: mean timing for each phase vs. image resolution."""
    pixels = [r["total_pixels"] for r in scale_results]
    detect_ms  = [np.mean(r["detect_times"])  * 1000 for r in scale_results]
    compute_ms = [np.mean(r["compute_times"]) * 1000 for r in scale_results]
    combined_ms = [np.mean(r["combined_times"]) * 1000 for r in scale_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pixels, detect_ms,   "o-", label="Detection",   color="#1f77b4")
    ax.plot(pixels, compute_ms,  "s-", label="Descriptor",  color="#ff7f0e")
    ax.plot(pixels, combined_ms, "^-", label="Combined",    color="#2ca02c")

    ax.set_xlabel("Image Size (pixels)", fontsize=12)
    ax.set_ylabel("Mean Time (ms)", fontsize=12)
    ax.set_title("SIFT Execution Time vs. Image Size", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate resolution labels on x-axis
    labels = [f"{r['width']}×{r['height']}" for r in scale_results]
    ax.set_xticks(pixels)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_keypoints_vs_size(scale_results, output_path):
    """Line plot: number of detected keypoints vs. image resolution."""
    pixels = [r["total_pixels"] for r in scale_results]
    keypoints = [r["num_keypoints"] for r in scale_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pixels, keypoints, "o-", color="#9467bd", linewidth=2)

    ax.set_xlabel("Image Size (pixels)", fontsize=12)
    ax.set_ylabel("Keypoints Detected", fontsize=12)
    ax.set_title("SIFT Keypoints Detected vs. Image Size", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    labels = [f"{r['width']}×{r['height']}" for r in scale_results]
    ax.set_xticks(pixels)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_timing_distribution(results, output_path):
    """Box plot: timing distribution (all runs) for detect / compute / combined."""
    # Collect times in ms across all provided results
    detect_ms  = np.concatenate([r["detect_times"]  for r in results]) * 1000
    compute_ms = np.concatenate([r["compute_times"] for r in results]) * 1000
    combined_ms = np.concatenate([r["combined_times"] for r in results]) * 1000

    data = [detect_ms, compute_ms, combined_ms]
    labels = ["Detection", "Descriptor\nComputation", "Combined"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("SIFT Timing Distribution (per phase)", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


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

        # --- Plotting ---
        plots_dir = os.path.join(dataset_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Scale analysis uses the first image in the dataset
        ref_image = image_paths[0]
        print(f"\nGenerating scale plots using: {ref_image}")
        scale_results = benchmark_across_scales(ref_image, n_runs=n_runs)

        plot_time_vs_size(
            scale_results,
            os.path.join(plots_dir, "time_vs_size.png"),
        )
        plot_keypoints_vs_size(
            scale_results,
            os.path.join(plots_dir, "keypoints_vs_size.png"),
        )
        plot_timing_distribution(
            all_results,
            os.path.join(plots_dir, "timing_distribution.png"),
        )
