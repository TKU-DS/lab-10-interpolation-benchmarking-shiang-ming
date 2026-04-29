import cv2
import numpy as np
import time

# =================================================================
# Course: Data Engineering (CSIE, Tamkang University)
# Lab 10: Spatial Manipulation & Interpolation Benchmarking
# =================================================================

def generate_4k_test_pattern():
    print("[*] Generating 4K test pattern in memory...")
    img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    
    for i in range(0, 3840, 40):
        cv2.line(img, (i, 0), (i, 2160), (100, 100, 100), 2)
    for i in range(0, 2160, 40):
        cv2.line(img, (0, i), (3840, i), (100, 100, 100), 2)
        
    cv2.circle(img, (1920, 1080), 800, (0, 0, 255), 15)
    cv2.circle(img, (1920, 1080), 400, (0, 255, 255), 10)
    cv2.putText(img, "EDGE AI BENCHMARK", (350, 1100), 
                cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 25)
    return img

def benchmark_interpolation(img_4k, target_size=(224, 224), iterations=100):
    print(f"[*] Starting benchmark: Scaling 4K to {target_size} over {iterations} iterations...")
    print("[!] Note: Running on GitHub VM. Focus on RELATIVE GAP.\n")
    
    results = {}
    visuals = {}

    methods = [
        ("INTER_NEAREST", cv2.INTER_NEAREST),
        ("INTER_LINEAR",  cv2.INTER_LINEAR),
        ("INTER_CUBIC",   cv2.INTER_CUBIC)
    ]

    for name, flag in methods:
        times_ms = []
        resized_img = None
        
        # warm-up
        cv2.resize(img_4k, target_size, interpolation=flag)
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            # ✅ TODO 1~3 完成
            resized_img = cv2.resize(img_4k, target_size, interpolation=flag)
            
            end_time = time.perf_counter()
            times_ms.append((end_time - start_time) * 1000)
            
        visuals[name] = resized_img
        
        # ✅ TODO 4 完成
        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        
        results[name] = (mean_time, std_time)

    return results, visuals

def render_comparison(visuals):
    print("[*] Rendering visual comparison to 'benchmark_visual_comparison.png'...")
    images = []
    
    for name, img in visuals.items():
        display_img = img.copy()
        cv2.rectangle(display_img, (0, 0), (224, 30), (0, 0, 0), -1)
        cv2.putText(display_img, name, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        images.append(display_img)
        
    final_canvas = np.hstack(images)
    cv2.imwrite("benchmark_visual_comparison.png", final_canvas)
    print("[+] Done.\n")

if __name__ == "__main__":
    print("=== Week 10: VM Interpolation Benchmark ===\n")
    
    source_image = generate_4k_test_pattern()
    
    stats, images = benchmark_interpolation(source_image)
    
    baseline_mean = stats["INTER_NEAREST"][0]
    
    print("| Method | Mean Latency (ms) | Std Dev (ms) | Relative Cost |")
    print("|---|---|---|---|")
    for name, (mean, std) in stats.items():
        multiplier = mean / baseline_mean
        print(f"| {name} | {mean:.3f} | {std:.3f} | {multiplier:.1f}x |")
        
    print("\n")
    
    render_comparison(images)