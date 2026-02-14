import glob
import pprint
from src.contrast_module import process_image_pipeline

def _strip_gray_keys(d):
    if not isinstance(d, dict):
        return d
    for k in ['out_gray', 'gray', 'gray_path', 'out_gray_path']:
        if k in d:
            d.pop(k, None)
    for v in d.values():
        if isinstance(v, dict):
            _strip_gray_keys(v)
    return d

def run_demo():
    image_paths = glob.glob('data/*.*')

    if not image_paths:
        print("⚠️ No images found in 'data/' folder. Please add some images first.")
        return

    print(f"🖼 Found {len(image_paths)} image(s) to process.\n")

    for path in image_paths:
        print(f"🔹 Processing: {path}")

        res = process_image_pipeline(
            path,
            out_dir='outputs',
            use_clahe=False,
            save_figs=True
        )
        _strip_gray_keys(res)
        print("   → Normal HistEq Done")

        print("\n📌 MEDIUM INFO (Histogram + Stats)")
        stats = res.get("stats", {})
        print("   Mean      :", stats.get("mean"))
        print("   Variance  :", stats.get("variance"))
        print("   Std Dev   :", stats.get("std"))
        print("   Exposure  :", res.get("exposure"))
        print("   Enhanced (HistEq) saved to :", res.get("out_enhanced"))

        res2 = process_image_pipeline(
            path,
            out_dir='outputs',
            use_clahe=True,
            clahe_params={'clipLimit':3.0, 'tileGridSize':(8,8)},
            save_figs=True
        )
        _strip_gray_keys(res2)
        print("\n   → CLAHE Done")

        print("\n📌 HARD INFO (CLAHE Enhancement)")
        print("   Enhancement Method :", res2.get("method"))
        print("   PSNR               :", res2.get("psnr"))
        print("   Entropy Before     :", res2.get("entropy_before"))
        print("   Entropy After      :", res2.get("entropy_after"))
        print("   Enhanced (CLAHE) saved to :", res2.get("out_enhanced"))

        print("\n--- RESULT SUMMARY ---")
        pprint.pprint({
            'Image': path,
            'Exposure': res2.get('exposure'),
            'PSNR': res2.get('psnr'),
            'Entropy Before': res2.get('entropy_before'),
            'Entropy After': res2.get('entropy_after')
        })
        print("------------------------\n")

    print("✅ All images processed! Check 'outputs/' folder for results.\n")

if __name__ == '__main__':
    run_demo()
