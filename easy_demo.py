import glob
import os
import cv2

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save_image(path, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def run_easy():
    image_paths = glob.glob('data/*.*')

    if not image_paths:
        print("⚠️ No images found in 'data/' folder.")
        return

    out_base = 'outputs/easy'
    ensure_dir(out_base)

    print(f"🖼 Found {len(image_paths)} image(s) to process (EASY).\n")

    for path in image_paths:
        print(f"🔹 Processing: {path}")

        # read + convert to grayscale
        img = read_image(path)
        h, w = img.shape[:2]
        gray = to_grayscale(img)

        # save grayscale output
        base_name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_base, f"{base_name}_gray.png")
        save_image(out_path, gray)

        # print EASY info
        print("\n EASY INFO")
        print("   Height   :", h)
        print("   Width    :", w)
        print("   Pixels   :", h * w)
        print("   Channels :", img.shape[2])
        print("   Grayscale saved to :", out_path)
        print("\n--- done (EASY) ---\n")

    print(" EASY run completed. Check 'outputs/easy/' for grayscale images.\n")

if __name__ == '__main__':
    run_easy()
