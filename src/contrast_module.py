import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import exposure 
from math import log2

#Easy Level
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_image(path, as_color=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR if as_color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if as_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(path, img):
    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
    else:
        cv2.imwrite(path, img)

def easy_inspect(image):
    h, w = image.shape[:2]
    pixels = h * w
    channels = 1 if image.ndim == 2 else image.shape[2]
    info = {
        'height': h,
        'width': w,
        'pixels': pixels,
        'channels': channels
    }
    return info


#Medium Level
def to_grayscale(image):
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def compute_histogram(gray, bins=256):
    hist, bin_edges = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
    return hist, bin_edges

def cumulative_histogram(hist):
    return np.cumsum(hist)

def intensity_stats(gray):
    arr = gray.flatten().astype(np.float64)
    mean = arr.mean()
    var = arr.var()
    std = arr.std()
    return {'mean': float(mean), 'variance': float(var), 'std': float(std)}

def detect_exposure_condition(hist, total_pixels, under_thresh_ratio=0.6, over_thresh_ratio=0.6):
    low = hist[:64].sum()
    high = hist[-64:].sum()
    low_frac = low / total_pixels
    high_frac = high / total_pixels
    if low_frac > under_thresh_ratio:
        return 'Underexposed'
    if high_frac > over_thresh_ratio:
        return 'Overexposed'
    return 'Well-exposed or balanced'


#Hard Level  
def histogram_equalization(gray):
    return cv2.equalizeHist(gray)

def clahe_equalization(gray, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def psnr(original_gray, processed_gray):
    return compare_psnr(original_gray, processed_gray, data_range=255)

def entropy(gray):
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,256))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def plot_image_and_histogram(rgb, gray, hist, bin_edges, title_prefix='', show=True, savepath=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    if rgb is not None:
        plt.imshow(rgb); plt.axis('off'); plt.title(f'{title_prefix} Image')
    else:
        plt.imshow(gray, cmap='gray'); plt.axis('off'); plt.title(f'{title_prefix} Image (gray)')
    plt.subplot(1,2,2)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(centers, hist)
    plt.title(f'{title_prefix} Histogram')
    plt.xlabel('Intensity'); plt.ylabel('Frequency')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close()


def process_image_pipeline(image_path, out_dir='outputs', use_clahe=False, clahe_params=None, save_figs=True):
    ensure_dir(out_dir)
    
    rgb = read_image(image_path, as_color=True)
    info = easy_inspect(rgb)
    gray = to_grayscale(rgb)

    hist, bins = compute_histogram(gray)
    cum_hist = cumulative_histogram(hist)
    stats = intensity_stats(gray)
    exposure = detect_exposure_condition(hist, info['pixels'])
    if use_clahe:
        if clahe_params is None:
            clahe_params = {'clipLimit':2.0, 'tileGridSize':(8,8)}
        enhanced = clahe_equalization(gray, **clahe_params)
        method = 'CLAHE'
    else:
        enhanced = histogram_equalization(gray)
        method = 'HistEq'

    psnr_val = psnr(gray, enhanced)
    entropy_before = entropy(gray)
    entropy_after = entropy(enhanced)
    base = os.path.splitext(os.path.basename(image_path))[0]

    save_image(os.path.join(out_dir, f'{base}_gray.png'), gray)
    save_image(os.path.join(out_dir, f'{base}_enhanced.png'), enhanced)
    if save_figs:
        plot_image_and_histogram(rgb, gray, hist, bins, title_prefix='Before', savepath=os.path.join(out_dir, f'{base}_before_hist.png'), show=False)
        hist2, bins2 = compute_histogram(enhanced)
        plot_image_and_histogram(None, enhanced, hist2, bins2, title_prefix=f'After ({method})', savepath=os.path.join(out_dir, f'{base}_after_hist.png'), show=False)
    
    results = {
        'info': info,
        'stats': stats,
        'exposure': exposure,
        'method': method,
        'psnr': float(psnr_val),
        'entropy_before': float(entropy_before),
        'entropy_after': float(entropy_after),
        'out_gray': os.path.join(out_dir, f'{base}_gray.png'),
        'out_enhanced': os.path.join(out_dir, f'{base}_enhanced.png')
    }
    return results
