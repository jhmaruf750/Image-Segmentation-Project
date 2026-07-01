import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

base = r"d:/4-1 lab/DIP/segmentation project/ImageSegmentationProject"
os.chdir(base)

out_dirs = [
    "output/preprocessing",
    "output/threshold",
    "output/edge",
    "output/kmeans",
    "output/watershed",
    "output/comparison",
    "output/multiple_images",
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def get_img_path(preferred):
    if os.path.exists(preferred):
        return preferred
    all_jpg = glob.glob("dataset/**/*.jpg", recursive=True)
    if not all_jpg:
        raise FileNotFoundError("No JPG image found under dataset/")
    return all_jpg[0]

img_path = get_img_path("dataset/BSDS500/images/train/100075.jpg")
image = cv2.imread(img_path)
if image is None:
    raise RuntimeError(f"Failed to read image: {img_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

_, threshold_image = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
_, otsu_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive_image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

sobel_x_f = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_f = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x_f, sobel_y_f))
sobel_x = cv2.convertScaleAbs(sobel_x_f)
sobel_y = cv2.convertScaleAbs(sobel_y_f)

kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
prewitt_x_f = cv2.filter2D(blur, cv2.CV_32F, kx)
prewitt_y_f = cv2.filter2D(blur, cv2.CV_32F, ky)
prewitt = cv2.convertScaleAbs(cv2.magnitude(prewitt_x_f, prewitt_y_f))
prewitt_x = cv2.convertScaleAbs(prewitt_x_f)
prewitt_y = cv2.convertScaleAbs(prewitt_y_f)

laplacian = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
canny = cv2.Canny(blur, 50, 150)

pixels = image_rgb.reshape((-1,3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
_, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented = centers[labels.flatten()].reshape(image_rgb.shape)

watershed_image = image.copy()
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(otsu_image, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(watershed_image, markers)
watershed_image[markers == -1] = [255,0,0]
watershed_rgb = cv2.cvtColor(watershed_image, cv2.COLOR_BGR2RGB)

def save_current(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

# preprocessing
plt.figure(figsize=(12,5));
plt.subplot(1,2,1); plt.imshow(image_rgb); plt.title("Original RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
save_current("output/preprocessing/rgb_gray.png")

plt.figure(figsize=(8,5)); plt.hist(gray.ravel(), bins=256, range=[0,256], color='black'); plt.title("Histogram")
plt.xlabel("Pixel Intensity"); plt.ylabel("Frequency"); plt.grid(alpha=0.3)
save_current("output/preprocessing/histogram.png")

plt.figure(figsize=(12,5));
plt.subplot(1,2,1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(blur, cmap='gray'); plt.title("Gaussian Blur"); plt.axis("off")
save_current("output/preprocessing/gaussian_blur.png")

# threshold
plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blur, cmap='gray'); plt.title("Blur"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(threshold_image, cmap='gray'); plt.title("Global Threshold"); plt.axis("off")
save_current("output/threshold/global_threshold.png")

plt.figure(figsize=(15,5))
for i, T in enumerate([60,100,150]):
    _, b = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY)
    plt.subplot(1,3,i+1); plt.imshow(b, cmap='gray'); plt.title(f"Threshold={T}"); plt.axis("off")
save_current("output/threshold/threshold_comparison.png")

plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blur, cmap='gray'); plt.title("Blur"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(otsu_image, cmap='gray'); plt.title("Otsu"); plt.axis("off")
save_current("output/threshold/otsu.png")

plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blur, cmap='gray'); plt.title("Blur"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(adaptive_image, cmap='gray'); plt.title("Adaptive"); plt.axis("off")
save_current("output/threshold/adaptive.png")

# edge
plt.figure(figsize=(18,5));
plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,4,2); plt.imshow(sobel_x, cmap='gray'); plt.title("Sobel X"); plt.axis("off")
plt.subplot(1,4,3); plt.imshow(sobel_y, cmap='gray'); plt.title("Sobel Y"); plt.axis("off")
plt.subplot(1,4,4); plt.imshow(sobel, cmap='gray'); plt.title("Sobel"); plt.axis("off")
save_current("output/edge/sobel.png")

plt.figure(figsize=(18,5));
plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,4,2); plt.imshow(prewitt_x, cmap='gray'); plt.title("Prewitt X"); plt.axis("off")
plt.subplot(1,4,3); plt.imshow(prewitt_y, cmap='gray'); plt.title("Prewitt Y"); plt.axis("off")
plt.subplot(1,4,4); plt.imshow(prewitt, cmap='gray'); plt.title("Prewitt"); plt.axis("off")
save_current("output/edge/prewitt.png")

plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blur, cmap='gray'); plt.title("Blur"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(laplacian, cmap='gray'); plt.title("Laplacian"); plt.axis("off")
save_current("output/edge/laplacian.png")

plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(gray, cmap='gray'); plt.title("Gray"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(blur, cmap='gray'); plt.title("Blur"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(canny, cmap='gray'); plt.title("Canny"); plt.axis("off")
save_current("output/edge/canny.png")

plt.figure(figsize=(20,5));
for i, (im, t) in enumerate([(gray,"Original"),(sobel,"Sobel"),(prewitt,"Prewitt"),(laplacian,"Laplacian"),(canny,"Canny")]):
    plt.subplot(1,5,i+1); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
save_current("output/edge/edge_comparison.png")

# kmeans
plt.figure(figsize=(12,5));
plt.subplot(1,2,1); plt.imshow(image_rgb); plt.title("Original RGB"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(segmented); plt.title("K-Means (K=3)"); plt.axis("off")
save_current("output/kmeans/kmeans_result.png")

plt.figure(figsize=(18,10))
pix = image_rgb.reshape((-1,3)).astype(np.float32)
for i, k in enumerate([2,3,5,7]):
    _, l, c = cv2.kmeans(pix, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    c = np.uint8(c)
    seg = c[l.flatten()].reshape(image_rgb.shape)
    plt.subplot(2,2,i+1); plt.imshow(seg); plt.title(f"K={k}"); plt.axis('off')
save_current("output/kmeans/k_comparison.png")

# watershed
plt.figure(figsize=(15,5));
plt.subplot(1,3,1); plt.imshow(image_rgb); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(otsu_image, cmap='gray'); plt.title("Binary"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(watershed_rgb); plt.title("Watershed"); plt.axis("off")
save_current("output/watershed/watershed_result.png")

# comparison
images_all = [image_rgb, threshold_image, otsu_image, adaptive_image, sobel, prewitt, laplacian, canny, segmented, watershed_rgb]
titles_all = ["Original","Global Threshold","Otsu","Adaptive","Sobel","Prewitt","Laplacian","Canny","K-Means","Watershed"]
plt.figure(figsize=(18,12))
for i, im in enumerate(images_all):
    plt.subplot(2,5,i+1)
    if len(im.shape)==2: plt.imshow(im, cmap='gray')
    else: plt.imshow(im)
    plt.title(titles_all[i], fontsize=10)
    plt.axis('off')
save_current("output/comparison/final_comparison.png")

# multiple_images validation outputs
selected = ["100075.jpg","12003.jpg","118035.jpg","113044.jpg","108073.jpg"]
out_names = ["bear_validation.png","starfish_validation.png","building_validation.png","horse_validation.png","tiger_validation.png"]

def process_and_plot(img_full):
    im = cv2.imread(img_full)
    if im is None:
        return False
    irgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(irgb, cv2.COLOR_RGB2GRAY)
    b = cv2.GaussianBlur(g, (5,5), 0)
    _, gth = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)
    _, ots = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cn = cv2.Canny(b, 100, 200)
    pv = irgb.reshape((-1,3)).astype(np.float32)
    _, ll, cc = cv2.kmeans(pv, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    cc = np.uint8(cc)
    km = cc[ll.flatten()].reshape(irgb.shape)
    ws_img = im.copy()
    k = np.ones((3,3), np.uint8)
    op = cv2.morphologyEx(ots, cv2.MORPH_OPEN, k, iterations=2)
    sbg = cv2.dilate(op, k, iterations=3)
    dt = cv2.distanceTransform(op, cv2.DIST_L2, 5)
    _, sfg = cv2.threshold(dt, 0.4*dt.max(), 255, 0)
    sfg = np.uint8(sfg)
    unk = cv2.subtract(sbg, sfg)
    _, mk = cv2.connectedComponents(sfg)
    mk = mk + 1
    mk[unk==255] = 0
    mk = cv2.watershed(ws_img, mk)
    ws_img[mk==-1] = [255,0,0]
    wsrgb = cv2.cvtColor(ws_img, cv2.COLOR_BGR2RGB)
    ims = [irgb, gth, ots, cn, km, wsrgb]
    titles = ["Original","Global","Otsu","Canny","K-Means","Watershed"]
    plt.figure(figsize=(18,10))
    for i in range(6):
        plt.subplot(2,3,i+1)
        if len(ims[i].shape)==2: plt.imshow(ims[i], cmap='gray')
        else: plt.imshow(ims[i])
        plt.title(titles[i]); plt.axis('off')
    return True

for src, outn in zip(selected, out_names):
    p = os.path.join("dataset/BSDS500/images/train", src)
    if not os.path.exists(p):
        p = img_path
    if process_and_plot(p):
        save_current(os.path.join("output/multiple_images", outn))

print("Done: all output images saved.")
