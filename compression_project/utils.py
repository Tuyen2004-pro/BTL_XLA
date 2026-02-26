import numpy as np


def compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return 0
    return original_size / compressed_size


def compression_percent(original_size, compressed_size):
    if original_size == 0:
        return 0
    return (1 - compressed_size / original_size) * 100


def check_lossless(original, decoded):
    return np.array_equal(original, decoded)


# ✅ phân loại ảnh cho báo cáo
def classify_image(gray):
    """
    Phân loại ảnh thông minh hơn cho đồ án:
    - Binary
    - Uniform (nhiều vùng đồng nhất)
    - Grayscale (phức tạp)
    """

    data = gray.flatten()
    unique_vals = np.unique(data)

    # ===== Binary =====
    if len(unique_vals) <= 2:
        return "Binary"

    # ===== Tính entropy =====
    hist = np.bincount(data, minlength=256).astype(float)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))

    # ===== Tính mật độ run RLE =====
    changes = np.sum(data[1:] != data[:-1])
    run_density = changes / len(data)

    # ===== Quy tắc phân loại (đã test cho đồ án) =====
    if entropy < 5.0 and run_density < 0.15:
        return "Uniform"

    return "Grayscale"