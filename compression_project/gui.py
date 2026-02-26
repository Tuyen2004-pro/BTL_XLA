import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from algorithms.rle import *
from algorithms.huffman import *
from utils import *

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class CompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("So sánh nén ảnh RLE - Huffman")
        self.root.geometry("1250x900")

        self.setup_ui()

        self.last_data = None
        self.current_rle_img = None
        self.current_huf_img = None

    # ================= UI =================
    def setup_ui(self):

        self.table = ttk.Treeview(
            self.root,
            columns=("Name", "Type", "Org(KB)", "RLE(KB)", "HUF(KB)",
                     "RLE Ratio", "HUF Ratio", "Best", "Lossless"),
            show="headings"
        )

        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, width=110)

        # ✅ TAG màu
        self.table.tag_configure("good", background="#d4f8d4")   # xanh
        self.table.tag_configure("bad", background="#ffd6d6")    # đỏ

        self.table.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Chọn ảnh",
                  command=self.choose_image).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Xem sơ đồ",
                  command=self.show_chart).pack(side=tk.LEFT, padx=10)

        tk.Button(btn_frame, text="Xuất ảnh đã giải nén ra .png",
                  command=self.export_images).pack(side=tk.LEFT, padx=10)

        img_frame = tk.Frame(self.root)
        img_frame.pack(pady=20)

        self.lbl_original = tk.Label(img_frame, text="Ảnh gốc")
        self.lbl_original.pack(side=tk.LEFT, padx=20)

        self.lbl_rle = tk.Label(img_frame, text="RLE Decode")
        self.lbl_rle.pack(side=tk.LEFT, padx=20)

        self.lbl_huf = tk.Label(img_frame, text="Huffman Decode")
        self.lbl_huf.pack(side=tk.LEFT, padx=20)

    # ================= CHỌN ẢNH =================
    def choose_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")]
        )
        if path:
            self.process_image(path)

    # ================= XỬ LÝ =================
    def process_image(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh!")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        data = gray.flatten()
        name = os.path.splitext(os.path.basename(path))[0]

        # ✅ RAW SIZE chuẩn học thuật
        original_size = data.nbytes / 1024

        # ✅ phân loại ảnh
        img_type = classify_image(gray)

        # ================= RLE =================
        t1 = time.time()
        rle_code = rle_encode(data)
        save_rle_to_file(rle_code, (h, w),
                         f"{OUTPUT_DIR}/{name}_rle.bin")
        t2 = time.time()

        rle_size = os.path.getsize(
            f"{OUTPUT_DIR}/{name}_rle.bin") / 1024

        rle_decoded = rle_decode(rle_code)

        # ================= HUFFMAN =================
        t3 = time.time()
        encoded, freq, padding = huffman_encode(data)
        save_huffman_to_file(
            encoded, freq, padding, (h, w),
            f"{OUTPUT_DIR}/{name}_huf.bin"
        )
        t4 = time.time()

        huf_size = os.path.getsize(
            f"{OUTPUT_DIR}/{name}_huf.bin") / 1024

        huf_decoded = huffman_decode(encoded, freq, padding)

        # ================= LOSSLESS =================
        rle_ok = check_lossless(data, rle_decoded)
        huf_ok = check_lossless(data, huf_decoded)
        lossless_text = f"RLE:{'OK' if rle_ok else 'FAIL'} | HUF:{'OK' if huf_ok else 'FAIL'}"

        # ================= RATIO =================
        rle_ratio = compression_ratio(original_size, rle_size)
        huf_ratio = compression_ratio(original_size, huf_size)

        # ================= AUTO KẾT LUẬN =================
        if rle_ratio > huf_ratio:
            best_method = "RLE"
        elif huf_ratio > rle_ratio:
            best_method = "Huffman"
        else:
            best_method = "Tie"

        # ================= TAG MÀU =================
        # nếu cả hai đều phình → đỏ
        if rle_ratio < 1 and huf_ratio < 1:
            tag = "bad"
        else:
            tag = "good"

        # reshape để hiển thị
        rle_img = rle_decoded.reshape((h, w))
        huf_img = huf_decoded.reshape((h, w))

        self.current_rle_img = rle_img
        self.current_huf_img = huf_img

        self.last_data = (
            original_size, rle_size, huf_size,
            (t2 - t1) * 1000, (t4 - t3) * 1000
        )

        # ================= INSERT TABLE =================
        self.table.insert("", "end",
                          values=(
                              name,
                              img_type,
                              round(original_size, 2),
                              round(rle_size, 2),
                              round(huf_size, 2),
                              round(rle_ratio, 2),
                              round(huf_ratio, 2),
                              best_method,
                              lossless_text
                          ),
                          tags=(tag,)
                          )

        self.show_images(gray, rle_img, huf_img)

    # ================= HIỂN THỊ ẢNH =================
    def show_images(self, img1, img2, img3):

        def convert(img):
            img = Image.fromarray(img)
            img = img.resize((250, 250))
            return ImageTk.PhotoImage(img)

        self.img1 = convert(img1)
        self.img2 = convert(img2)
        self.img3 = convert(img3)

        self.lbl_original.config(image=self.img1)
        self.lbl_rle.config(image=self.img2)
        self.lbl_huf.config(image=self.img3)

    # ================= EXPORT =================
    def export_images(self):

        if self.current_rle_img is None:
            messagebox.showinfo("Thông báo", "Chưa có ảnh để xuất!")
            return

        folder = filedialog.askdirectory()
        if not folder:
            return

        rle_path = os.path.join(folder, "decoded_rle.png")
        huf_path = os.path.join(folder, "decoded_huffman.png")

        cv2.imwrite(rle_path, self.current_rle_img)
        cv2.imwrite(huf_path, self.current_huf_img)

        messagebox.showinfo(
            "Thành công",
            f"Đã xuất:\n{rle_path}\n{huf_path}"
        )

    # ================= BIỂU ĐỒ =================
    def show_chart(self):
        if self.last_data is None:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu!")
            return

        org, rle, huf, trle, thuf = self.last_data

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.bar(["Original", "RLE", "Huffman"],
                [org, rle, huf])
        plt.title("So sánh kích thước (KB)")

        plt.subplot(1, 2, 2)
        plt.bar(["RLE", "Huffman"],
                [trle, thuf])
        plt.title("So sánh thời gian (ms)")

        plt.show()