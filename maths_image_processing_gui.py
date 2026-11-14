"""
Maths Project: Convolution, Inner Product, and Image Processing Revisited
GUI image editor built with Tkinter, OpenCV, NumPy, and Pillow.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_TITLE = "Maths Image Lab | Convolution & Inner Product"
PREVIEW_SIZE = (520, 390)


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1120x750")
        self.root.minsize(900, 650)

        self.original = None          # OpenCV BGR image
        self.processed = None         # OpenCV BGR or grayscale image
        self.original_tk = None
        self.processed_tk = None

        header = tk.Frame(root, padx=16, pady=12)
        header.pack(fill="x")
        tk.Label(
            header, text="Maths Image Lab",
            font=("Arial", 22, "bold")
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Explore convolution, local inner products, and image processing.",
            font=("Arial", 11)
        ).pack(anchor="w", pady=(3, 0))

        # Split controls into two rows to accommodate new buttons
        controls_top = tk.Frame(root, padx=16, pady=4)
        controls_top.pack(fill="x")
        controls_bot = tk.Frame(root, padx=16, pady=4)
        controls_bot.pack(fill="x")

        tk.Button(controls_top, text="Open Image", command=self.open_image,
                  width=14).pack(side="left", padx=(0, 7), pady=3)
        tk.Button(controls_top, text="Grayscale", command=self.apply_grayscale,
                  width=12).pack(side="left", padx=4, pady=3)
        tk.Button(controls_top, text="Blur (3x3 Avg)", command=self.apply_blur,
                  width=15).pack(side="left", padx=4, pady=3)
        tk.Button(controls_top, text="Gaussian Blur", command=self.apply_gaussian_blur,
                  width=15).pack(side="left", padx=4, pady=3)
        tk.Button(controls_top, text="Sharpen", command=self.apply_sharpen,
                  width=12).pack(side="left", padx=4, pady=3)

        tk.Button(controls_bot, text="Edge Detect (Laplacian)", command=self.apply_edges,
                  width=22).pack(side="left", padx=(0, 7), pady=3)
        tk.Button(controls_bot, text="Sobel Edges (X)", command=self.apply_sobel_edges,
                  width=18).pack(side="left", padx=4, pady=3)
        tk.Button(controls_bot, text="Inner Product Response", command=self.apply_inner_product,
                  width=22).pack(side="left", padx=4, pady=3)

        panels = tk.Frame(root, padx=16, pady=8)
        panels.pack(fill="both", expand=True)

        left = tk.LabelFrame(panels, text="Original Image", padx=8, pady=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.LabelFrame(panels, text="Processed Image", padx=8, pady=8)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.original_label = tk.Label(left, text="Click “Open Image” to begin.",
                                       compound="center", anchor="center")
        self.original_label.pack(fill="both", expand=True)
        self.processed_label = tk.Label(right, text="Choose an effect above.",
                                        compound="center", anchor="center")
        self.processed_label.pack(fill="both", expand=True)

        footer = tk.Frame(root, padx=16, pady=10)
        footer.pack(fill="x")
        self.status = tk.StringVar(value="Ready. Open an image to begin.")
        tk.Label(footer, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(footer, text="Reset", command=self.reset_image, width=10).pack(side="right", padx=4)
        tk.Button(footer, text="Save Processed Image", command=self.save_image,
                  width=20).pack(side="right", padx=4)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Could not open image", "Please select a valid image file.")
            return
        self.original = image
        self.processed = image.copy()
        self.show_images()
        self.status.set(f"Loaded: {Path(file_path).name} — {image.shape[1]} × {image.shape[0]} pixels")

    def require_image(self):
        if self.original is None:
            messagebox.showinfo("No image", "Please click Open Image and select an image first.")
            return False
        return True

    def apply_grayscale(self):
        if not self.require_image():
            return
        self.processed = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.show_images()
        self.status.set("Grayscale: each pixel is represented by an intensity value.")

    def apply_blur(self):
        if not self.require_image():
            return
        # 3x3 averaging kernel: every nearby pixel contributes equally.
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        self.processed = cv2.filter2D(self.original, ddepth=-1, kernel=kernel)
        self.show_images()
        self.status.set("Blur: applied a 3×3 averaging convolution kernel.")

    def apply_gaussian_blur(self):
        if not self.require_image():
            return
        self.processed = cv2.GaussianBlur(self.original, (5, 5), 0)
        self.show_images()
        self.status.set("Gaussian Blur: applied a 5x5 Gaussian weighted kernel.")

    def apply_sharpen(self):
        if not self.require_image():
            return
        # Centre is emphasized and surrounding pixels are subtracted.
        kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        self.processed = cv2.filter2D(self.original, ddepth=-1, kernel=kernel)
        self.show_images()
        self.status.set("Sharpen: applied a convolution kernel that emphasizes local detail.")

    def apply_edges(self):
        if not self.require_image():
            return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        # Laplacian kernel responds strongly where intensity changes rapidly.
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        response = np.abs(response)
        maximum = float(response.max())
        if maximum > 0:
            response = response * (255.0 / maximum)
        self.processed = response.astype(np.uint8)
        self.show_images()
        self.status.set("Edge detection: the Laplacian kernel highlights rapid intensity changes.")

    def apply_sobel_edges(self):
        if not self.require_image():
            return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        # Sobel X (vertical edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        response = np.abs(sobelx)
        maximum = float(response.max())
        if maximum > 0:
            response = response * (255.0 / maximum)
        self.processed = response.astype(np.uint8)
        self.show_images()
        self.status.set("Sobel X: highlighted vertical edges.")

    def apply_inner_product(self):
        if not self.require_image():
            return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        # A local 3x3 image patch is treated as a vector of 9 values.
        # The kernel is another vector of 9 values. Their dot product gives
        # the local response at each pixel (shown as an image).
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        response = np.abs(response)
        maximum = float(response.max())
        if maximum > 0:
            response = response * (255.0 / maximum)
        self.processed = response.astype(np.uint8)
        self.show_images()
        self.status.set("Inner product: each 3×3 image patch is dotted with a 3×3 filter kernel.")

    def reset_image(self):
        if not self.require_image():
            return
        self.processed = self.original.copy()
        self.show_images()
        self.status.set("Reset: processed image restored to the original.")

    def save_image(self):
        if self.processed is None:
            messagebox.showinfo("Nothing to save", "Open an image and apply an effect first.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("Bitmap image", "*.bmp")]
        )
        if not file_path:
            return
        
        # Enforce file extension to prevent cv2.imwrite failure
        if not Path(file_path).suffix:
            file_path += ".png"

        image_to_save = self.processed
        if not cv2.imwrite(file_path, image_to_save):
            messagebox.showerror("Save failed", "The image could not be saved to that location.")
            return
        self.status.set(f"Saved processed image: {Path(file_path).name}")
        messagebox.showinfo("Saved", f"Image saved successfully:\n{file_path}")

    @staticmethod
    def to_tk_image(cv_image):
        if cv_image.ndim == 2:
            pil_image = Image.fromarray(cv_image)
        else:
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
        pil_image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_image)

    def show_images(self):
        if self.original is not None:
            self.original_tk = self.to_tk_image(self.original)
            self.original_label.configure(image=self.original_tk, text="")
        if self.processed is not None:
            self.processed_tk = self.to_tk_image(self.processed)
            self.processed_label.configure(image=self.processed_tk, text="")


def main():
    root = tk.Tk()
    ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()