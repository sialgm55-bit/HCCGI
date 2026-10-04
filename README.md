# Weeks 1–3 Lab: Image Pixels, Filters, and Edges

**Course Project:** Computer Vision & Digital Image Processing  
**Lab Assignment:** Image Pixels, Filters, and Edges  
**Student Initial / Author:** G.M (HCCGI Image Lab)  
**Script Name:** `weeks01-03_image_lab.py`

---

## 1. Environment Setup & Requirements

This project requires **Python 3.8+** along with OpenCV (`opencv-python`), NumPy (`numpy`), and Matplotlib (`matplotlib`).

### Installation
To set up your environment, clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### Exact Run Command
To run the full lab and generate all output montages:

```bash
python3 weeks01-03_image_lab.py
```
*(On Windows systems, you can also execute `python weeks01-03_image_lab.py`)*

---

## 2. Submission Contract Checklist

| Required Artifact | Path | Status |
| :--- | :--- | :--- |
| **Original Photograph** | `images/original.jpg` | Verified |
| **Main Python Script** | `weeks01-03_image_lab.py` | Verified |
| **Pixel Views Montage** | `outputs/pixel_views.png` | Generated |
| **Adjustments Montage** | `outputs/adjustments.png` | Generated |
| **Blur & Edges Montage** | `outputs/blur_and_edges.png` | Generated |
| **Dependencies File** | `requirements.txt` | Verified |
| **Documentation** | `README.md` | Complete |

---

## 3. Task 1 — Image Data Inspection Report

The loaded image (`images/original.jpg`) was programmatically analyzed via `inspect_image()` yielding the following measurements:

- **Width:** `896` pixels
- **Height:** `1200` pixels
- **Channels:** `3` (Color depth: 24-bit total, 8 bits/channel)
- **NumPy Shape:** `[1200, 896, 3]`
- **Pixel Count:** `1,075,200` total spatial pixels
- **Estimated Data Size:** `3,225,600` bytes (~3.075 MB uncompressed raw buffer)
- **Color Order:** `"BGR"` (OpenCV default storage order)

### Dimension & Format Explanation
1. **NumPy Shape `[1200, 896, 3]`:** NumPy indexes images in matrix format as `[rows, columns, channels]`. The height (`1200`) corresponds to matrix rows ($Y$-axis), width (`896`) corresponds to matrix columns ($X$-axis), and $3$ represents color channels.
2. **Color Space (`BGR`):** OpenCV loads standard color images in **Blue-Green-Red (BGR)** channel order rather than RGB.
3. **Uncompressed Buffer Size:** $\text{Width} \times \text{Height} \times \text{Channels} \times 1\text{ byte} = 896 \times 1200 \times 3 = 3,225,600\text{ bytes}$.

---

## 4. Task 2 — Color and Resolution Observations

- **Original Image Dimensions:** `896 x 1200`
- **Downsampled Image Dimensions:** `448 x 600` (50% scale reduction)
- **Generated Output Montage:** `outputs/pixel_views.png`

### Channel Differences & Observations
- **Channel Difference:** In the **Blue channel view**, the handwritten blue pen ink ("HCCGI G.M") exhibits high intensity brightness relative to the paper background. Conversely, in the **Red channel view**, warm wooden desk surfaces and ambient reflections display significantly higher luminance, while blue ink strokes appear noticeably darker.
- **Lost Detail in Downsampling:** In the downsampled `448 x 600` view (produced via area interpolation), fine high-frequency details such as subtle paper grain textures and faint faint notebook rule lines are smoothed out and partially lost due to spatial pixel averaging.

---

## 5. Task 3 — Adjustments & Thresholding Behavior

- **Parameters Used:** `brightness_delta = 40`, `contrast_factor = 1.5`, `threshold = 127`
- **Generated Output Montage:** `outputs/adjustments.png`

### Parameter Impact & Mathematical Behavior
1. **Brightness (`brightness_delta = 40`):** Implemented via additive offset ($I' = \text{clip}(I + 40, 0, 255)$). Shifting intensity values upward brightens dark shadow regions on the desk while maintaining valid unsigned 8-bit integer bounds (`uint8`).
2. **Contrast (`contrast_factor = 1.5`):** Implemented via multiplicative scaling ($I' = \text{clip}(I \times 1.5, 0, 255)$). Widens the dynamic range between darker handwritten text and lighter notebook paper.
3. **Binary Thresholding (`threshold = 127`):** Maps grayscale pixel values $I(x,y)$ to binary output:
   $$\text{Output}(x,y) = \begin{cases} 255 & \text{if } I(x,y) > 127 \\ 0 & \text{otherwise} \end{cases}$$
   This converts handwriting strokes into crisp black foreground markings ($0$) against a pure white background ($255$).

---

## 6. Task 4 — Blur and Edge Detection Analysis

- **Parameters Used:** `kernel_size = 5` (5x5 Mean Averaging Filter)
- **Generated Output Montage:** `outputs/blur_and_edges.png`

### Edge Comparison & Blur Impact
- **Prominent Visible Edges:** Major macroscopic contours (such as the outer boundary of the spiral notebook and laptop bezel) produce strong, high-contrast magnitude responses in both the original and blurred Sobel edge maps.
- **Detail Reduced by Blur:** Applying the 5x5 mean blur suppresses high-frequency spatial noise and fine surface grain on the paper. As a result, the Sobel edge response of the blurred image eliminates spurious background texture edges, yielding cleaner, continuous structural object boundaries.

---

## 7. External Sources & References

1. OpenCV Documentation (v4.x): [OpenCV Image Processing Utilities](https://docs.opencv.org/master/d7/da8/tutorial_table_of_content_imgproc.html)
2. NumPy Documentation: [Array Manipulation & Clipping Routines](https://numpy.org/doc/stable/reference/routines.math.html)
3. Matplotlib Documentation: [Visualization Subplots & Agg Backend](https://matplotlib.org/stable/contents.html)
