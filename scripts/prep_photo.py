#!/usr/bin/env python3
"""
Prepare the profile photo for clean ASCII conversion:
  1. remove the background so the subject is isolated (uses native macOS Vision or rembg)
  2. smooth texture while keeping edges sharp
  3. stretch tones so skin lands near light/white and hair/lines stay dark
  4. darken line work (difference-of-gaussians ridges)
  5. composite onto white and crop square around the subject

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
Usage:
    python scripts/prep_photo.py [source-photo.jpg] [source-prepped.png]
"""
import os
import sys
import subprocess
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

def extract_subject_mask(input_path):
    """Cut out the subject using native macOS Vision framework or rembg."""
    # First try rembg if installed
    try:
        from rembg import remove
        cut = remove(Image.open(input_path).convert("RGBA"))
        return cut
    except ImportError:
        pass

    # On macOS, use Swift Vision API (zero external weights required)
    swift_script = """
import Foundation
import Cocoa
import Vision
import CoreImage

guard CommandLine.arguments.count >= 3 else { exit(1) }
let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
guard let ciImage = CIImage(contentsOf: inputURL) else { exit(1) }

let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
let request = VNGenerateForegroundInstanceMaskRequest()
do {
    try handler.perform([request])
    guard let result = request.results?.first else { exit(1) }
    let maskBuffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
    let maskCI = CIImage(cvPixelBuffer: maskBuffer)
    let transparent = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: ciImage.extent)
    let blendFilter = CIFilter(name: "CIBlendWithMask")!
    blendFilter.setValue(ciImage, forKey: kCIInputImageKey)
    blendFilter.setValue(transparent, forKey: kCIInputBackgroundImageKey)
    blendFilter.setValue(maskCI, forKey: kCIInputMaskImageKey)
    guard let blended = blendFilter.outputImage else { exit(1) }
    let context = CIContext()
    let cgImage = context.createCGImage(blended, from: ciImage.extent)!
    let rep = NSBitmapImageRep(cgImage: cgImage)
    let pngData = rep.representation(using: .png, properties: [:])!
    try pngData.write(to: outputURL)
} catch {
    exit(1)
}
"""
    tmp_swift = "/tmp/bg_remover_run.swift"
    tmp_out = "/tmp/cutout_temp.png"
    with open(tmp_swift, "w") as f:
        f.write(swift_script)
    res = subprocess.run(["swift", tmp_swift, os.path.abspath(input_path), tmp_out])
    if res.returncode == 0 and os.path.exists(tmp_out):
        return Image.open(tmp_out).convert("RGBA")

    # Fallback: return as RGBA with full alpha
    im = Image.open(input_path).convert("RGBA")
    return im

def prep_photo():
    if not os.path.exists(INP):
        print(f"Input photo not found at {INP}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {INP}...")
    cut = extract_subject_mask(INP)
    alpha = np.array(cut.split()[-1])
    gray_im = cut.convert("L")

    # Smooth texture
    smooth_im = gray_im
    for _ in range(2):
        smooth_im = smooth_im.filter(ImageFilter.SMOOTH_MORE)
    smooth = np.array(smooth_im, dtype=np.float32)

    # Tone stretch over subject
    subject_pixels = smooth[alpha > 128]
    if len(subject_pixels) > 0:
        lo, hi = np.percentile(subject_pixels, [3, 90])
        tone = np.clip((smooth - lo) / (hi - lo), 0, 1) if hi > lo else smooth / 255.0
    else:
        tone = smooth / 255.0

    # Line / edge darkening via difference of gaussians
    fine = np.array(smooth_im.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    coarse = np.array(smooth_im.filter(ImageFilter.GaussianBlur(radius=6.0)), dtype=np.float32)
    lines = np.clip((coarse - fine) / 40.0, 0, 1)
    line_weight = 0.5
    out = np.clip(tone - line_weight * lines, 0, 1) * 255.0

    # Feather mask and composite on white
    alpha_im = cut.split()[-1].filter(ImageFilter.GaussianBlur(radius=1.0))
    mask = np.array(alpha_im, dtype=np.float32) / 255.0
    out = out * mask + 255.0 * (1.0 - mask)

    # Square crop around subject
    ys, xs = np.where(alpha > 20)
    if len(xs) > 0 and len(ys) > 0:
        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()
        side = max(max_x - min_x, max_y - min_y) + 60
        cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
        canvas = np.full((side, side), 255, dtype=np.uint8)
        x0, y0 = cx - side // 2, cy - side // 2
        sx0, sy0 = max(x0, 0), max(y0, 0)
        sx1, sy1 = min(x0 + side, out.shape[1]), min(y0 + side, out.shape[0])
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1].astype(np.uint8)
        prepped = Image.fromarray(canvas, mode="L")
    else:
        prepped = Image.fromarray(out.astype(np.uint8), mode="L")

    prepped.save(OUT)
    print(f"wrote {OUT} ({prepped.size[0]}x{prepped.size[1]})")

if __name__ == "__main__":
    prep_photo()
