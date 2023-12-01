import numpy as np
from PIL import Image
import os
import math

def rgb2lms(r, g, b):
    rr = r / 255.0
    gg = g / 255.0
    bb = b / 255.0

    # convert to srgb
    rr = rr / 12.92 if rr <= 0.04045 else ((rr + 0.055) / 1.055) ** 2.4
    gg = gg / 12.92 if gg <= 0.04045 else ((gg + 0.055) / 1.055) ** 2.4
    bb = bb / 12.92 if bb <= 0.04045 else ((bb + 0.055) / 1.055) ** 2.4

    # convert to lms
    l = rr * 0.31399022 + gg * 0.15537241 + bb * 0.01775239
    m = rr * 0.63951294 + gg * 0.75789446 + bb * 0.10944209
    s = rr * 0.04649755 + gg * 0.08670142 + bb * 0.87256922

    return l, m, s



def lms2rgb(l, m, s):
    rr = (l * 5.47221206) + (m * -1.1252419) + (s * 0.02980165)
    gg = (l * -4.6419601) + (m * 2.29317094) + (s * -0.19318073)
    bb = (l * 0.16963708) + (m * -0.1678952) + (s * 1.16364789)

    rr = 12.92 * rr if rr <= 0.0031308 else 1.055 * rr ** (1 / 2.4) - 0.055
    gg = 12.92 * gg if gg <= 0.0031308 else 1.055 * gg ** (1 / 2.4) - 0.055
    bb = 12.92 * bb if bb <= 0.0031308 else 1.055 * bb ** (1 / 2.4) - 0.055

    rr = np.clip(rr * 255, 0, 255)
    gg = np.clip(gg * 255, 0, 255)
    bb = np.clip(bb * 255, 0, 255)
    return int(rr), int(gg), int(bb)

# Define the simulate and correct functions for protanopia, deuteranopia, and tritanopia

def simulate_protanopia(l, m, s):
    ll = (0.0 * l) + (2.02344 * m) + (-2.52581 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

def simulate_deuteranopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.494207 * l) + (0.0 * m) + (1.24827 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

def simulate_tritanopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (-0.395913 * l) + (0.801109 * m) + (0.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

# Define correct functions for each type of color vision deficiency

def correct_protanopia(l, m, s):
    ll = (0.0 * l) + (2.02344 * m) + (-2.52581 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)

    # Color correction logic specific to protanopia
    # Convert back to rgb
    rr, gg, bb = lms2rgb(ll, mm, ss)

    # get the rgb
    r, g, b = lms2rgb(l, m, s)

    # Calculate error
    rr = r - rr
    gg = g - gg
    bb = b - bb

    # Shift the color towards visible spectrum and add compensation
    rr = r
    gg = ((0.7 * rr) + (1.0 * gg)) + g
    bb = ((0.7 * rr) + (1.0 * bb)) + b

    # Clamp values towards unsigned char
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return r, g, b

def correct_deuteranopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.494207 * l) + (0.0 * m) + (1.24827 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)

    # Color correction logic specific to deuteranopia
    # Convert back to rgb
    rr, gg, bb = lms2rgb(ll, mm, ss)

    # Calculate error
    error_r, error_g, error_b = lms2rgb(l, m, s)

    # Shift the color towards visible spectrum and add compensation
    r = error_r
    g = ((0.7 * error_r) + (1.0 * error_g)) + gg
    b = ((0.7 * error_r) + (1.0 * error_b)) + bb

    # Clamp values towards unsigned char
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return r, g, b

def correct_tritanopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (-0.395913 * l) + (0.801109 * m) + (0.0 * s)

    # Color correction logic specific to tritanopia
    # Convert back to rgb
    rr, gg, bb = lms2rgb(ll, mm, ss)

    # Calculate error
    error_r, error_g, error_b = lms2rgb(l, m, s)

    # Shift the color towards visible spectrum and add compensation
    r = error_r
    g = ((0.7 * error_r) + (1.0 * error_g)) + gg
    b = ((0.7 * error_r) + (1.0 * error_b)) + bb

    # Clamp values towards unsigned char
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return r, g, b

# Function to process image for a specific color vision deficiency
def process_image(image_path, process_function, output_path):
    image = Image.open(image_path)
    image_data = np.array(image)

    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            r, g, b = image_data[i, j]
            l, m, s = rgb2lms(r, g, b)
            ll, mm, ss = process_function(l, m, s)
            image_data[i, j] = [ll, mm, ss]

    output_image = Image.fromarray(image_data)
    output_image.save(output_path)

def driver(image_file_name):
    base_image_path = f"images/{image_file_name}.bmp"
    out_directory = f"out/{image_file_name}"

    # Check if output directory exists, if not, create it
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Process for each type of color vision deficiency
    process_image(base_image_path, simulate_protanopia, os.path.join(out_directory, f"simulate_protanopia.bmp"))
    process_image(base_image_path, simulate_deuteranopia, os.path.join(out_directory, f"simulate_deuteranopia.bmp"))
    process_image(base_image_path, simulate_tritanopia, os.path.join(out_directory, f"simulate_tritanopia.bmp"))
    process_image(base_image_path, correct_protanopia, os.path.join(out_directory, f"correct_protanopia.bmp"))
    process_image(base_image_path, correct_deuteranopia, os.path.join(out_directory, f"correct_deuteranopia.bmp"))
    process_image(base_image_path, correct_tritanopia, os.path.join(out_directory, f"correct_tritanopia.bmp"))

if __name__ == "__main__":
    driver("lena_color")


