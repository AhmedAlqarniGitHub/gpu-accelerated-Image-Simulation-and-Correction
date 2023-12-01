import numpy as np
from PIL import Image
import os
import math
from numba import cuda

# Define process types as integer constants
SIMULATE_PROTANOPIA = 1
SIMULATE_DEUTERANOPIA = 2
SIMULATE_TRITANOPIA = 3
CORRECT_PROTANOPIA = 4
CORRECT_DEUTERANOPIA = 5
CORRECT_TRITANOPIA = 6


@cuda.jit(device=True)
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


@cuda.jit(device=True)
def lms2rgb(l, m, s):
    rr = (l * 5.47221206) + (m * -1.1252419) + (s * 0.02980165)
    gg = (l * -4.6419601) + (m * 2.29317094) + (s * -0.19318073)
    bb = (l * 0.16963708) + (m * -0.1678952) + (s * 1.16364789)

    rr = 12.92 * rr if rr <= 0.0031308 else 1.055 * rr ** (1 / 2.4) - 0.055
    gg = 12.92 * gg if gg <= 0.0031308 else 1.055 * gg ** (1 / 2.4) - 0.055
    bb = 12.92 * bb if bb <= 0.0031308 else 1.055 * bb ** (1 / 2.4) - 0.055

    rr = rr * 255
    gg = gg * 255
    bb = bb * 255

    # Manually implement clipping
    rr = 0 if rr < 0 else (255 if rr > 255 else rr)
    gg = 0 if gg < 0 else (255 if gg > 255 else gg)
    bb = 0 if bb < 0 else (255 if bb > 255 else bb)

    return int(rr), int(gg), int(bb)


# Define the simulate and correct functions for protanopia, deuteranopia, and tritanopia
@cuda.jit(device=True)
def simulate_protanopia(l, m, s):
    ll = (0.0 * l) + (2.02344 * m) + (-2.52581 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

@cuda.jit(device=True)
def simulate_deuteranopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.494207 * l) + (0.0 * m) + (1.24827 * s)
    ss = (0.0 * l) + (0.0 * m) + (1.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

@cuda.jit(device=True)
def simulate_tritanopia(l, m, s):
    ll = (1.0 * l) + (0.0 * m) + (0.0 * s)
    mm = (0.0 * l) + (1.0 * m) + (0.0 * s)
    ss = (-0.395913 * l) + (0.801109 * m) + (0.0 * s)
    rr, gg, bb = lms2rgb(ll, mm, ss)
    return rr, gg, bb

# Define correct functions for each type of color vision deficiency

@cuda.jit(device=True)
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

@cuda.jit(device=True)
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

@cuda.jit(device=True)
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

@cuda.jit
def process_image_kernel(image_data, processed_data, process_type):
    i, j = cuda.grid(2)
    if i < image_data.shape[0] and j < image_data.shape[1]:
        r, g, b = image_data[i, j]
        l, m, s = rgb2lms(r, g, b)
        if process_type == SIMULATE_PROTANOPIA:
            ll, mm, ss = simulate_protanopia(l, m, s)
        elif process_type == SIMULATE_DEUTERANOPIA:
            ll, mm, ss = simulate_deuteranopia(l, m, s)
        elif process_type == SIMULATE_TRITANOPIA:
            ll, mm, ss = simulate_tritanopia(l, m, s)
        elif process_type == CORRECT_PROTANOPIA:
            ll, mm, ss = correct_protanopia(l, m, s)
        elif process_type == CORRECT_DEUTERANOPIA:
            ll, mm, ss = correct_deuteranopia(l, m, s)
        elif process_type == CORRECT_TRITANOPIA:
            ll, mm, ss = correct_tritanopia(l, m, s)
        
        # Ensure the values are of type uint8
        ll, mm, ss = int(ll), int(mm), int(ss)
        ll, mm, ss = max(0, min(ll, 255)), max(0, min(mm, 255)), max(0, min(ss, 255))

        # Assign the RGB values to the processed_data array
        processed_data[i, j, 0] = ll
        processed_data[i, j, 1] = mm
        processed_data[i, j, 2] = ss



# Function to process image for a specific color vision deficiency
def process_image(image_path, process_type, output_path):
    image = Image.open(image_path)
    image_data = np.array(image)

    # Allocate device memory and copy data
    image_data_device = cuda.to_device(image_data)
    processed_data_device = cuda.device_array(image_data.shape, dtype=np.uint8)

    # Define grid size
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(image_data.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(image_data.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    process_image_kernel[blockspergrid, threadsperblock](image_data_device, processed_data_device, process_type)

    # Copy result back to host
    processed_data = processed_data_device.copy_to_host()

    output_image = Image.fromarray(processed_data)
    output_image.save(output_path)


def driver(image_file_name):
    base_image_path = f"images/{image_file_name}.bmp"
    out_directory = f"out/{image_file_name}"

    # Check if output directory exists, if not, create it
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Process for each type of color vision deficiency
    process_image(base_image_path, SIMULATE_PROTANOPIA, os.path.join(out_directory, "simulate_protanopia.bmp"))
    process_image(base_image_path, SIMULATE_DEUTERANOPIA, os.path.join(out_directory, "simulate_deuteranopia.bmp"))
    process_image(base_image_path, SIMULATE_TRITANOPIA, os.path.join(out_directory, "simulate_tritanopia.bmp"))
    process_image(base_image_path, CORRECT_PROTANOPIA, os.path.join(out_directory, "correct_protanopia.bmp"))
    process_image(base_image_path, CORRECT_DEUTERANOPIA, os.path.join(out_directory, "correct_deuteranopia.bmp"))
    process_image(base_image_path, CORRECT_TRITANOPIA, os.path.join(out_directory, "correct_tritanopia.bmp"))

if __name__ == "__main__":
    driver("lena_color")

