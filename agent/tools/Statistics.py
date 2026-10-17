import argparse

from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8

mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def calc_single_image_mean(file_path: str, uint8: bool = False) -> float:
    """
    Compute mean value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        mean (float): Mean pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")

    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmean(flat))

@mcp.tool(description='''
Compute mean value of an batch of images.

Args:
    file_list (list): List of image file paths.
    uint8 (bool): Whether to convert image to uint8 format.

Returns:
    mean (list): List of mean pixel values.
''')
def calc_batch_image_mean(file_list: list[str], uint8: bool = False) -> list[float]:
    """
    Compute mean value of an batch of images.

    Parameters:
        file_list (list(str)): Paths to input images.

    Returns:
        mean (list(float)): Mean pixel value
    """

    return [float(calc_single_image_mean(file_path, uint8)) for file_path in file_list]


def calc_single_image_std(file_path: str, uint8: bool = False) -> float:
    """
    Compute standard deviation value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        std (float): Standard deviation
    """
    import numpy as np

    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")

    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanstd(flat, ddof=1))

@mcp.tool(description='''
Description:
Compute the standard deviation (spread of pixel values) for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- std (list[float]): List of standard deviation values, one for each input image.
''')
def calc_batch_image_std(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the standard deviation (spread of pixel values) for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - std (list[float]): List of standard deviation values, one for each input image.
    '''
    return [float(calc_single_image_std(file_path, uint8)) for file_path in file_list]


# @mcp.tool()


# @mcp.tool()
def calc_single_image_min(file_path: str, uint8: bool = False) -> float:
    """
    Compute min value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        min (flaot): Minimum pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")

    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmin(flat))


@mcp.tool(description='''
Description:
Compute the minimum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- min (list[float]): List of minimum pixel values, one for each input image.
''')

def calc_batch_image_min(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the minimum pixel value for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - min (list[float]): List of minimum pixel values, one for each input image.
    '''
    return [float(calc_single_image_min(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_max(file_path: str, uint8: bool = False) -> float:
    """
    Compute max value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        max (float): Maximum pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")

    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmax(flat))

@mcp.tool(description='''
Description:
Compute the maximum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- max (list[float]): List of maximum pixel values, one for each input image.
''')
def calc_batch_image_max(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the maximum pixel value for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - max (list[float]): List of maximum pixel values, one for each input image.
    '''
    return [float(calc_single_image_max(file_path, uint8)) for file_path in file_list]


# @mcp.tool()


# @mcp.tool()


@mcp.tool(description='''
Description:
Calculate the percentage change between two numbers, useful for comparing relative growth or decline.

Parameters:
- a (float): The original value (denominator).
- b (float): The new value (numerator).

Returns:
- percent (float): The percentage change, computed as ((b - a) / a) * 100.
                   Positive values indicate increase, negative values indicate decrease.
''')
def percentage_change(a: float, b: float):
    '''
    Description:
    Calculate the percentage change between two numbers, useful for comparing relative growth or decline.

    Parameters:
    - a (float): The original value (denominator).
    - b (float): The new value (numerator).

    Returns:
    - percent (float): The percentage change, computed as ((b - a) / a) * 100.
                    Positive values indicate increase, negative values indicate decrease.
    '''

    percent = (b - a) / a * 100
    return float(percent)


@mcp.tool(description=
    """
    Description:
        Compute the arithmetic mean (average) of a dataset.

    Parameters:
        x (list[float]):
            Input data array.

    Returns:
        mean_value (float):
            The arithmetic mean of the input values.

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """)
def mean(x: list):
    """
    Description:
        Compute the arithmetic mean (average) of a dataset.

    Parameters:
        x (list[float]):
            Input data array.

    Returns:
        mean_value (float):
            The arithmetic mean of the input values.

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    import numpy as np
    x = np.asarray(x)
    return float(np.mean(x))


@mcp.tool(description=
    """
    Description:
        Calculate the average percentage of pixels relative to a given threshold for
        one or more images and a specified band.

    Parameters:
        image_paths (str or list[str]):
            Path or list of image file paths.
        threshold (float, optional):
            Threshold value. Default = 0.75.
        mode (str, optional):
            Comparison mode: 'above' (>), 'below' (<), 'equal' (==),
            'above_equal' (>=), 'below_equal' (<=). Default = 'above'.
        band_index (int, optional):
            Band index to use (0-based). Default = 0 (first band).

    Returns:
        percentage (float):
            Average percentage of pixels matching the threshold condition across all images.

    Example:
        >>> calculate_threshold_ratio("image1.tif", threshold=0.5, mode='above')
        42.37
        >>> calculate_threshold_ratio(["img1.tif", "img2.tif"], threshold=0.8, mode='below', band_index=1)
        33.12
    """)
def calculate_threshold_ratio(image_paths: str | list[str], threshold: float = 0.75, mode: str = 'above', band_index: int = 0) -> float:
    """
    Description:
        Calculate the average percentage of pixels relative to a given threshold for
        one or more images and a specified band.

    Parameters:
        image_paths (str or list[str]):
            Path or list of image file paths.
        threshold (float, optional):
            Threshold value. Default = 0.75.
        mode (str, optional):
            Comparison mode: 'above' (>), 'below' (<), 'equal' (==),
            'above_equal' (>=), 'below_equal' (<=). Default = 'above'.
        band_index (int, optional):
            Band index to use (0-based). Default = 0 (first band).

    Returns:
        percentage (float):
            Average percentage of pixels matching the threshold condition across all images.

    Example:
        >>> calculate_threshold_ratio("image1.tif", threshold=0.5, mode='above')
        42.37
        >>> calculate_threshold_ratio(["img1.tif", "img2.tif"], threshold=0.8, mode='below', band_index=1)
        33.12
    """
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    ratios = []
    for image_path in image_paths:
        img = read_image(image_path)
        # Handle multi-band images
        if img.ndim == 3:
            # Assume (bands, height, width) or (height, width, bands)
            if img.shape[0] <= 5 and img.shape[0] < img.shape[-1]:  # bands likely first
                band = img[band_index]
            else:  # bands likely last
                band = img[..., band_index]
        else:
            band = img

        valid_pixels = ~np.isnan(band)
        total_valid_pixels = np.sum(valid_pixels)
        if total_valid_pixels == 0:
            ratios.append(0.0)
            continue

        # Apply threshold comparison based on mode
        if mode == 'above':
            matching_pixels = np.sum((band > threshold) & valid_pixels)
        elif mode == 'below':
            matching_pixels = np.sum((band < threshold) & valid_pixels)
        elif mode == 'equal':
            matching_pixels = np.sum((band == threshold) & valid_pixels)
        elif mode == 'above_equal':
            matching_pixels = np.sum((band >= threshold) & valid_pixels)
        elif mode == 'below_equal':
            matching_pixels = np.sum((band <= threshold) & valid_pixels)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: 'above', 'below', 'equal', 'above_equal', 'below_equal'")

        percentage = (matching_pixels / total_valid_pixels) * 100
        ratios.append(float(percentage))

    return float(np.mean(ratios)) if ratios else 0.0


@mcp.tool(description=
    """
    Description:
        Calculate the N-th percentile value of pixel values in a raster image,
        and return it as a native Python type matching the image's data type.

    Parameters:
        image_path (str):
            Path to the input raster (.tif) file.
        percentile (int or float):
            Percentile to calculate (range 1–100).

    Returns:
        value (int or float):
            The pixel value corresponding to the specified percentile,
            cast to the appropriate native Python type (int for integer rasters,
            float for floating-point rasters).

    Example:
        >>> get_percentile_value_from_image("fire_map.tif", 90)
        235
    """)
def get_percentile_value_from_image(image_path, percentile):
    """
    Description:
        Calculate the N-th percentile value of pixel values in a raster image,
        and return it as a native Python type matching the image's data type.

    Parameters:
        image_path (str):
            Path to the input raster (.tif) file.
        percentile (int or float):
            Percentile to calculate (range 1–100).

    Returns:
        value (int or float):
            The pixel value corresponding to the specified percentile,
            cast to the appropriate native Python type (int for integer rasters,
            float for floating-point rasters).

    Example:
        >>> get_percentile_value_from_image("fire_map.tif", 90)
        235
    """
    import rasterio
    import numpy as np
    if not (1 <= percentile <= 100):
        raise ValueError("Percentile must be between 1 and 100.")

    with rasterio.open(image_path) as src:
        image = src.read(1)
        dtype = image.dtype

        # Convert to float32 for computation
        image = image.astype(np.float32)

    # Mask invalid values
    image = image[np.isfinite(image)]

    if image.size == 0:
        raise ValueError("No valid pixel values found in the image.")

    # Compute percentile
    result = np.percentile(image, percentile)

    # Map numpy dtype to native Python type
    if np.issubdtype(dtype, np.integer):
        return int(round(result))  # Round float to nearest int if original was int
    elif np.issubdtype(dtype, np.floating):
        return float(result)
    else:
        raise TypeError(f"Unsupported raster data type: {dtype}")


@mcp.tool(description=
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy
        threshold conditions in two raster images.

    Parameters:
        path1 (str):
            Path to the first raster image (e.g., NDVI).
        threshold1 (float):
            Threshold value for the first image (e.g., NDVI > 0.3).
        path2 (str):
            Path to the second raster image (e.g., TVDI).
        threshold2 (float):
            Threshold value for the second image (e.g., TVDI > 0.7).

    Returns:
        percentage (float):
            Percentage of pixels that satisfy both conditions
            over the total valid pixels.

    Example:
        >>> calculate_intersection_percentage("ndvi.tif", 0.3, "tvdi.tif", 0.7)
        12.54
    """)
def calculate_intersection_percentage(path1, threshold1, path2, threshold2):
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy
        threshold conditions in two raster images.

    Parameters:
        path1 (str):
            Path to the first raster image (e.g., NDVI).
        threshold1 (float):
            Threshold value for the first image (e.g., NDVI > 0.3).
        path2 (str):
            Path to the second raster image (e.g., TVDI).
        threshold2 (float):
            Threshold value for the second image (e.g., TVDI > 0.7).

    Returns:
        percentage (float):
            Percentage of pixels that satisfy both conditions
            over the total valid pixels.

    Example:
        >>> calculate_intersection_percentage("ndvi.tif", 0.3, "tvdi.tif", 0.7)
        12.54
    """
    import rasterio
    import numpy as np
    # Read first image
    with rasterio.open(path1) as src1:
        data1 = src1.read(1).astype(np.float32)
        mask1 = src1.read_masks(1) > 0  # valid data mask

    # Read second image
    with rasterio.open(path2) as src2:
        data2 = src2.read(1).astype(np.float32)
        mask2 = src2.read_masks(1) > 0  # valid data mask

    # Combine valid masks
    valid_mask = mask1 & mask2

    # Filter out invalid pixels (NaN or no data)
    valid_data1 = np.where(valid_mask, data1, np.nan)
    valid_data2 = np.where(valid_mask, data2, np.nan)

    # Apply thresholds to create boolean masks
    condition1 = valid_data1 > threshold1
    condition2 = valid_data2 > threshold2

    # Logical AND to find pixels satisfying both conditions
    intersection_mask = condition1 & condition2

    # Count valid and intersecting pixels
    total_valid_pixels = np.count_nonzero(~np.isnan(valid_data1) & ~np.isnan(valid_data2))
    intersection_pixels = np.count_nonzero(intersection_mask)

    # Avoid division by zero
    if total_valid_pixels == 0:
        return 0.0

    # Calculate intersection percentage
    percentage = (intersection_pixels / total_valid_pixels) * 100
    return percentage


@mcp.tool(description=
    """
    Description:
        Count the number of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        int:
            Number of pixels satisfying all threshold conditions (intersection).

    Example:
        >>> count_pixels_satisfying_conditions(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        1250
    """)
def count_pixels_satisfying_conditions(
    image_path: str,
    band_conditions: list
) -> int:
    """
    Description:
        Count the number of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        int:
            Number of pixels satisfying all threshold conditions (intersection).

    Example:
        >>> count_pixels_satisfying_conditions(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        1250
    """
    import rasterio
    import numpy as np

    with rasterio.open(image_path) as src:
        # Read required bands
        bands = []
        for band_index, _, _ in band_conditions:
            band = src.read(band_index + 1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            bands.append(band)

    # Initialize mask as all True
    combined_mask = np.ones_like(bands[0], dtype=bool)

    for band, (band_index, threshold, compare_type) in zip(bands, band_conditions):
        valid = ~np.isnan(band)
        if compare_type.lower() == "above":
            mask = (band > threshold) & valid
        elif compare_type.lower() == "below":
            mask = (band < threshold) & valid
        else:
            raise ValueError(f"Invalid compare_type '{compare_type}', must be 'above' or 'below'")

        combined_mask &= mask  # intersection

    return int(np.sum(combined_mask))


@mcp.tool(description="""
Calculate difference between two tif files (image_b - image_a) and save result.

Parameters:
    image_a_path (str): Path to first image (will be subtracted from).
    image_b_path (str): Path to second image (will subtract from).
    output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"
    uint8 (bool): Convert to uint8 format, default False.

Returns:
    output_path (str): Full path of output file.
""")
def calculate_tif_difference(image_a_path: str, image_b_path: str, output_path: str, uint8: bool = False) -> str:
    """
    Calculate difference between two tif files (image_b - image_a) and save result.

    Parameters:
        image_a_path (str): Path to first image (will be subtracted from).
        image_b_path (str): Path to second image (will subtract from).
        output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"
        uint8 (bool): Convert to uint8 format, default False.

    Returns:
        output_path (str): Full path of output file.
    """
    import os
    from osgeo import gdal
    import numpy as np

    # Get output directory from first file
    output_path = TEMP_DIR / output_path
    os.makedirs(output_path.parent, exist_ok=True)

    # Read first image (image_a)
    ds_a = gdal.Open(image_a_path)
    if ds_a is None:
        raise RuntimeError(f"Failed to open {image_a_path}")

    bands_a = ds_a.RasterCount
    rows_a = ds_a.RasterYSize
    cols_a = ds_a.RasterXSize
    geotransform = ds_a.GetGeoTransform()
    projection = ds_a.GetProjection()

    if bands_a == 1:
        img_a = ds_a.GetRasterBand(1).ReadAsArray()
    else:
        img_a = np.stack([ds_a.GetRasterBand(i + 1).ReadAsArray() for i in range(bands_a)], axis=0)
        img_a = np.transpose(img_a, (1, 2, 0))
    ds_a = None

    # Read second image (image_b)
    ds_b = gdal.Open(image_b_path)
    if ds_b is None:
        raise RuntimeError(f"Failed to open {image_b_path}")

    bands_b = ds_b.RasterCount
    rows_b = ds_b.RasterYSize
    cols_b = ds_b.RasterXSize

    if bands_b == 1:
        img_b = ds_b.GetRasterBand(1).ReadAsArray()
    else:
        img_b = np.stack([ds_b.GetRasterBand(i + 1).ReadAsArray() for i in range(bands_b)], axis=0)
        img_b = np.transpose(img_b, (1, 2, 0))
    ds_b = None

    # Check if images have same dimensions
    if rows_a != rows_b or cols_a != cols_b or bands_a != bands_b:
        raise ValueError(f"Images must have same dimensions. Image A: {rows_a}x{cols_a}x{bands_a}, Image B: {rows_b}x{cols_b}x{bands_b}")

    # Calculate difference (image_b - image_a)
    diff_img = img_b.astype(np.float64) - img_a.astype(np.float64)

    # Convert to uint8 if needed
    if uint8:
        if len(diff_img.shape) == 2:
            min_val = np.min(diff_img)
            max_val = np.max(diff_img)
            if max_val > min_val:
                diff_img = (diff_img - min_val) / (max_val - min_val) * 255
                diff_img = diff_img.astype(np.uint8)
            else:
                diff_img = np.zeros_like(diff_img, dtype=np.uint8)
        else:
            for band in range(diff_img.shape[2]):
                band_data = diff_img[:, :, band]
                min_val = np.min(band_data)
                max_val = np.max(band_data)
                if max_val > min_val:
                    band_data = (band_data - min_val) / (max_val - min_val) * 255
                    diff_img[:, :, band] = band_data.astype(np.uint8)
                else:
                    diff_img[:, :, band] = 0

    # Save result
    driver = gdal.GetDriverByName('GTiff')
    data_type = gdal.GDT_Byte if uint8 else gdal.GDT_Float32

    if len(diff_img.shape) == 2:
        # Single band
        out_ds = driver.Create(output_path, cols_a, rows_a, 1, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_ds.GetRasterBand(1).WriteArray(diff_img)
    else:
        # Multi band
        out_ds = driver.Create(output_path, cols_a, rows_a, bands_a, data_type)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        for i in range(bands_a):
            out_ds.GetRasterBand(i + 1).WriteArray(diff_img[:, :, i])

    out_ds = None
    return f"Result save at {TEMP_DIR / output_path}"


@mcp.tool()
def grayscale_to_colormap(image_path: str, save_name: str, cmap_name: str = 'viridis', preserve_geo: bool = False):
    """
    Apply a colormap to a grayscale image and save as a color image.

    Parameters:
        image_path (str): Path to input grayscale image (e.g. .tif).
        save_name (str): Filename for save color image (.png, .jpg, or .tif).
        cmap_name (str): Name of a matplotlib colormap, e.g. 'viridis', 'RdBu', etc.
        preserve_geo (bool): If True, preserves georeferencing.
    """
    import os
    from osgeo import gdal
    import numpy as np
    import matplotlib.cm as cm
    import cv2

    # Read grayscale image
    ds = gdal.Open(image_path)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    gray = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gray = np.nan_to_num(gray, nan=0.0)

    # Normalize to 0-1
    norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)

    # Apply colormap
    cmap = cm.get_cmap(cmap_name)
    color_img = cmap(norm)[:, :, :3]  # RGB, discard alpha
    color_img_uint8 = (color_img * 255).astype(np.uint8)

    save_path = TEMP_DIR / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if preserve_geo:
        save_path = save_path.with_suffix('.tif')
        # Save as RGB GeoTIFF or plain TIF
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            str(save_path),
            xsize=color_img_uint8.shape[1],
            ysize=color_img_uint8.shape[0],
            bands=3,
            eType=gdal.GDT_Byte
        )
        for i in range(3):
            out_ds.GetRasterBand(i + 1).WriteArray(color_img_uint8[:, :, i])

            gt = ds.GetGeoTransform()
            prj = ds.GetProjection()
            if gt:
                out_ds.SetGeoTransform(gt)
            if prj:
                out_ds.SetProjection(prj)
        out_ds.FlushCache()
        out_ds = None
    else:
        # Save as PNG or JPG using OpenCV (RGB → BGR)
        bgr_img = cv2.cvtColor(color_img_uint8, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(save_path), bgr_img):
            raise OSError(f"Failed to save color image: {save_path}")
    return f'Result save at {save_path}'


if __name__ == "__main__":
    mcp.run()
