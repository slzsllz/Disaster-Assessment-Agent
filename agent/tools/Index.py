import argparse

from pathlib import Path
from fastmcp import FastMCP


mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def calculate_ndvi(input_nir_path, input_red_path, output_path):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_red_path (str): Path to the Red band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndvi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDVI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and Red band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_red_path) as red_src:
        red_band = red_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    red_band = np.array(red_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + red_band + 1e-6

    # Calculate NDVI using the formula: NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir_band - red_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndvi_profile = nir_profile.copy()
    ndvi_profile.update(
        dtype=rasterio.float32,  # NDVI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDVI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndvi_profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDVI from multiple pairs of NIR/Red raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_red_paths (list[str]): Paths to Red band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndvi_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndvi`.
""")
def calculate_batch_ndvi(
    input_nir_paths: list[str],
    input_red_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDVI from multiple pairs of NIR/Red rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_red_paths (list[str]): Paths to Red band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndvi_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """

    return [
        calculate_ndvi(nir_path, red_path, out_path)
        for nir_path, red_path, out_path in zip(input_nir_paths, input_red_paths, output_paths)
    ]


def calculate_ndbi(input_swir_path, input_nir_path, output_path):
    """
    Calculate the Normalized Difference Built-Up Index (NDBI) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/ndbi_2022-01-16.tif"

    Returns:
        str: Path to the saved NDBI file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the SWIR and NIR band raster files
    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)
        swir_profile = swir_src.profile  # Get the metadata profile

    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    swir_band = np.array(swir_band, dtype=np.float32)
    nir_band = np.array(nir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = swir_band + nir_band + 1e-6

    # Calculate NDBI using the formula: NDBI = (SWIR - NIR) / (SWIR + NIR)
    ndbi = (swir_band - nir_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    ndbi_profile = swir_profile.copy()
    ndbi_profile.update(
        dtype=rasterio.float32,  # NDBI values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NDBI result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **ndbi_profile) as dst:
        dst.write(ndbi.astype(rasterio.float32), 1)  # Write the NDBI band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NDBI from multiple pairs of SWIR/NIR raster files and save results.

Parameters:
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/ndbi_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_ndbi`.
""")
def calculate_batch_ndbi(
    input_swir_paths: list[str],
    input_nir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NDBI from multiple pairs of SWIR/NIR rasters.

    Parameters:
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        input_nir_paths (list[str]): Paths to NIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/ndbi_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """

    return [
        calculate_ndbi(swir_path, nir_path, out_path)
        for swir_path, nir_path, out_path in zip(input_swir_paths, input_nir_paths, output_paths)
    ]


def calculate_nbr(input_nir_path, input_swir_path, output_path):
    """
    Calculate the Normalized Burn Ratio (NBR) from input raster files
    and save the result to a specified output path.

    Parameters:
        input_nir_path (str): Path to the Near-Infrared (NIR) band raster file.
        input_swir_path (str): Path to the Short-Wave Infrared (SWIR) band raster file.
        output_path (str): relative path for the output raster file, e.g. "question17/nbr_2022-01-16.tif"

    Returns:
        str: Path to the saved NBR file.
    """
    import os
    import rasterio
    import numpy as np

    # Open the NIR and SWIR band raster files
    with rasterio.open(input_nir_path) as nir_src:
        nir_band = nir_src.read(1)  # Read the first band (assuming single-band rasters)
        nir_profile = nir_src.profile  # Get the metadata profile

    with rasterio.open(input_swir_path) as swir_src:
        swir_band = swir_src.read(1)  # Read the first band (assuming single-band rasters)

    # Ensure the input band data is in numpy array format
    nir_band = np.array(nir_band, dtype=np.float32)
    swir_band = np.array(swir_band, dtype=np.float32)

    # Prevent division by zero by adding a small offset (e.g., 1e-6)
    denominator = nir_band + swir_band + 1e-6

    # Calculate NBR using the formula: NBR = (NIR - SWIR) / (NIR + SWIR)
    nbr = (nir_band - swir_band) / denominator

    # Update the profile for the output raster (e.g., data type and NoData value)
    nbr_profile = nir_profile.copy()
    nbr_profile.update(
        dtype=rasterio.float32,  # NBR values are floating-point numbers
        nodata=-9999,  # Set a NoData value
        compress='lzw'  # Optional: compress the output file
    )

    # Save the NBR result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **nbr_profile) as dst:
        dst.write(nbr.astype(rasterio.float32), 1)  # Write the NBR band

    return f'Result save at {TEMP_DIR / output_path}'

@mcp.tool(description="""
Batch-calculate NBR from multiple pairs of NIR/SWIR raster files and save results.

Parameters:
    input_nir_paths (list[str]): Paths to Near-Infrared (NIR) band raster files.
    input_swir_paths (list[str]): Paths to Short-Wave Infrared (SWIR) band raster files.
    output_paths (list[str]): Relative output paths (e.g., "question17/nbr_2022-01-16.tif") for each pair.

Returns:
    list[str]: A list of result messages (one per output), as returned by `calculate_nbr`.
""")
def calculate_batch_nbr(
    input_nir_paths: list[str],
    input_swir_paths: list[str],
    output_paths: list[str]
) -> list[str]:
    """
    Batch-calculate NBR from multiple pairs of NIR/SWIR rasters.

    Parameters:
        input_nir_paths (list[str]): Paths to NIR band rasters.
        input_swir_paths (list[str]): Paths to SWIR band rasters.
        output_paths (list[str]): Relative output paths (e.g., "question17/nbr_2022-01-16.tif").

    Returns:
        list[str]: A list of result messages (e.g., saved file paths).
    """
    results: list[str] = []

    for nir_path, swir_path, out_path in zip(input_nir_paths, input_swir_paths, output_paths):
        res = calculate_nbr(nir_path, swir_path, out_path)
        results.append(res)

    return results


if __name__ == "__main__":
    mcp.run()
