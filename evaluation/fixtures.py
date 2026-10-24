"""Deterministic development inputs; never inspect private conversation files."""
from pathlib import Path
import shutil


def prepare(root: Path, destination: Path) -> dict:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from PIL import Image, ImageDraw

    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def register(key, path, provenance):
        manifest[key] = {"path": str(path.resolve()), "provenance": provenance}

    def bundled(key, relative):
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundled sample: {source}")
        target = destination / (key + source.suffix)
        shutil.copyfile(source, target)
        register(key, target, str(relative))

    image = Image.new("RGB", (608, 608), (75, 100, 70))
    draw = ImageDraw.Draw(image)
    for y in range(20, 580, 60):
        for x in range(20, 580, 60):
            draw.rectangle((x, y, x + 32, y + 32), fill=(180, 170, 155))
    post = image.copy()
    draw = ImageDraw.Draw(post)
    for y in range(20, 260, 60):
        for x in range(20, 260, 60):
            draw.rectangle((x, y, x + 32, y + 32), fill=(105, 100, 90))
    for key, value in (("eq_pre", image), ("eq_post", post),
                       ("eq_first", image), ("eq_second", post), ("plain_rgb", image)):
        path = destination / (key + ".png")
        value.save(path)
        register(key, path, "synthetic geometric RGB fixture; not a real earthquake")

    sar = Path("model/sen1floods11-segmentation-main/app/sample_data/S1")
    for key, filename in (("flood_ghana", "Ghana_53713_S1Hand.tif"),
                          ("flood_usa", "USA_905409_S1Hand.tif"),
                          ("flood_mekong", "Mekong_333434_S1Hand.tif"),
                          ("flood_somalia", "Somalia_699062_S1Hand.tif")):
        bundled(key, sar / filename)
    floga = Path("model/FLOGA-main/FLOGA-main/data/floga2018_infer/sen2_20_mod_500/2018")
    for index in range(3):
        for phase in ("pre", "post"):
            bundled(f"fire_{index}_{phase}", floga / f"sample{index:08d}_0_2018.sen2_20_{phase}.npy")
    for index in range(1, 4):
        bundled(f"landslide_{index}", Path(f"model/landslide_l4s/samples/images/image_{index}.h5"))
        bundled(f"oil_{index}", Path(f"model/oilspill_cbdnet/samples/images/{10000+index}_sat.jpg"))
    for index, filename in enumerate(("images_(1)_0007.jpg", "images_(1)_0042.jpg", "images_(1)_0037.jpg"), 1):
        bundled(f"pest_{index}", Path("model/pest_yolo/samples/images") / filename)
    for key, filename in (("algal_09", "S2_09_bloom.tif"), ("algal_05", "S2_05_bloom.tif"), ("algal_06", "S2_06_clean.tif")):
        bundled(key, Path("model/algalbloom_ndci/samples") / filename)
    for index, filename in enumerate(("AoF07755.jpg", "AoF07740.jpg", "AoF07745.jpg"), 1):
        bundled(f"urban_{index}", Path("model/urban_fire_smoke_yolov5l/test_data/images") / filename)

    rng = np.random.default_rng(20261024)
    path = destination / "traffic_features.npz"
    np.savez(path, data=rng.normal(0, .1, (50, 20, 4096)).astype("float32"),
             det=np.tile(np.array([20, 20, 80, 80], dtype="float32"), (50, 19, 1)),
             fps=10., model_profile="ccd", timestamps_s=np.arange(50) / 10.)
    register("traffic_features", path, "synthetic CCD-format features; not a real accident video")

    def raster(key, array):
        path = destination / (key + ".tif")
        with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1],
                           count=len(array), dtype="float32", crs="EPSG:3857",
                           transform=from_origin(12697000, 2570000, 10, 10)) as target:
            target.write(array.astype("float32"))
        register(key, path, "synthetic constant/gradient raster with known numerical oracle")

    for key, value in (("nir", .6), ("red", .2), ("swir", .3), ("time_a", 2.), ("time_b", 5.)):
        raster(key, np.full((1, 16, 16), value))
    raster("gradient", np.broadcast_to(np.linspace(0, 1, 16)[None, None, :], (1, 16, 16)).copy())
    raster("five_band", np.full((5, 16, 16), .3))
    return manifest
