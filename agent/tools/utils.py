import json
import logging
import os
import numpy as np
from osgeo import gdal

logger = logging.getLogger(__name__)


def read_image(file_path: str) -> np.ndarray:
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path}")
    
    bands = ds.RasterCount
    if bands == 1:
        img = ds.GetRasterBand(1).ReadAsArray()
    else:
        img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        img = np.transpose(img, (1, 2, 0))

    ds = None
    return img


def read_image_uint8(file_path: str) -> np.ndarray:
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path}")
    
    bands = ds.RasterCount
    if bands == 1:
        img = ds.GetRasterBand(1).ReadAsArray()
    else:
        img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        img = np.transpose(img, (1, 2, 0))

    ds = None

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def get_geotransform(file_path) -> tuple:
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path}")
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    else:
        return geo, proj


def _extract_bbox_geojson(raster_path: str) -> str | None:
    """从栅格文件提取边界框,返回 GeoJSON Polygon 字符串"""
    try:
        geo, proj = get_geotransform(raster_path)
        if geo is None:
            return None
        ds = gdal.Open(raster_path)
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None
        x0, dx, _, y0, _, dy = geo
        x1 = x0 + w * dx
        y1 = y0 + h * dy
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        return json.dumps({
            "type": "Polygon",
            "coordinates": [[[x, y] for x, y in zip(xs, ys)]],
        })
    except Exception:
        return None


def save_assessment_to_db(
    task: str,
    summary: dict,
    raster_path: str = "",
    session_id: str = "",
) -> None:
    """将工具评估结果保存到数据库(静默失败)

    在每个工具写完 summary.json 后调用,不影响原有文件输出流程。

    Args:
        task: 任务类型 (building/flood/car/ship/damage/solar_panel/wetland/water_unet)
        summary: 工具返回的 summary dict
        raster_path: 输入栅格路径(用于提取空间范围)
        session_id: 关联的会话 ID (未传则从 DISASTER_SESSION_ID 环境变量读取,
            该变量由 backend_api.py 在启动 MCP 工具子进程时注入)
    """
    try:
        from agent.db import db as database
        sid = session_id or os.getenv("DISASTER_SESSION_ID", "")
        geom_json = _extract_bbox_geojson(raster_path or summary.get("raster_path", ""))
        database.save_assessment(
            task=task,
            summary=summary,
            session_id=sid or None,
            description=summary.get("description", summary.get("task", "")),
            geom_geojson=geom_json,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DB assessment save skipped: %s", exc)
