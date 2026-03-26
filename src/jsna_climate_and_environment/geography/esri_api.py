from typing import Literal
import json
import httpx
import geopandas as gpd
import polars as pl
from loguru import logger


def create_replica(
    dataset: str,
    layer_queries: dict[int, dict[str, str]],
    replica_name: str = "SurreyPhit",
    data_fromat: Literal["shapefile"] = "shapefile",
) -> str:
    """for large datasets where replicas are supported, read the shapefile directly from a replica"""
    url = f"https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/{dataset}/FeatureServer/createReplica"
    logger.info(f"creating replica at {url}")
    response = httpx.post(
        url=url,
        data=dict(
            replicaName=replica_name,
            layers=json.dumps(list(layer_queries.keys())),
            layerQueries=json.dumps(layer_queries),
            dataFormat="shapefile",
            syncDirection="download",
            syncModel="none",
            f="json",
        ),
        timeout=5 * 60,
    )
    response.raise_for_status()
    response_content = response.json()
    if "responseUrl" in response_content:
        return response_content["responseUrl"]
    raise httpx.HTTPStatusError(
        str(response_content), response=response, request=response.request
    )


def get_item_id(
    dataset: str,
) -> str:
    """for large datasets where replicas are not supported, an item_id may be used to dowload the full file (limitations may apply)"""
    response = httpx.get(
        url=f"https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/{dataset}/FeatureServer",
        params=dict(f="json"),
        timeout=5 * 60,
    )
    response.raise_for_status()
    response_content = response.json()
    if "serviceItemId" in response_content:
        return response_content["serviceItemId"]
    raise httpx.HTTPStatusError(
        str(response_content), response=response, request=response.request
    )


def read_csv_dataset(
    dataset: str,
    layer: int = 0,
    where: str = "1=1",
    format: str = "csv",
    spatialRefId: int = 4326,
) -> pl.DataFrame:
    """for data hosted on arcgis hub, full file downloads may be possible"""
    item_id = get_item_id(dataset)
    url = f"https://hub.arcgis.com/api/v3/datasets/{item_id}_{layer}/downloads/data"
    logger.info(f"reading data from {url}")
    response = httpx.get(
        url=url,
        params=dict(format="csv", spatialRefId=spatialRefId, where=where),
        timeout=5 * 60,
    )
    return pl.read_csv(response.content, infer_schema=False).with_columns(
        source_url=pl.lit(str(response.url))
    )


def read_boundary_dataset(
    dataset: str,
    layer_queries: dict[int, dict[str, str]],
    replica_name: str = "SurreyPhit",
) -> gpd.GeoDataFrame:
    url = create_replica(dataset, layer_queries, replica_name)
    logger.info(f"reading data from {url}")
    gdf = gpd.read_file(url)
    gdf["source_url"] = str(url)
    return gdf
