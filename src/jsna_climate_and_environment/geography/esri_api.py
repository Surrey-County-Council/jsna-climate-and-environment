from pydantic import BaseModel
from typing import Literal
import json
import httpx
import geopandas as gpd
import polars as pl
from loguru import logger

MET_OFFICE_MAP_SERVER = "https://services.arcgis.com/Lq3V5RFuTBC9I7kv"
ONS_OPEN_GEOGRAPHY_MAP_SERVER = "https://services1.arcgis.com/ESMARspQHYMw9BZ9"


class FieldModel(BaseModel):
    name: str
    type: str
    alias: str
    sqlType: str
    domain: str | None
    defaultValue: str | None

    @property
    def normalized_alias(self):
        name = self.name
        if self.alias:
            name = self.alias
        return name.lower().strip()


class FeatureModel(BaseModel):
    attributes: dict


class QueryResponse(BaseModel):
    objectIdFieldName: str
    uniqueIdField: dict
    globalIdFieldName: str
    geometryProperties: dict
    geometryType: str
    spatialReference: dict
    fields: list[FieldModel]
    features: list[FeatureModel]

    def alias_mapper(self):
        return {f.name: f.normalized_alias for f in self.fields}

    @classmethod
    def df_validate(cls, data: dict) -> pl.DataFrame:
        validated = cls.model_validate(data)
        return pl.DataFrame([x.attributes for x in validated.features]).rename(
            validated.alias_mapper()
        )


def source_url(service_url: str, dataset: str):
    return f"{service_url}/arcgis/rest/services/{dataset}/FeatureServer"


async def async_query_without_geometry(
    client: httpx.AsyncClient, service: str, dataset: str, layer: int = 0, where="1=1"
) -> pl.DataFrame:
    response = await client.get(
        url=f"{source_url(service, dataset)}/{layer}/query",
        params=dict(where=where, outFields="*", returnGeometry=False, f="json"),
        timeout=5 * 60,
    )
    response.raise_for_status()
    return QueryResponse.df_validate(response.json())


def query_without_geometry(
    service: str, dataset: str, layer: int = 0, where="1=1"
) -> pl.DataFrame:
    response = httpx.get(
        url=f"{source_url(service, dataset)}/{layer}/query",
        params=dict(where=where, outFields="*", returnGeometry=False, f="json"),
        timeout=5 * 60,
    )
    response.raise_for_status()
    return QueryResponse.df_validate(response.json())


def create_replica(
    service: str,
    dataset: str,
    layer_queries: dict[int, dict[str, str]],
    replica_name: str = "SurreyPhit",
    data_fromat: Literal["shapefile"] = "shapefile",
) -> str:
    """for large datasets where replicas are supported, read the shapefile directly from a replica"""
    url = f"{source_url(service, dataset)}/createReplica"
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
    service: str,
    dataset: str,
) -> str:
    """for large datasets where replicas are not supported, an item_id may be used to dowload the full file (limitations may apply)"""
    response = httpx.get(
        url=f"{source_url(service, dataset)}",
        params=dict(f="json"),
        timeout=5 * 60,
    )
    response.raise_for_status()
    response_content = response.json()
    if "serviceItemId" in response_content:
        return response_content["serviceItemId"]
    logger.error(response.url)
    raise httpx.HTTPStatusError(
        str(response_content), response=response, request=response.request
    )


def read_csv_lookup(
    dataset: str,
    layer: int = 0,
    where: str = "1=1",
    format: str = "csv",
    spatialRefId: int = 4326,
) -> pl.DataFrame:
    """for data hosted on arcgis hub, full file downloads may be possible, this is true for lookup files hosted by ONS"""
    item_id = get_item_id(ONS_OPEN_GEOGRAPHY_MAP_SERVER, dataset)
    url = f"https://hub.arcgis.com/api/v3/datasets/{item_id}_{layer}/downloads/data"
    logger.info(f"reading data from {url}")
    response = httpx.get(
        url=url,
        params=dict(format=format, spatialRefId=spatialRefId, where=where),
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
    url = create_replica(
        ONS_OPEN_GEOGRAPHY_MAP_SERVER, dataset, layer_queries, replica_name
    )
    logger.info(f"reading data from {url}")
    gdf = gpd.read_file(url)
    gdf["source_url"] = str(url)
    return gdf
