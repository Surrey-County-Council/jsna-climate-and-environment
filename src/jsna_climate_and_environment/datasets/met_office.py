from pydantic import BaseModel, computed_field
from typing import Literal
from pathlib import Path
from jsna_climate_and_environment.config import OUTPUT_DIR
import httpx
import polars as pl
import polars.selectors as cs
import anyio
from jsna_climate_and_environment.geography.places import SURREY_DISTRICTS
from jsna_climate_and_environment.geography.esri_api import (
    async_query_without_geometry,
    MET_OFFICE_MAP_SERVER,
    source_url,
)


class Metadata(BaseModel):
    dataset: str
    indicator_type: str
    readable_name: str
    deffinition: str

    @computed_field
    @property
    def source_url(self) -> str:
        return source_url(MET_OFFICE_MAP_SERVER, self.dataset)


LA_DATASETS: tuple[Metadata, ...] = (
    Metadata(
        dataset="annual_count_of_icing_days_projections_local_authority_v2",
        indicator_type="icing days",
        readable_name="Number of days where maximum temperature is below 0°c",
        deffinition="When temperatures fail to increase above 0°c there is expected to be more extreeme damage to crops, transport disruption and increased energy demand",
    ),
    Metadata(
        dataset="annual_count_of_frost_days_projections_local_authority_v2",
        indicator_type="frost days",
        readable_name="Number of days where minimum temperature is below 0°c",
        deffinition="When temperatures fall below 0°c there is expected to be damage to crops, transport disruption and increased energy demand",
    ),
    Metadata(
        dataset="annual_count_of_growing_degree_days_projections_local_authority_v2",
        indicator_type="gdd",
        readable_name="Number of days where average temperature is above 5.5°c",
        deffinition="When the average temperature is above 5.5°c the conditions are suitable for plant growth",
    ),
    Metadata(
        dataset="annual_count_of_cooling_degree_days_projections_local_authority_v2",
        indicator_type="cdd",
        readable_name="Number of days where average temperature is above 22°c",
        deffinition="When the average temperature is above 22°c the sustained increased temperatures are expected to increase energy demand for cooling",
    ),
    Metadata(
        dataset="annual_count_of_summer_days_projections_local_authority_v2",
        indicator_type="summer days",
        readable_name="Count of days where maximum temperature is above 25°c",
        deffinition="When temperatures reach above 25°c there is expected to be an increase in heat related stress and hospital admissions",
    ),
    Metadata(
        dataset="annual_count_of_tropical_nights_projections_local_authority_v2",
        indicator_type="tropical nights",
        readable_name="Count of days where minimum temperature is above 20°c",
        deffinition="When temperatures do not fall below 20°c there is expected to be an increase in heat related stress and hospital admissions",
    ),
    Metadata(
        dataset="annual_count_of_hot_summer_days_projections_local_authority_v2",
        indicator_type="hsd",
        readable_name="Number of days where maximum temperature is above 30°c",
        deffinition="When temperatures go above 30°c there is expected to be an increase in heat related illness, transport disruption due to overheating and increased water demand",
    ),
    Metadata(
        dataset="annual_count_of_extreme_summer_days_projections_local_authority_v2",
        indicator_type="esd",
        readable_name="Number of days where maximum temperature is above 35°c",
        deffinition="When temperatures go above 35°c there is expected to be an extreme increase in heat related illness, transport disruption due to overheating and increased water demand",
    ),
    Metadata(
        dataset="winter_minimum_temperature_change_projections_local_authority_v2",
        indicator_type="tasmin winter",
        readable_name="Winter minimum temperature",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="winter_average_temperature_change_projections_local_authority_v2",
        indicator_type="tas winter",
        readable_name="Winter average temperature",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="annual_average_temperature_change_projections_local_authority_v2",
        indicator_type="tas annual",
        readable_name="Annual average temperature",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="summer_average_temperature_change_projections_local_authority_v2",
        indicator_type="tas summer",
        readable_name="Summer average temperature",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="summer_maximum_temperature_change_projections_local_authority_v2",
        indicator_type="tasmax summer",
        readable_name="Summer maximum temperature",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="winter_precipitation_change_projections_local_authority_v2",
        indicator_type="precipitation winter",
        readable_name="Winter precipitation (mm/day)",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
    Metadata(
        dataset="summer_precipitation_change_projections_local_authority_v2",
        indicator_type="precipitation summer",
        readable_name="Summer precipitation (mm/day)",
        deffinition="1981-2000 is roughly equivalent to a 0.51°c warming scenario, 2001-2021 is roughly equivalent to a 0.87°c warming scenario",
    ),
)
indicator_order = {
    "1981-2000": 0.51,
    "2001-2020": 0.87,
    "1.5°c": 1.5,
    "2°c": 2.0,
    "2.5°c": 2.5,
    "3°c": 3.0,
    "3.5°c": 3.5,
    "4°c": 4.0,
}


class ValidationError(AssertionError):
    pass


def extract_category_part(group: Literal[1, 2, 3]) -> pl.Expr:
    """The 'category' column follows a convention which can be split and validated by a regex.

    When this regex fails, the convention observed in the data no longer holds true.

    - The convention '{name} {scenario} {aggregate}'
    - an example 'ccd 2.5°c median'
    """
    # group 1 any characters preceeding the scenario pattern
    # group 2 any digits or the following special characters '-.°c'
    # group 3 the final word at the end of the string
    variable_pattern = r"(.+)\s([\d\-\.°c]+)\s(\w+)$"
    return pl.col("variable").str.extract(variable_pattern, group)


def transform_adjustment(calculation: pl.Expr) -> pl.Expr:
    return (
        pl.when(pl.col("scenario_type").str.contains("°c"))
        .then(calculation)
        .otherwise(pl.col("value"))
    )


def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    """A single dataframe undergoes the following transformations:
    - drop redundant columns
    """
    # This only works on data including districts only
    assert df["category"].unique().item() == "District"
    # the pattern below extracts the last word, the numeric measure and the prefix. spaces are assumed to sepperate each part
    indicator_type = extract_category_part(1)
    scenario_type = extract_category_part(2)
    agg_type = extract_category_part(3)

    df = (
        df.drop("objectid", "name", "category", "shape__area", "shape__length")
        .unpivot(index="code")
        .select(
            district_code="code",
            value="value",
            variable="variable",
            indicator_type=indicator_type,
            scenario_type=scenario_type,
            agg_type=agg_type,
            degrees_above_baseline=scenario_type.replace_strict(indicator_order),
        )
    )

    records_with_nulls = (
        df.null_count()
        .transpose(include_header=True, column_names=["null_count"])
        .filter(pl.col("null_count") > 0)["column"]
        .to_list()
    )
    if records_with_nulls:
        null_records = df.select(
            *records_with_nulls, original_variable="variable"
        ).filter(pl.any_horizontal(pl.col(c).is_null() for c in records_with_nulls))
        raise ValidationError(f"null values in {null_records}")
    return df


async def _iter_all_datasets() -> list[pl.DataFrame]:
    client = httpx.AsyncClient()
    data: list[pl.DataFrame] = []
    for metadata in LA_DATASETS:
        df = await async_query_without_geometry(
            client,
            MET_OFFICE_MAP_SERVER,
            metadata.dataset,
            where=f"CODE in {SURREY_DISTRICTS}",
        )
        if df.is_empty():
            raise ValidationError(
                f"Dataset is empty indicating a data sourcing issue for {metadata.source_url}"
            )

        transformed = transform_data(df).filter(
            indicator_type=pl.lit(metadata.indicator_type)
        )
        if transformed.is_empty():
            raise ValidationError(
                f"Dataset is empty indicating a metadata issue for {metadata}"
            )
        data.append(transformed)

    return data


def get_climate_projections_all_districts():
    # the baseline year according to the doccuumentation is the first range of years available (ie. 1981-2020)
    baseline = pl.first("value").over(
        "district_code",
        "indicator_type",
        "agg_type",
        order_by="degrees_above_baseline",
        descending=False,
    )

    # predicted values indicating a temperature change can be added
    tas_adjusted = transform_adjustment(baseline + pl.col("value"))

    # predicted values indicating percentage change can be transformed into a value
    proportion_increase = pl.col("value") / 100
    value_change = proportion_increase * baseline.abs()
    precipitation_adjusted = transform_adjustment(baseline + value_change)

    # observed values should remain as they are, predicted values shuld be transformed to include a temperature change
    value_adjusted = (
        pl.when(pl.col("indicator_type").str.starts_with("tas"))
        .then(tas_adjusted)
        .when(pl.col("indicator_type").str.starts_with("precipitation"))
        .then(precipitation_adjusted)
        .otherwise(pl.col("value"))
    )
    return (
        pl.concat(anyio.run(_iter_all_datasets), how="vertical_relaxed")
        .with_columns(value_adjusted=value_adjusted)
        .pivot(
            "agg_type",
            values=["value", "value_adjusted"],
            index=cs.all().exclude(
                "variable", "agg_type", "value", "source", "value_adjusted"
            ),
        )
        .sort(
            "indicator_type",
            "district_code",
            "degrees_above_baseline",
        )
        .drop("degrees_above_baseline")
    )


def get_data(
    output_file: Path = OUTPUT_DIR / "climate_projections.csv", overwrite: bool = False
) -> pl.DataFrame:
    if output_file.exists() and not overwrite:
        return pl.read_csv(output_file)

    surrey_df = get_climate_projections_all_districts()
    surrey_df.write_csv(output_file)
    return surrey_df


def get_metadata(
    data: pl.DataFrame,
    output_file: Path = OUTPUT_DIR / "climate_projections_meta.csv",
    overwrite: bool = False,
) -> pl.DataFrame:
    if output_file.exists() and not overwrite:
        return pl.read_csv(output_file)

    meta_df = pl.DataFrame(LA_DATASETS)
    meta_df.write_csv(output_file)
    return meta_df


if __name__ == "__main__":
    df = get_data(overwrite=True)
    print(get_metadata(df))
