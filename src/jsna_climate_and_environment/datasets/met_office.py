from jsna_climate_and_environment.datasets.data_model import Metadata, HasDataset
from pydantic import Field
from typing import Literal
from pathlib import Path
from jsna_climate_and_environment.config import OUTPUT_DIR
import httpx
import polars as pl
import anyio
from jsna_climate_and_environment.geography.places import SURREY_DISTRICTS
from jsna_climate_and_environment.geography.esri_api import (
    async_query_without_geometry,
    MET_OFFICE_MAP_SERVER,
    source_url,
)
from loguru import logger


class ProjectionMetadata(Metadata):
    source: str = Field(
        default_factory=lambda data: source_url(MET_OFFICE_MAP_SERVER, data["dataset"])
    )
    description: str = Field(default_factory=lambda data: data["indicator_name"])


RISK_DAY_SCENARIOS: tuple[Metadata, ...] = (
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_icing_days_projections_local_authority_v2",
        name_at_source="icing days",
        indicator_name="sustained freeze risk",
        description="Number of days per year where temperatures reamin below 0°c",
        rationalle="When temperatures stay below 0°c there is expected to be more extreme damage to crops, transport disruption and increased energy demand",
    ),
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_frost_days_projections_local_authority_v2",
        name_at_source="frost days",
        indicator_name="freeze risk",
        description="Number of days per year where temperatures fall below 0°c",
        rationalle="When temperatures fall below 0°c there is expected to be damage to crops, transport disruption and increased energy demand",
    ),
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_summer_days_projections_local_authority_v2",
        name_at_source="summer days",
        indicator_name="heat risk",
        description="Number of days per year where temperatures reach above 25°c",
        rationalle="When temperatures reach above 25°c there is expected to be an increase in heat related stress and hospital admissions",
    ),
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_tropical_nights_projections_local_authority_v2",
        name_at_source="tropical nights",
        indicator_name="sustained heat risk",
        description="Number of days per year where temperatures remain above 20°c",
        rationalle="When temperatures stay above 20°c there is expected to be an increase in heat related stress and hospital admissions",
    ),
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_hot_summer_days_projections_local_authority_v2",
        name_at_source="hsd",
        indicator_name="major heat risk",
        description="Number of days per year where temperatures reach above 30°c",
        rationalle="When temperatures reach 30°c there is expected to be an increase in heat related illness, transport disruption due to overheating and increased water demand",
    ),
    ProjectionMetadata(
        measure_name="Days per Year",
        dataset="annual_count_of_extreme_summer_days_projections_local_authority_v2",
        name_at_source="esd",
        indicator_name="extreme heat risk",
        description="Number of days per year where temperatures reach above 35°c",
        rationalle="When temperatures reach 35°c there is expected to be an extreme increase in heat related illness, transport disruption due to overheating and increased water demand",
    ),
)
TEMPERATURE_SCENARIOS: tuple[Metadata, ...] = (
    ProjectionMetadata(
        measure_name="Temperature (°c)",
        dataset="winter_minimum_temperature_change_projections_local_authority_v2",
        name_at_source="tasmin winter",
        indicator_name="Winter minimum",
        description="Minimum temperature reached over 20 years",
        rationalle="Understanding the lowest temperatures expected is important when considering tollerances to extreme cold",
    ),
    ProjectionMetadata(
        measure_name="Temperature (°c)",
        dataset="winter_average_temperature_change_projections_local_authority_v2",
        name_at_source="tas winter",
        indicator_name="Winter average",
        description="Average winter temperature over 20 years",
        rationalle="Understanding the ambient winter temperature is important when considering demand for heating",
    ),
    ProjectionMetadata(
        measure_name="Temperature (°c)",
        dataset="annual_average_temperature_change_projections_local_authority_v2",
        name_at_source="tas annual",
        indicator_name="Annual average",
        description="Average temperature over 20 years",
        rationalle="Understanding the ambient yearly temperature is important when comparing local warming to global warming. Urban areas typically experience more warming.",
    ),
    ProjectionMetadata(
        measure_name="Temperature (°c)",
        dataset="summer_average_temperature_change_projections_local_authority_v2",
        name_at_source="tas summer",
        indicator_name="Summer average",
        description="Average summer temperature in over 20 years",
        rationalle="Understanding the ambient summer temperature is important when considering demand for cooling",
    ),
    ProjectionMetadata(
        measure_name="Temperature (°c)",
        dataset="summer_maximum_temperature_change_projections_local_authority_v2",
        name_at_source="tasmax summer",
        indicator_name="Summer maximum",
        description="Maximum summer temperature over 20 years",
        rationalle="Understanding the highest temperatures expected is important when considering tollerances to extreme heat",
    ),
)
PRECIPITATION_SCENARIOS: tuple[Metadata, ...] = (
    ProjectionMetadata(
        measure_name="Precipitation (mm/day)",
        dataset="winter_precipitation_change_projections_local_authority_v2",
        name_at_source="precipitation winter",
        indicator_name="Winter average",
        description="Average mm of rainfall each day in winter",
        rationalle="Higher rainfall over the winter months may indicate flooding risk",
    ),
    ProjectionMetadata(
        measure_name="Precipitation (mm/day)",
        dataset="summer_precipitation_change_projections_local_authority_v2",
        name_at_source="precipitation summer",
        indicator_name="Summer average",
        description="Average mm of rainfall each day in summer",
        rationalle="Lower rainfall over the summer months may indicate drought risk",
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


def extract_variable_part(group: Literal[1, 2, 3]) -> pl.Expr:
    """The 'category' column follows a convention which can be split and validated by a regex.

    When this regex fails, the convention observed in the data no longer holds true.

    - The convention '{name} {scenario} {aggregate}'
    - an example 'ccd 2.5°c median'
    """
    # group 1 any characters preceeding the scenario pattern
    # group 2 any digits or the following special characters '-.°c'
    # group 3 the final word at the end of the string

    #                      1          2         3
    variable_pattern = r"(.+)\s([\d\-\.°c]+)\s(\w+)$"
    return pl.col("variable").str.extract(variable_pattern, group)


def tranform_predicted_values(calculation: pl.Expr) -> pl.Expr:
    """some columns have different measure types depending on whether the value is predicted or observed.

    the 'scenario_type' column is used to apply transformations only on predicted values
    which are indicated by being a temperature instead of a date range"""

    predicted_flag = pl.col("scenario_type").str.contains("°c")
    return pl.when(predicted_flag).then(calculation).otherwise("value")


def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    """A single dataframe undergoes the following transformation and validation:
    - ensure the category is District
    - drop redundant columns
    - unpivot the dataframe to turn each column name into a variable
    - extract indicator type (tas, precititation, days)
    - extract scenario type (date range or scenario in °c)
    - map scenario type to be in degrees only
    - extract aggregation type (min, median, max)
    - extract baseline value (the value at 1981-2020)
    - rename columns
    - ensure no null values are present
    """
    # This only works on data including districts only
    assert df["category"].unique().item() == "District"

    # the pattern below extracts the last word, the numeric measure and the prefix. spaces are assumed to sepperate each part
    indicator_type = extract_variable_part(1)
    scenario_type = extract_variable_part(2)
    predicted_flag = scenario_type.str.contains("°c")
    agg_type = extract_variable_part(3)
    degrees_above_baseline = scenario_type.replace_strict(indicator_order)
    district_code = pl.col("code")
    baseline = pl.first("value").over(
        district_code,
        indicator_type,
        agg_type,
        order_by=degrees_above_baseline,
        descending=False,
    )
    df = (
        df.drop("objectid", "name", "category", "shape__area", "shape__length")
        .unpivot(index="code")
        .select(
            district_code=district_code,
            value="value",
            variable="variable",
            indicator_type=indicator_type,
            scenario_type=scenario_type,
            agg_type=agg_type,
            degrees_above_baseline=degrees_above_baseline,
            baseline=baseline,
            predicted_flag=predicted_flag,
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


async def async_get_data(dataset: HasDataset | str, client: httpx.AsyncClient):
    """async function to fetch met office data. Use when downloading multiple datasets"""
    if isinstance(dataset, HasDataset):
        dataset = dataset.dataset
    df = await async_query_without_geometry(
        client,
        MET_OFFICE_MAP_SERVER,
        dataset,
        where=f"CODE in {SURREY_DISTRICTS}",
    )
    return df


def validate_initial_transformation(
    df: pl.DataFrame, metadata: Metadata
) -> pl.DataFrame:
    if df.is_empty():
        raise ValidationError(
            f"Dataset is empty indicating a data sourcing issue for {metadata.source}"
        )

    transformed = (
        transform_data(df)
        .filter(indicator_type=pl.lit(metadata.name_at_source))
        .with_columns(
            indicator_name=pl.lit(metadata.indicator_name),
            measure_name=pl.lit(metadata.measure_name),
        )
    )
    if transformed.is_empty():
        raise ValidationError(
            f"Dataset is empty indicating a metadata issue for {metadata}"
        )
    return transformed


async def async_get_datasets(*datasets: Metadata) -> pl.DataFrame:
    """asynchronously download the data improving the speed of downloads.

    Validation applied to check dataframe is not empty.
    An empty dataframe indicates an issue with the metadata."""
    client = httpx.AsyncClient()
    data: list[pl.DataFrame] = []
    for metadata in datasets:
        df = await async_get_data(metadata, client)
        transformed = validate_initial_transformation(df, metadata)
        data.append(transformed)

    return pl.concat(data, how="vertical_relaxed")


def pivot_with_adjusted_values(df: pl.DataFrame, calculation: pl.Expr) -> pl.DataFrame:
    """following an initial transformation, the variables and information extracted from these can be used to unpivot the data.

    This allows the min max and median values to all be included in a single record.

    Custom calculations must be applied depending on the dataset to ensure the values are consistent across observed or predicted scenarios
    """
    return (
        df.with_columns(value_adjusted=tranform_predicted_values(calculation))
        .pivot(
            "agg_type",
            values=["value_adjusted"],
            index=[
                "district_code",
                "indicator_name",
                "measure_name",
                "scenario_type",
                "degrees_above_baseline",
            ],
        )
        .sort(
            "measure_name",
            "indicator_name",
            "district_code",
            "degrees_above_baseline",
        )
    )


async def get_risk_day_scenarios(*metadata: Metadata) -> pl.DataFrame:
    """the risk day scenarios are a measure of how many days meet temperature thresholds associated with some risk.
    these scenarios require no custom transformation as the number of days observed and the number of days predicted are the same measure

    can be used for testing with
    >>> anyio.run(get_risk_day_scenarios, Metadata(...))
    """
    df = await async_get_datasets(*metadata)
    return pivot_with_adjusted_values(df, pl.col("value"))


async def get_temperature_scenarios(*metadata: Metadata) -> pl.DataFrame:
    """the temperature scenarios are a measure of temperature change.
    these scenarios require transformation as the temperature observed and the predicted temperature increase are different values.
    - the predicted increase is added to the baseline value (ie. the observed value in 1981-2000)

    can be used for testing with
    >>> anyio.run(get_risk_day_scenarios, Metadata(...))
    """
    df = await async_get_datasets(*metadata)
    return pivot_with_adjusted_values(df, pl.col("baseline") + pl.col("value"))


async def get_precipitation_scenarios(*metadata: Metadata) -> pl.DataFrame:
    """the preciptation scenarios are a measure of precipitation change.
    these scenarios require transformation as the precipitation observed and the predicted percentage increase are different values.
    - the predicted percentage increase is converted to a proportion
    - the proportion is multiplied by the absolute baseline value (when temperatures are negative, the increase should not be negative)
    - the value change is added to the baseline value (ie. the observed value in 1981-2000)

    can be used for testing with
    >>> anyio.run(get_risk_day_scenarios, Metadata(...))
    """
    proportion_increase = pl.col("value") / 100
    value_change = proportion_increase * pl.col("baseline").abs()
    df = await async_get_datasets(*metadata)
    return pivot_with_adjusted_values(df, pl.col("baseline") + value_change)


async def _get_climate_projections(
    risk_day_meta: tuple[Metadata, ...] = RISK_DAY_SCENARIOS,
    temperature_meta: tuple[Metadata, ...] = TEMPERATURE_SCENARIOS,
    precipitation_meta: tuple[Metadata, ...] = PRECIPITATION_SCENARIOS,
):
    """utility function enabling all the transformations to run asynchronously with
    >>> anyio.run(get_climate_projections)

      should not need to change the default values.
    They are defined as input parameters to highlight refactoring risks
    """
    return pl.concat(
        [
            await get_risk_day_scenarios(*risk_day_meta),
            await get_temperature_scenarios(*temperature_meta),
            await get_precipitation_scenarios(*precipitation_meta),
        ],
        how="vertical_relaxed",
    )


def get_metadata(
    risk_day_meta: tuple[Metadata, ...] = RISK_DAY_SCENARIOS,
    temperature_meta: tuple[Metadata, ...] = TEMPERATURE_SCENARIOS,
    precipitation_meta: tuple[Metadata, ...] = PRECIPITATION_SCENARIOS,
) -> pl.DataFrame:
    """mirrors the above function in order to match the boilerplate and highlight refactoring risks.
    Users who change the above fucnction MUST also change this functon to ensure consistency.
    """
    return pl.DataFrame([*risk_day_meta, *temperature_meta, *precipitation_meta])


def get_climate_projections(
    cache: Path | None = OUTPUT_DIR / "climate_projections.csv",
) -> pl.DataFrame:
    """utility function returns the data. If the data exists locally, this is used.

    to overwrite data ise the refresh_data function"""
    if cache is None or not cache.exists():
        logger.info("Downloading Data for climate projections...")
        return anyio.run(_get_climate_projections)
    return pl.read_csv(cache)


def refresh_data(output_dir: Path = OUTPUT_DIR) -> None:
    """ETL entry point. Will always overwrite data"""
    meta_dir = output_dir / "metadata"
    if not meta_dir.exists():
        logger.info(f"Creating local store for data at: {output_dir}...")
        meta_dir.mkdir(parents=True)

    surrey_df = get_climate_projections(cache=None)
    meta_df = get_metadata()
    logger.info(f"Writing data to: '{output_dir / 'climate_projections.csv'}'")
    surrey_df.write_csv(output_dir / "climate_projections.csv")
    logger.info(f"Writing metadata to: '{meta_dir / 'climate_projections.csv'}'")
    meta_df.write_csv(meta_dir / "climate_projections.csv")
    logger.success("Data has been refreshed for 'climate_projections.csv'")


if __name__ == "__main__":
    refresh_data()
