from typing import Literal
from pathlib import Path
from jsna_climate_and_environment.config import OUTPUT_DIR
from functools import cache
from jsna_climate_and_environment.datasets.data_model import Metadata
import polars as pl
import fastexcel
import fsspec
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_frame_not_equal
from jsna_climate_and_environment.geography.places import SURREY_DISTRICTS, read_gdb
from loguru import logger


class AccessToNatureMeta(Metadata):
    dataset: str = "Access_to_green_and_blue_space_England"
    source: str = "https://assets.publishing.service.gov.uk/media/69a184aef534e7e99adaeab4/Access_to_green_and_blue_space_England_data_table.ods"
    caveats: str = "scores indicate if more or less households meet this criteria when compared to the national average"


class AccessToWoodlandMeta(Metadata):
    dataset: str = "Access_to_woodland_England"
    source: str = (
        "https://cdn.forestresearch.gov.uk/Access_to_Woodland_in_England_2025.ods"
    )
    caveats: str = "scores indicate if more or less households meet this criteria when compared to the national average"


METADATA: tuple[Metadata, ...] = (
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="bluespace only",
        description="Households that are within 1km walk of more than 550m walkable bluespace",
        rationalle=(
            "Large bodies of water are only considered accessible when it is possible to walk at least 550m along the water's edge. "
            "This may highlight barriers that prevent access or the absence of footpaths when overlaid with water boundaries. "
        ),
    ),
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="greenspace only",
        description="Households that meet any of the government's commitments to access excluding bluespace",
        rationalle=(
            "All access metrics for greenspace can be combined to account for differences in the feasibility of including large greenspaces in urban areas. "
            "Urban areas are less likely to be able to contain 10 hectares of greenspace but may contain numerous smaller areas of greenspace"
        ),
    ),
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="greenspace and bluespace",
        description="Households that meet any of the government's commitments to access",
        rationalle=(
            "100% of households are expected to meet this committment: "
            "Make sure that everyone has access to green or blue spaces within a 15-minute walk from home."
        ),
        caveats="scores indicate if the area meets the committment, a score of 0 indicates the commitment has been reached, negative scores indicate how far from the committment we may be",
    ),
    AccessToNatureMeta(
        name_at_source="doorstep",
        measure_name="greenspace only",
        description="Households that are within 200 meters of more than 0.5 hectares of greenspace",
        rationalle=(
            "0.5 hectares as a perfect square would have sides 70 meters in length, and be slightly smaller than a football pitch. "
            "The 'Swan Center Urban Regeneration project' will be approximately 0.8 hectares and would be an example of greenspace in development considering doorstep access.  "
        ),
    ),
    AccessToNatureMeta(
        name_at_source="local",
        measure_name="greenspace only",
        description="Households that are within 300 meters of more than 2 hectares of greenspace",
        rationalle=(
            "2 hectares as a perfect square would have sides 141 meters in length, "
            "and would be capable of containing a sports field alongside other varieties of activities. "
            "Moor park Nature Reserve is approximately 7.6 hectares and would be an example of one such greenspace in surrey."
        ),
    ),
    AccessToNatureMeta(
        name_at_source="neighbourhood",
        measure_name="greenspace only",
        description="Households that are within 1 kilometer of more than 10 hectares or 550m walkable greenspace",
        rationalle=(
            "550m in a straight line would highlight walking trails that might be excluded when measuring area. "
            "10 hectares as a perfect square would have sides 316 meters in length. "
            "Woking park is approximately 18 hectares and would be an example of one such park in surrey."
        ),
    ),
    AccessToNatureMeta(
        name_at_source="neighbourhood",
        measure_name="greenspace and bluespace",
        description="Households that are within 1 kilometer of more than 10 hectares or 550m walkable greenspace or bluespace",
        rationalle=(
            "550m in a straight line would highlight walking trails that might be excluded when measuring area. "
            "10 hectares as a perfect square would have sides 316 meters in length. "
            "The the Lower Earlswood Lake Circular trail is about 800m and would be an example of the kind of bluespace included in surrey."
        ),
    ),
    AccessToWoodlandMeta(
        name_at_source="Scenario1",
        indicator_name="local small",
        measure_name="woodland",
        description="Households that are within 500 meters of more than 0.5 hectares woodland",
        rationalle=(
            "0.5 hectares as a perfect square would have sides 70 meters in length, and be slightly smaller than a football pitch. "
            "Areas of woodland this size are often privately owned and used for amenities, conservation or firewood. "
            "They may also be used as natural protection from noise polution on busy roadsides. "
        ),
    ),
    AccessToWoodlandMeta(
        name_at_source="Scenario2",
        indicator_name="local medium",
        measure_name="woodland",
        description="Households that are within 500 meters of more than 2 hectares woodland",
        rationalle=(
            "2 hectares as a perfect square would have sides 141 meters in length, you could expect to walk through woodlands meeting this criteria for 2-10 minutes"
            "Westborough wood in Guildford is 5.08 hectares and an example of urban woodland that would fit this criteria. "
        ),
    ),
    AccessToWoodlandMeta(
        name_at_source="Scenario3",
        indicator_name="neighbourhood",
        measure_name="woodland",
        description="Households that are within 1km walk of more than 2 hectares woodland",
        rationalle=(
            "2 hectares as a perfect square would have sides 141 meters in length, you could expect to walk through woodlands meeting this criteria for 2-10 minutes"
            "Westborough wood in Guildford is 5.08 hectares and an example of urban woodland that would fit this criteria. "
        ),
    ),
    AccessToWoodlandMeta(
        name_at_source="Scenario4",
        indicator_name="community",
        measure_name="woodland",
        description="Households that are within 4 kilometers of more than 10 hectares woodland",
        rationalle=(
            "10 hectares as a perfect square would have sides 316 meters in length and would facilitate walks of longer than 10 minutes. "
            "Barley Mow Wood is 10.4 hectares representing the smallest woodland in surrey that would fit this criteria. "
            "Box Hill National Trust site is a larger well known site of 291.53 hectares."
        ),
    ),
)


def percent_calc(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return ((numerator / denominator) * 100).round(2, mode="half_away_from_zero")


def unpivot_numerators(df: pl.DataFrame, measure_name: str) -> pl.DataFrame:
    """For the purpose of aggregation, we ignore the calculated fields in the sheets and use only the numerator and denominator for each Output Area.

    This supports aggregating up in the simplest way possible.

    the numerator columns will start with 'uprn_in' and end with the indicator name as determined by the column

    we ignore column names conatining and as we can reverse engineer these calculations if needed, but I suspect we don't need to.

    the denominator is always the total uprn.

    the urban/rural flag is retained in case it simplifies any downstream analysis."""
    return df.unpivot(
        on=cs.starts_with("uprn_in") & ~cs.contains("_and_"),
        index=["total_uprn", "OA21CD", "LSOA21CD"],
    ).select(
        "OA21CD",
        "LSOA21CD",
        denominator="total_uprn",
        numerator="value",
        indicator=pl.col("variable").str.strip_prefix("uprn_in_"),
        measure=pl.lit(measure_name),
        value=percent_calc(pl.col("value"), pl.col("total_uprn")),
    )


def select_numerators(
    df: pl.DataFrame, measure_name: str, indicator_name: str
) -> pl.DataFrame:
    return df.select(
        OA21CD="OA21",
        LSOA21CD="LSOA21",
        denominator="Household count",
        numerator="Household with access count",
        indicator=pl.lit(indicator_name),
        measure=pl.lit(measure_name),
        value=percent_calc(
            pl.col("Household with access count"), pl.col("Household count")
        ),
    )


# The get_unpivoted_indicators functon is cached in memory to prevent it from running more than needed
# see functols.cache
@cache
def get_unpivoted_indicators() -> pl.DataFrame:
    """combines data from all 3 worksheets and aggregates into a single dataframe in long format."""
    with fsspec.open(
        "https://assets.publishing.service.gov.uk/media/69a184aef534e7e99adaeab4/Access_to_green_and_blue_space_England_data_table.ods"
    ) as file:
        reader = fastexcel.read_excel(file.read())
    green_blue_df_full = reader.load_sheet(3, header_row=4).to_polars()
    blue_df = reader.load_sheet(4, header_row=4).to_polars()
    green_df = reader.load_sheet(5, header_row=4).to_polars()

    # validation logic
    # test which overlapping columns in the green and green/blue sheet are identical
    # in other words, the local and doorstep metrics should always only look at greenspace
    # meanwhile the commitment and neighbourhood metrics should include/exclude bluespace depending on the sheet
    assert_frame_not_equal(
        green_blue_df_full.select(~cs.contains("local", "doorstep")),
        green_df.select(~cs.contains("local", "doorstep")),
        check_row_order=False,
    )
    assert_frame_equal(
        green_blue_df_full.select(~cs.contains("commitment", "neighbourhood")),
        green_df.select(~cs.contains("commitment", "neighbourhood")),
        check_row_order=False,
    )
    # if the validation passes, we can safely assume local and doorstep indicators never include both greenspace and bluespace
    # by default these are included in the greenspace only data and we just use them once for simplicity
    green_blue_df = green_blue_df_full.select(~cs.contains("local", "doorstep"))

    # Forestry data
    with fsspec.open(
        "https://cdn.forestresearch.gov.uk/Access_to_Woodland_in_England_2025.ods"
    ) as file:
        forestry_reader = fastexcel.read_excel(file.read())
    scenario1 = forestry_reader.load_sheet("Scenario1", header_row=5).to_polars()
    scenario2 = forestry_reader.load_sheet("Scenario2", header_row=5).to_polars()
    scenario3 = forestry_reader.load_sheet("Scenario3", header_row=5).to_polars()
    scenario4 = forestry_reader.load_sheet("Scenario4", header_row=5).to_polars()

    long_df = pl.concat(
        [
            unpivot_numerators(green_blue_df, "greenspace and bluespace"),
            unpivot_numerators(green_df, "greenspace only"),
            unpivot_numerators(blue_df, "bluespace only"),
            select_numerators(scenario1, "woodland", "local small"),
            select_numerators(scenario2, "woodland", "local medium"),
            select_numerators(scenario3, "woodland", "neighbourhood"),
            select_numerators(scenario4, "woodland", "community"),
        ]
    )
    return long_df


def get_metadata(metadata: tuple[Metadata, ...] = METADATA) -> pl.DataFrame:
    """in addition to the standard metadata, comparison values are included for this dataset.

    This requires the original data to be filtered and grouped for a surrey and a england comparison
    """
    # The get_unpivoted_indicators functon is cached in memory to prevent it from running twice
    # see functols.cache
    long_df = get_unpivoted_indicators()
    surrey_df = long_df.filter(
        pl.col("LSOA21CD").is_in(read_gdb(layer="lsoa")["lsoa21_code"])
    )

    england_val = long_df.group_by(
        indicator_name="indicator", measure_name="measure"
    ).agg(
        england_numerator=pl.sum("numerator"),
        england_denominator=pl.sum("denominator"),
        england_value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
    )

    surrey_val = surrey_df.group_by(
        indicator_name="indicator", measure_name="measure"
    ).agg(
        surrey_numerator=pl.sum("numerator"),
        surrey_denominator=pl.sum("denominator"),
        surrey_value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
    )
    return (
        pl.DataFrame(metadata)
        .join(surrey_val, on=["indicator_name", "measure_name"], validate="1:1")
        .join(england_val, on=["indicator_name", "measure_name"], validate="1:1")
    )


def group_surrey_data(area_code: Literal["LAD25CD", "LSOA21CD"]):
    total_over_indicator = percent_calc(
        pl.sum("numerator").over("indicator", "measure"),
        pl.sum("denominator").over("indicator", "measure"),
    )
    comparison_metric = (
        (
            pl.when(
                indicator=pl.lit("commitment"),
                measure=pl.lit("greenspace and bluespace"),
            )
            .then(pl.col("value") - 100)
            .otherwise(pl.col("value") - total_over_indicator)
        )
        // 10
    ).cast(pl.Int32)
    return (
        get_unpivoted_indicators()
        .with_columns(
            is_surrey=pl.col("LSOA21CD").is_in(read_gdb(layer="lsoa")["lsoa21_code"])
        )
        .group_by("indicator", "measure", area_code)
        .agg(
            pl.all("is_surrey"),
            pl.sum("numerator"),
            pl.sum("denominator"),
            value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
        )
        .sort("indicator")
        .with_columns(score=comparison_metric)
        .filter("is_surrey")
        .drop("is_surrey")
    )


def get_access_to_nature_lsoa(
    cache: Path | None = OUTPUT_DIR / "access_to_nature_lsoa.csv",
) -> pl.DataFrame:
    """utility function returns the data. If the data exists locally, this is used.

    to overwrite data use the refresh_data function"""
    if cache is None or not cache.exists():
        logger.info("Downloading Data for access to nature...")
        return group_surrey_data("LSOA21CD")
    return pl.read_csv(cache)


def refresh_data(output_dir: Path = OUTPUT_DIR) -> None:
    """ETL entry point. Will always overwrite data"""
    meta_dir = output_dir / "metadata"
    if not meta_dir.exists():
        logger.info(f"Creating local store for data at: {output_dir}...")
        meta_dir.mkdir(parents=True)

    surrey_lsoa_df = get_access_to_nature_lsoa(cache=None)
    meta_df = get_metadata()
    logger.info(f"Writing data to: '{output_dir / 'access_to_nature_lsoa.csv'}'")
    surrey_lsoa_df.write_csv(output_dir / "access_to_nature_lsoa.csv")
    logger.info(f"Writing metadata to: '{meta_dir / 'access_to_nature.csv'}'")
    meta_df.write_csv(meta_dir / "access_to_nature.csv")
    logger.success("Data has been refreshed for 'access_to_nature_lsoa.csv'")


if __name__ == "__main__":
    refresh_data()
