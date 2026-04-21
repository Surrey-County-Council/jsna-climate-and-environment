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
from jsna_climate_and_environment.geography.places import SURREY_DISTRICTS
from loguru import logger


class AccessToNatureMeta(Metadata):
    dataset: str = "Access_to_green_and_blue_space_England"
    source: str = "https://assets.publishing.service.gov.uk/media/69a184aef534e7e99adaeab4/Access_to_green_and_blue_space_England_data_table.ods"


METADATA: tuple[Metadata, ...] = (
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="bluespace only",
        description="Households that are within 1km walk of more than 500m walkable bluespace",
        rationalle=(
            "Fewer households will have bluespace within a 15 minute walk. "
            "Areas are expected to be clustered together, gaps inside clusters are indicitive of barriers to access. "
            "It's expected people would only make a 15 minute walk if there was a large area of bluespace. "
        ),
    ),
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="greenspace only",
        description="Households that meet any of the government's commitments to access excluding bluespace",
        rationalle=(
            "Even where households can access bluespace, they will still benefit from accessible greenspace. "
            "Areas are expected to have a more uniform distribution with urban areas more likely to have poorer access. "
        ),
    ),
    AccessToNatureMeta(
        name_at_source="commitment",
        measure_name="greenspace or bluespace",
        description="Households that meet any of the government's commitments to access",
        rationalle=(
            "100% of households are expected to meet this committment: "
            "Make sure that everyone has access to green or blue spaces within a 15-minute walk from home."
        ),
    ),
    AccessToNatureMeta(
        name_at_source="doorstep",
        measure_name="greenspace only",
        description="Households that are within 200 meters of more than 0.5 hectares of greenspace",
        rationalle=(
            "0.5 hectares as a perfect square would have sides 70 meters in length, and be slightly smaller than a football pitch. "
            "Dense urban areas often have more limited space available and greenspace is often smaller. "
            "The 'Swan Center Urban Regeneration project' will be approximately 0.8 hectares and would be an example of greenspace in development considering doorstep access.  "
            "Clusters of areas with good doorstep access might help indicate areas where more options for greenspace access exist."
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
        description="Households that are within 1 kilometer of more than 10 minutes walkable greenspace",
        rationalle=(
            "550m in a straight line would highlight walking trails that might be excluded when measuring area. "
            "10 hectares as a perfect square would have sides 316 meters in length. "
            "Woking park is approximately 18 hectares and would be an example of one such park in surrey."
        ),
    ),
    AccessToNatureMeta(
        name_at_source="neighbourhood",
        measure_name="greenspace or bluespace",
        description="Households that are within 1 kilometer of more than 10 minutes walkable greenspace or bluespace",
        rationalle=(
            "550m in a straight line would highlight walking trails that might be excluded when measuring area. "
            "10 hectares as a perfect square would have sides 316 meters in length. "
            "The the Lower Earlswood Lake Circular trail is about 800m and would be an example of the kind of bluespace included in surrey."
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
        index=cs.ends_with("CD") | cs.by_name("total_uprn", "urban_rural_flag"),
        value_name="numerator",
        variable_name="indicator",
    ).with_columns(
        denominator="total_uprn",
        indicator=pl.col("indicator").str.strip_prefix("uprn_in_"),
        measure=pl.lit(measure_name),
        value=percent_calc(pl.col("numerator"), pl.col("total_uprn")),
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

    long_df = pl.concat(
        [
            unpivot_numerators(green_blue_df, "greenspace or bluespace"),
            unpivot_numerators(green_df, "greenspace only"),
            unpivot_numerators(blue_df, "bluespace only"),
        ]
    )
    return long_df


@cache
def get_surrey_unpivoted_indicators() -> pl.DataFrame:
    return get_unpivoted_indicators().filter(pl.col("LAD25CD").is_in(SURREY_DISTRICTS))


def get_metadata(metadata: tuple[Metadata, ...] = METADATA):
    """in addition to the standard metadata, comparison values are included for this dataset.

    This requires the original data to be filtered and grouped for a surrey and a england comparison
    """
    # The get_unpivoted_indicators functon is cached in memory to prevent it from running twice
    # see functols.cache
    long_df = get_unpivoted_indicators()
    surrey_df = get_surrey_unpivoted_indicators()

    england_val = long_df.group_by("indicator", "measure").agg(
        england_numerator=pl.sum("numerator"),
        england_denominator=pl.sum("denominator"),
        england_value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
    )

    surrey_val = surrey_df.group_by("indicator", "measure").agg(
        surrey_numerator=pl.sum("numerator"),
        surrey_denominator=pl.sum("denominator"),
        surrey_value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
    )
    return (
        pl.DataFrame(metadata)
        .join(surrey_val, on=["indicator", "measure"], validate="1:1")
        .join(england_val, on=["indicator", "measure"], validate="1:1")
    )


def group_surrey_data(area_code: Literal["LAD25CD", "LSOA21CD"]):
    return (
        get_surrey_unpivoted_indicators()
        .group_by("indicator", "measure", area_code)
        .agg(
            pl.sum("numerator"),
            pl.sum("denominator"),
            value=percent_calc(pl.sum("numerator"), pl.sum("denominator")),
        )
        .sort("indicator")
    )


def get_access_to_nature_district(
    cache: Path | None = OUTPUT_DIR / "access_to_nature_district.csv",
) -> pl.DataFrame:
    """utility function returns the data. If the data exists locally, this is used.

    to overwrite data use the refresh_data function"""
    if cache is None or not cache.exists():
        logger.info("Downloading Data for access to nature...")
        return group_surrey_data("LAD25CD")
    return pl.read_csv(cache)


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

    surrey_district_df = get_access_to_nature_district(cache=None)
    surrey_lsoa_df = get_access_to_nature_lsoa(cache=None)
    meta_df = get_metadata()
    logger.info(f"Writing data to: '{output_dir / 'access_to_nature_district.csv'}'")
    surrey_district_df.write_csv(output_dir / "access_to_nature_district.csv")
    logger.info(f"Writing data to: '{output_dir / 'access_to_nature_lsoa.csv'}'")
    surrey_lsoa_df.write_csv(output_dir / "access_to_nature_lsoa.csv")
    logger.info(f"Writing metadata to: '{meta_dir / 'access_to_nature.csv'}'")
    meta_df.write_csv(meta_dir / "access_to_nature.csv")
    logger.success("Data has been refreshed for 'access_to_nature_{district/lsoa}.csv'")
