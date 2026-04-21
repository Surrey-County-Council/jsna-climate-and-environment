"""Access to the data requires registration. go to https://apps.cdrc.ac.uk/datasetportal/Identity/Account/Register

Once registered generate an API key and save it as an env variable 'CDRC_API_KEY'

"""

from pydantic import Field

from typing import Literal

from jsna_climate_and_environment.datasets.data_model import Metadata
from jsna_climate_and_environment.geography.places import read_gdb
from polars.dataframe.frame import DataFrame
from polars.expr.expr import Expr
from pathlib import Path
import polars as pl
from jsna_climate_and_environment.config import OUTPUT_DIR
from loguru import logger


class AhahMetadata(Metadata):
    source: Literal[
        "https://data.geods.ac.uk/dataset/8fa47a8b-345c-438d-a8c3-0ac758ee12be/resource/db102d9c-1515-43c4-910e-9700d13fed24/download/ahah_v5.csv"
    ] = "https://data.geods.ac.uk/dataset/8fa47a8b-345c-438d-a8c3-0ac758ee12be/resource/db102d9c-1515-43c4-910e-9700d13fed24/download/ahah_v5.csv"
    dataset: Literal["ahah_v5"] = "ahah_v5"
    indicator_name: str = Field(
        default_factory=lambda data: f"access_to_{data['name_at_source']}"
    )
    caveats: str | None = (
        "Where drive time is measured from the center of a postcode this assumes access to a private vehicle. "
        "A postcode typically contains around 15 addresses, though they can contain up to 100. Population standardisation has not been completed. "
        "Drive times from a postcode center to its edge vary significantly based on urban density, "
        "typically ranging from 2-5 minutes in cities to over 15-20 minutes in rural areas. "
        "For greenspace/bluespace in particular, the access to nature dashboard provides a more robust measure."
    )


RISK_DAY_SCENARIOS: tuple[Metadata, ...] = (
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="gp",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest gp surgery",
    ),
    AhahMetadata(
        measure_name="greenspace_bluespace_domain",
        name_at_source="bluespace",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest bluespace",
    ),
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="dentist",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest dentist",
    ),
    AhahMetadata(
        measure_name="retail_domain",
        name_at_source="fast_food",
        description="Scored by drive time in minutes - longer drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest fast food outlet",
    ),
    AhahMetadata(
        measure_name="retail_domain",
        name_at_source="gambling",
        description="Scored by drive time in minutes - longer drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest gambling outlet",
    ),
    AhahMetadata(
        measure_name="greenspace_bluespace_domain",
        name_at_source="greenspace_active",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle=(
            "The average travel time from each postcode centroid in this area to the closest accessible greenspace. "
            "(accessible greenspace is defined by ordinance survey and includes parks, "
            "playing fields, sports facilities and other publically accessible greenspace.)"
        ),
    ),
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="hospital",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle=(
            "The average travel time from each postcode centroid in this area to the closest hospital. "
            "(Hospitals are any NHS trust site and do not necessarily include an A&E. These sites may be specialist units.)"
        ),
    ),
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="leisure",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle="The average travel time from each postcode centroid in this area to the closest sports and fitness facility.",
    ),
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="pharmacy",
        description="Scored by drive time in minutes - shorter drive time = healthier",
        rationalle=(
            "The average travel time from each postcode centroid in this area to the closest pharmacy. "
            "This may produce different results from the authoritative Pharmaceutical Needs Assessments completed by each Local Authority."
        ),
    ),
    AhahMetadata(
        measure_name="retail_domain",
        name_at_source="pub_bar",
        description="Scored by drive time in minutes - longer drive time = healthier",
        rationalle=(
            "The average travel time from each postcode centroid in this area to the closest pub or bar. "
            "This includes lisenced premises such as resteraunts or cafe's only when these premices are legally permitted to sell alcohol. "
            "Do not drink and drive!"
        ),
    ),
    AhahMetadata(
        measure_name="retail_domain",
        name_at_source="tobacco",
        description="Scored by drive time in minutes - longer drive time = healthier",
        rationalle=(
            "The average travel time from each postcode centroid in this area to the closest outlet selling tobacco or vape products. "
            "(This includes any outlets which sell tobacco products such as supermarkets, off-licences and vape shops.)"
        ),
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="no2",
        indicator_name="exposure to smog",
        description="Scored by average concentration - lower concentration = healthier",
        rationalle=(
            "The average average concentration of nitrogen dioxide in this area. "
            "It is primarily associated with burning fossil fuels (vehicles, power plants). "
            "It causes respiratory issues, including airway inflammation and asthma, and contributes to smog, acid rain, and ozone formation."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="pm10",
        indicator_name="exposure to dust",
        description="Scored by average concentration - lower concentration = healthier",
        rationalle=(
            "The average average concentration of coarse particulate matter (pm10) in this area. "
            "It is primarily associated with construction, agriculture, and vehicle emissions. "
            "These particles penetrate the throat and lungs, causing respiratory issues, asthma, and cardiovascular strain."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="so2",
        indicator_name="exposure to Sulfur dioxide",
        description="Scored by average concentration - lower concentration = healthier",
        rationalle=(
            "The average average concentration of Sulfur dioxide in this area. "
            "Sulfur dioxide is a toxic gas that causes acid rain. It is primarily associated with industrial polutants. "
            "It has a pungent odor and causes respitory issues."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="greenspace_bluespace_domain",
        name_at_source="greenspace",
        indicator_name="Median Normalized Difference Vegetation Index (NDVI) value",
        description="Scored by analysing light sensors. Values range between -1 (no green) and 1 (all green)",
        rationalle=(
            "The literal 'greenness' of an area based on satelite imagery. "
            "This means grey (urban), blue (water) or white (snow) areas will have low scores and the measure can be impacted by cloud cover. "
            "NDVI has strong associacions with improved physical activity, healthy birth weight and low mortality."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="overall_index",
        name_at_source="ahah",
        indicator_name="Overall health of the environment based on access risks",
        description="Scored by combining all domain scores - lower scores = healthier",
        rationalle=(
            "The overall health is an indication of how each domain intersects and can be used as a holistic view. "
            "A holistic view is useful for identifying areas that are most at risk when considering all domains. "
            "It might be used alongside the indicies of deprivation to identify priority populations based on an alternative metric than deprivation."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="health_domain",
        name_at_source="domain_h",
        indicator_name="Overall access to health assets",
        description="Scored by combining all indicators in the health domain - lower scores = healthier",
        rationalle=(
            "The overall access to health assets is an indication of access to various kinds of health sites. "
            "The measure of access for all underlying indicators is drive time in minutes to the closest health site of it's kind. "
            "It should be seen as a measure of convenience rather than density."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="greenspace_bluespace",
        name_at_source="domain_g",
        indicator_name="Overall access to greenspace and bluespace assets",
        description="Scored by combining all indicators in the greenspace bluespace domain - lower scores = healthier",
        rationalle=(
            "The overall access to greenspace and bluespace assets is an indication of the health of nature in the area. "
            "The measure of access for underlying indicators includes both drive time and overall green-ness. "
            "It balances the density of greenspace against public access and bluespace."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="domain_e",
        indicator_name="Overall access to air quality hazards",
        description="Scored by combining all indicators in the air quality domain - lower scores = healthier",
        rationalle=(
            "The overall access to air quality hazards is a simplified indication of air quality. "
            "The measure of access for all underlying indicators is concentration. "
            "It can identify hotspots where different air quality hazards intersect."
        ),
        caveats=None,
    ),
    AhahMetadata(
        measure_name="retail_domain",
        name_at_source="domain_r",
        indicator_name="Overall access to retail hazards",
        description="Scored by combining all indicators in the retail domain - lower scores = healthier",
        rationalle=(
            "The overall access to retail hazards is an indication of access across multiple retail areas. "
            "The measure of access for all underlying indicators is drive time in minutes to the closest retail hazard. "
            "It should be seen as a measure of convenience rather than density."
        ),
        caveats=None,
    ),
)


def download_aha(
    url: Literal[
        "https://data.geods.ac.uk/dataset/8fa47a8b-345c-438d-a8c3-0ac758ee12be/resource/db102d9c-1515-43c4-910e-9700d13fed24/download/ahah_v5.csv"
    ] = "https://data.geods.ac.uk/dataset/8fa47a8b-345c-438d-a8c3-0ac758ee12be/resource/db102d9c-1515-43c4-910e-9700d13fed24/download/ahah_v5.csv",
):
    return pl.read_csv(url).rename(str.lower)


def transform_ahah(
    risk_day_meta: tuple[Metadata, ...] = RISK_DAY_SCENARIOS,
) -> pl.DataFrame:
    aha: DataFrame = download_aha()
    domain_map = {x.name_at_source: x.measure_name for x in risk_day_meta}
    indicator_map = {x.name_at_source: x.indicator_name for x in risk_day_meta}

    # unpivoting the data puts the column names into a default column named variable
    # it's possible to extract the rnk and pct from column names, the scores are un-labled
    index_type: Expr = (
        pl.col("variable")
        .str.extract(r"rnk\z|pct\z", 0)
        .fill_null("score")
        .replace({"rnk": "rank", "pct": "percentile"})
        .alias("index_type")
    )
    indicator = (
        pl.col("variable")
        .str.replace(r"_rnk\z|_pct\z", "")
        .str.strip_prefix(
            "vape_"
        )  # vape_tobacco and tobacco are labled differently as columns but part of the same indicator
        .alias("indicator")
    )

    # pivoting the data after extracting the rank pct and score creates a unique record for each indicator and lsoa
    # each unique record has 3 kinds of value
    aha_long = (
        aha.unpivot(index=["lsoa21cd"])
        .with_columns(index_type, indicator)
        .pivot(
            "index_type",
            values="value",
            index=["lsoa21cd", "indicator"],
        )
    )

    # using replace strict will cause errors if the metadata fails to cover all values in the dataset
    # as a result this serves as metadata validation
    return aha_long.with_columns(
        indicator=pl.col("indicator").replace_strict(indicator_map),
        domain=pl.col("indicator").replace_strict(domain_map),
        decile=(pl.col("percentile") % 10).cast(pl.Int32) + 1,
    )


def get_metadata(
    risk_day_meta: tuple[Metadata, ...] = RISK_DAY_SCENARIOS,
) -> pl.DataFrame:
    """mirrors the above function in order to match the boilerplate and highlight refactoring risks.
    Users who change the above fucnction MUST also change this functon to ensure consistency.
    """
    return pl.DataFrame([*risk_day_meta])


def get_data(
    risk_day_meta: tuple[Metadata, ...] = RISK_DAY_SCENARIOS,
) -> pl.DataFrame:
    link_df = pl.from_pandas(read_gdb("linking"))
    surrey_df = transform_ahah(risk_day_meta).filter(
        pl.col("lsoa21cd").is_in(link_df["lsoa21_code"].to_list())
    )
    return surrey_df


def get_surrey_ahah(
    cache: Path | None = OUTPUT_DIR / "ahah.csv",
) -> pl.DataFrame:
    """utility function returns the data. If the data exists locally, this is used.

    to overwrite data use the refresh_data function"""
    if cache is None or not cache.exists():
        logger.info("Downloading Data for climate projections...")
        return get_data()
    return pl.read_csv(cache)


def refresh_data(output_dir: Path = OUTPUT_DIR) -> None:
    """ETL entry point. Will always overwrite data"""
    meta_dir = output_dir / "metadata"
    if not meta_dir.exists():
        logger.info(f"Creating local store for data at: {output_dir}...")
        meta_dir.mkdir(parents=True)

    surrey_df = get_surrey_ahah(cache=None)
    meta_df = get_metadata()
    logger.info(f"Writing data to: '{output_dir / 'ahah.csv'}'")
    surrey_df.write_csv(output_dir / "ahah.csv")
    logger.info(f"Writing metadata to: '{meta_dir / 'ahah.csv'}'")
    meta_df.write_csv(meta_dir / "ahah.csv")
    logger.success("Data has been refreshed for 'ahah.csv'")


if __name__ == "__main__":
    refresh_data()
