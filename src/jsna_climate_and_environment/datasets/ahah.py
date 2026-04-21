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
from jsna_climate_and_environment.config import DATA_DIR, OUTPUT_DIR
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
        measure_name="overall_index",
        name_at_source="ahah",
        indicator_name="overall_access",
        description="Scored by combining all domains - lower score = healthier",
        rationalle="This value aggregates all inputs. It is an indication of the health of the built environment overall",
        caveats=None,
    ),
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
            "Nitrogen Dioxide is mainly produced by burning fossil fuels (vehicles, power plants). "
            "It causes respiratory issues, including airway inflammation and asthma, and contributes to smog, acid rain, and ozone formation."
        ),
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="pm10",
        indicator_name="exposure to dust",
        description="Scored by average concentration - lower concentration = healthier",
        rationalle=(
            "The average average concentration of coarse particulate matter (pm10) in this area. "
            "This includes dust, construction, agriculture, and vehicle emissions. "
            "These particles penetrate the throat and lungs, causing respiratory issues, asthma, and cardiovascular strain."
        ),
    ),
    AhahMetadata(
        measure_name="air_quality_domain",
        name_at_source="so2",
        indicator_name="exposure to coarse particulate matter",
        description="Scored by average concentration - lower concentration = healthier",
        rationalle=(
            "The average average concentration of coarse particulate matter (pm10) in this area. "
            "This includes dust, construction, agriculture, and vehicle emissions. "
            "These particles penetrate the throat and lungs, causing respiratory issues, asthma, and cardiovascular strain."
        ),
    ),
)

domain_map: dict[str, str] = {
    "gp": "health",
    "bluespace": "greenspace_bluespace",
    "dentist": "health",
    "fast_food": "retail",
    "gambling": "retail",
    "greenspace_active": "greenspace_bluespace",
    "hospital": "health",
    "leisure": "health",
    "pharmacy": "health",
    "pub_bar": "retail",
    "tobacco": "retail",
    "no2": "air_quality",
    "pm10": "air_quality",
    "so2": "air_quality",
    "greenspace": "greenspace_bluespace",
    "ahah": "overall_index",
    "domain_h": "health",
    "domain_g": "greenspace_bluespace",
    "domain_e": "air_quality",
    "domain_r": "retail",
}


ranking_system = "lower (better) rank = lower (better) decile"
travel_time_ranking = f"scored by travel time in minutes - lower travel time = better access = {ranking_system}"
shortest_distance_ranking = (
    "scored by travel time in minutes - shorter distance = better access"
)
longest_distance_ranking = (
    "scored by travel time in minutes - longer distance = healthier"
)
ndvi_ranking = "scored by NDVI - higher NDVI = more vegetation"
polution_ranking = (
    "scored by concentration of polutant in air - lower pollution = healthier"
)
aggregation_ranking = "scored by aggregation - lower score = healthier"

score_description_map: dict[str, str] = {
    "gp": travel_time_ranking,
    "bluespace": shortest_distance_ranking,
    "dentist": travel_time_ranking,
    "fast_food": longest_distance_ranking,
    "gambling": longest_distance_ranking,
    "greenspace_active": shortest_distance_ranking,
    "hospital": shortest_distance_ranking,
    "leisure": shortest_distance_ranking,
    "pharmacy": shortest_distance_ranking,
    "pub_bar": longest_distance_ranking,
    "tobacco": longest_distance_ranking,
    "no2": polution_ranking,
    "pm10": polution_ranking,
    "so2": polution_ranking,
    "greenspace": ndvi_ranking,
    "ahah": aggregation_ranking,
    "domain_h": aggregation_ranking,
    "domain_g": aggregation_ranking,
    "domain_e": aggregation_ranking,
    "domain_r": aggregation_ranking,
}


def transform_local_file(path: Path = DATA_DIR / "ahah_v5.csv") -> pl.DataFrame:
    aha: DataFrame = pl.read_csv(path).rename(str.lower)

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

    return aha_long.with_columns(
        domain=pl.col("indicator").replace_strict(domain_map),
        decile=(pl.col("percentile") % 10).cast(pl.Int32) + 1,
        description=pl.col("indicator").replace_strict(score_description_map),
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


def get_data(
    output_file: Path = OUTPUT_DIR / "ahah.csv", overwrite: bool = False
) -> pl.DataFrame:
    if output_file.exists() and not overwrite:
        return pl.read_csv(output_file)

    link_df = pl.from_pandas(read_gdb("linking"))
    surrey_df = transform_local_file().filter(
        pl.col("lsoa21cd").is_in(link_df["lsoa21_code"].to_list())
    )
    surrey_df.write_csv(output_file)
    return surrey_df


if __name__ == "__main__":
    df = get_data(overwrite=True)
    print(df)
