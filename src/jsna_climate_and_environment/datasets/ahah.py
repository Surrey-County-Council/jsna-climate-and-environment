from jsna_climate_and_environment.geography.places import read_gdb
from polars.dataframe.frame import DataFrame
from polars.expr.expr import Expr
from pathlib import Path
import polars as pl
from jsna_climate_and_environment.config import DATA_DIR, OUTPUT_DIR

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
    f"scored by distance - shorter distance = better access = {ranking_system}"
)
longest_distance_ranking = (
    f"scored by distance - longer distance = healthier = {ranking_system}"
)
ndvi_ranking = f"scored by NDVI - higher NDVI = more vegetation = {ranking_system}"
polution_ranking = f"scored by concentration of polutant in air - lower pollution = healthier = {ranking_system}"
aggregation_ranking = f"scored by aggregation - lower (better) score = {ranking_system}"

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
