from typing import Literal
from jsna_climate_and_environment.datasets.data_model import Metadata
from pathlib import Path
from jsna_climate_and_environment.config import OUTPUT_DIR
from jsna_climate_and_environment.geography.places import read_gdb
import polars as pl
import polars.selectors as cs
from fastexcel import read_excel
import fsspec
import re
from collections.abc import Iterable
from loguru import logger


class IodScoreMetadata(Metadata):
    source: Literal[
        "https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
    ] = "https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
    dataset: Literal["2025_indicies_of_deprivation"] = "2025_indicies_of_deprivation"
    description: str = "Higher rank/recile = more deprived"
    rationalle: str = "Domains and subdomains of deprivation give an indication of the types of deprivation"


IMD_INDICATOR: Metadata = IodScoreMetadata(
    measure_name="Overall Index",
    name_at_source="index_of_multiple_deprivation_imd",
    indicator_name="Index of Multiple Deprivation",
    description="Higher rank/recile = more deprived",
    rationalle="Key neighborhoods are identified by an IMD decile of 2 or 3",
)

IOD_INDICATORS: tuple[Metadata, ...] = (
    IMD_INDICATOR,
    IodScoreMetadata(
        measure_name="Living Environment",
        name_at_source="living_environment",
        indicator_name="Living Environment Domain",
    ),
    IodScoreMetadata(
        measure_name="Living Environment",
        name_at_source="indoors_sub_domain",
        indicator_name="Indoors Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Living Environment",
        name_at_source="outdoors_sub_domain",
        indicator_name="Outdoors Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Income",
        name_at_source="income",
        indicator_name="Income Domain",
    ),
    IodScoreMetadata(
        measure_name="Income",
        name_at_source="income_deprivation_affecting_older_people",
        indicator_name="Income Deprivation Affecting Older People",
    ),
    IodScoreMetadata(
        measure_name="Income",
        name_at_source="income_deprivation_affecting_children_index_idaci",
        indicator_name="Income Deprivation Affecting Older People",
    ),
    IodScoreMetadata(
        measure_name="Education",
        name_at_source="education_skills_and_training",
        indicator_name="Education Domain",
    ),
    IodScoreMetadata(
        measure_name="Education",
        name_at_source="adult_skills_sub_domain",
        indicator_name="Adult Skills Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Education",
        name_at_source="children_and_young_people_sub_domain",
        indicator_name="Children And Young People Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Barriers",
        name_at_source="barriers_to_housing_and_services",
        indicator_name="Education Domain",
    ),
    IodScoreMetadata(
        measure_name="Barriers",
        name_at_source="wider_barriers_sub_domain",
        indicator_name="Wider Barriers Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Barriers",
        name_at_source="geographical_barriers_sub_domain",
        indicator_name="Geographical Barriers Sub Domain",
    ),
    IodScoreMetadata(
        measure_name="Crime",
        name_at_source="crime",
        indicator_name="Crime Domain",
    ),
    IodScoreMetadata(
        measure_name="Health",
        name_at_source="health_deprivation_and_disability",
        indicator_name="Health Domain",
    ),
    IodScoreMetadata(
        measure_name="Employment",
        name_at_source="employment",
        indicator_name="Employment Domain",
    ),
)


def _iter_excel_sheet(source: str) -> Iterable[pl.DataFrame]:
    with fsspec.open(source) as file:
        reader = read_excel(file.read())

        for sheet_name in reader.sheet_names[1:]:
            yield (
                reader.load_sheet(sheet_name)
                .to_polars()
                .with_columns(sheet_name=pl.lit(iod_column_cleaner(sheet_name)))
            )


def iod_column_cleaner(col_name: str) -> str:
    remove_symbols = re.sub(r"[^\w\s]", " ", col_name.lower())
    replace_whitespace = re.sub(r"\s+", "_", remove_symbols.strip())
    concat_digits = re.sub(r"(\d)_(\d)", r"\g<1>\g<2>", replace_whitespace.strip("_"))
    return concat_digits


def _transform_underlying_indicators(df: pl.DataFrame) -> pl.DataFrame:
    domain = pl.col("sheet_name").str.strip_prefix("iod25_").str.strip_suffix("_domain")
    return (
        df.rename(iod_column_cleaner)
        .unpivot(index=cs.string(), variable_name="indicator")
        .select(
            lsoa21cd="lsoa_code_2021",
            population_estimate_2022=pl.lit(None),
            domain=domain,
            indicator="indicator",
            score="value",
            rank=pl.lit(None),
            decile=pl.lit(None),
        )
    )


def get_imd_domains(
    metadata: tuple[Metadata, ...],
) -> pl.DataFrame:
    domain_map = {i.name_at_source: i.measure_name for i in metadata}
    indicator_map = {x.name_at_source: x.indicator_name for x in metadata}
    index_type = pl.col("variable").str.extract("rank|score|decile", 0)
    suffix_detail = pl.col("variable").str.extract("_(rank|score|decile).*", 0)
    indicator = pl.col("variable").str.strip_suffix(suffix_detail)

    index_cols = (
        "lsoa_code_2021",
        "total_population_mid_2022",
    )

    return (
        pl.read_csv(
            "https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
        )
        .rename(iod_column_cleaner)
        .unpivot(index=index_cols, on=cs.numeric() ^ cs.contains("mid_2022"))
        .with_columns(index_type=index_type, indicator=indicator)
        .pivot("index_type", values="value", index=(*index_cols, "indicator"))
        .filter(pl.col("indicator").is_in(indicator_map.keys()))
        .select(
            lsoa21cd="lsoa_code_2021",
            population_estimate_2022="total_population_mid_2022",
            domain=pl.col("indicator").replace_strict(domain_map),
            indicator=pl.col("indicator").replace_strict(indicator_map),
            score="score",
            rank="rank",
            decile="decile",
        )
    )


def get_metadata(metadata: tuple[Metadata, ...] = (IMD_INDICATOR,)) -> pl.DataFrame:
    """mirrors the above function in order to match the boilerplate and highlight refactoring risks.
    Users who change the above fucnction MUST also change this functon to ensure consistency.
    """
    return pl.DataFrame([*metadata])


def get_data(metadata: tuple[Metadata, ...] = (IMD_INDICATOR,)) -> pl.DataFrame:
    link_df = pl.from_pandas(read_gdb("nspl"))
    return get_imd_domains(metadata).filter(
        pl.col("lsoa21cd").is_in(link_df["lsoa21_code"].to_list())
    )


def get_surrey_imd(
    cache: Path | None = OUTPUT_DIR / "imd.csv",
) -> pl.DataFrame:
    """utility function returns the data. If the data exists locally, this is used.

    to overwrite data use the refresh_data function"""
    if cache is None or not cache.exists():
        logger.info("Downloading Data for Access to healthy assets and hazards...")
        return get_data()
    return pl.read_csv(cache)


def refresh_data(output_dir: Path = OUTPUT_DIR) -> None:
    """ETL entry point. Will always overwrite data"""
    meta_dir = output_dir / "metadata"
    if not meta_dir.exists():
        logger.info(f"Creating local store for data at: {output_dir}...")
        meta_dir.mkdir(parents=True)

    surrey_df = get_surrey_imd(cache=None)
    meta_df = get_metadata()
    logger.info(f"Writing data to: '{output_dir / 'imd.csv'}'")
    surrey_df.write_csv(output_dir / "imd.csv")
    logger.info(f"Writing metadata to: '{meta_dir / 'imd.csv'}'")
    meta_df.write_csv(meta_dir / "imd.csv")
    logger.success("Data has been refreshed for 'imd.csv'")


if __name__ == "__main__":
    refresh_data()
