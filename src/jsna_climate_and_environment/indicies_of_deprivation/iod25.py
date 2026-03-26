import polars as pl
import polars.selectors as cs
from fastexcel import read_excel
import fsspec
import re
from collections.abc import Iterable

domain_map = {
    "income_deprivation_affecting_older_people_idaopi": "income",
    "children_and_young_people_sub_domain": "education",
    "barriers_to_housing_and_services": "barriers",
    "living_environment": "living_env",
    "indoors_sub_domain": "living_env",
    "adult_skills_sub_domain": "education",
    "wider_barriers_sub_domain": "barriers",
    "income_deprivation_affecting_children_index_idaci": "income",
    "income": "income",
    "geographical_barriers_sub_domain": "barriers",
    "index_of_multiple_deprivation_imd": "overall_index",
    "outdoors_sub_domain": "living_env",
    "education_skills_and_training": "education",
    "crime": "crime",
    "health_deprivation_and_disability": "health",
    "employment": "employment",
}


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


def _transform_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.rename(lambda x: x.split("-")[-1])
        .rename(iod_column_cleaner)
        .unpivot(index=cs.string(), variable_name="measure_type")
    )


def get_underlying_indicators(
    source: str = "https://assets.publishing.service.gov.uk/media/691dec012c6b98ecdbc500d4/File_8_IoD2025_Underlying_Indicators.xlsx",
) -> pl.DataFrame:
    return pl.concat(
        _transform_underlying_indicators(df) for df in _iter_excel_sheet(source)
    )


def get_ltla_summary(
    source: str = "https://assets.publishing.service.gov.uk/media/6917412ebc34c86ce4e6e7fc/File_10_-_IoD2025_Local_Authority_District_Summaries__lower-tier__v2.xlsx",
) -> pl.DataFrame:
    return pl.concat(_transform_summary(df) for df in _iter_excel_sheet(source))


def get_utla_summary(
    source: str = "https://assets.publishing.service.gov.uk/media/6917414ab49cc44345161802/File_11_-_IoD2025_Local_Authority_District_Summaries__upper-tier__v2.xlsx",
) -> pl.DataFrame:
    return pl.concat(_transform_summary(df) for df in _iter_excel_sheet(source))


def get_imd_domains(
    source: str = "https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv",
) -> pl.DataFrame:
    domain_map = {
        "income_deprivation_affecting_older_people_idaopi": "income",
        "children_and_young_people_sub_domain": "education",
        "barriers_to_housing_and_services": "barriers",
        "living_environment": "living_env",
        "indoors_sub_domain": "living_env",
        "adult_skills_sub_domain": "education",
        "wider_barriers_sub_domain": "barriers",
        "income_deprivation_affecting_children_index_idaci": "income",
        "income": "income",
        "geographical_barriers_sub_domain": "barriers",
        "index_of_multiple_deprivation_imd": "overall_index",
        "outdoors_sub_domain": "living_env",
        "education_skills_and_training": "education",
        "crime": "crime",
        "health_deprivation_and_disability": "health",
        "employment": "employment",
    }
    index_type = pl.col("variable").str.extract("rank|score|decile", 0)
    suffix = pl.col("variable").str.extract("_(rank|score|decile).*", 0)
    indicator = pl.col("variable").str.strip_suffix(suffix)

    index_cols = (
        "lsoa_code_2021",
        "total_population_mid_2022",
    )

    return (
        pl.read_csv(source)
        .rename(iod_column_cleaner)
        .unpivot(index=index_cols, on=cs.numeric() ^ cs.contains("mid_2022"))
        .with_columns(index_type=index_type, indicator=indicator)
        .pivot("index_type", values="value", index=(*index_cols, "indicator"))
        .select(
            lsoa21cd="lsoa_code_2021",
            population_estimate_2022="total_population_mid_2022",
            domain=pl.col("indicator").replace_strict(domain_map),
            indicator="indicator",
            score="score",
            rank="rank",
            decile="decile",
        )
    )
