import warnings
from typing import Literal
from jsna_climate_and_environment.config import OUTPUT_DIR
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from jsna_climate_and_environment.geography.esri_api import (
    read_boundary_dataset,
    read_csv_lookup,
)
import geopandas as gpd
import polars as pl


BRITISH_NATIONAL_GRID_SRID = 27700
BRITISH_NATIONAL_GRID = f"EPSG:{BRITISH_NATIONAL_GRID_SRID}"
SURREY_COUNTY_CODE = "E10000030"

SURREY_EAST_NAMES = {
    "E07000207": "Elmbridge",
    "E07000208": "Epsom and Ewell",
    "E07000210": "Mole Valley",
    "E07000211": "Reigate and Banstead",
    "E07000215": "Tandridge",
}

SURREY_EAST = (
    "E07000207",  # "Elmbridge",
    "E07000208",  # "Epsom and Ewell",
    "E07000210",  # "Mole Valley",
    "E07000211",  # "Reigate and Banstead",
    "E07000215",  # "Tandridge",
)

SURREY_WEST_NAMES = {
    "E07000209": "Guildford",
    "E07000212": "Runnymede",
    "E07000213": "Spelthorne",
    "E07000214": "Surrey Heath",
    "E07000216": "Waverley",
    "E07000217": "Woking",
}
SURREY_WEST = (
    "E07000209",  # "Guildford",
    "E07000212",  # "Runnymede",
    "E07000213",  # "Spelthorne",
    "E07000214",  # "Surrey Heath",
    "E07000216",  # "Waverley",
    "E07000217",  # "Woking",
)

SURREY_DISTRICT_NAMES = SURREY_EAST_NAMES | SURREY_WEST_NAMES
SURREY_DISTRICTS = (*SURREY_EAST, *SURREY_WEST)


def get_nspl(
    dataset: Literal[
        "National_Statistics_Postcode_Lookup_(February_2026)_for_the_UK_(Hosted_Table)"
    ] = "National_Statistics_Postcode_Lookup_(February_2026)_for_the_UK_(Hosted_Table)",
) -> pl.DataFrame:
    """"""

    msoa_names = pl.read_csv(
        "https://houseofcommonslibrary.github.io/msoanames/MSOA-Names-Latest2.csv"
    ).select(msoa21_code="msoa21cd", msoa21_name="msoa21hclnm")

    lsoa_names = (
        read_csv_lookup("LSOA_DEC_2021_EW_NC_v3")
        .rename(str.lower)
        .select(lsoa21_code="lsoa21cd", lsoa21_name="lsoa21nm")
    )

    return (
        read_csv_lookup(dataset)
        .filter(cty25cd=SURREY_COUNTY_CODE)
        .select(
            county_code="cty25cd",
            lgr=pl.when(pl.col("lad25cd").is_in(SURREY_EAST))
            .then(pl.lit("East Surrey"))
            .when(pl.col("lad25cd").is_in(SURREY_WEST))
            .then(pl.lit("West Surrey")),
            district_code="lad25cd",
            district_name=pl.col("lad25cd").replace_strict(SURREY_DISTRICT_NAMES),
            msoa21_code="msoa21cd",
            lsoa21_code="lsoa21cd",
            source_url="source_url",
        )
        .unique()
        .join(msoa_names, on="msoa21_code", how="left", validate="m:1")
        .join(lsoa_names, on="lsoa21_code", how="left", validate="1:1")
    )


# census boundaries change less frequently but are versioned
def get_surrey_lsoas(
    dataset: Literal[
        "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10"
    ] = "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10",
) -> gpd.GeoDataFrame:
    mapping = get_nspl()

    gdf = read_boundary_dataset(
        dataset,
        layer_queries={
            0: {"where": f"LSOA21CD in {tuple(mapping['lsoa21_code'].unique())}"}
        },
    ).explode(index_parts=False)  # tableau doesn't like multipolygons
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf["lsoa21_code"] = gdf["LSOA21CD"]
    return gdf[["lsoa21_code", "source_url", "geometry"]]


# utility function can work with any configuration
def write_tableau_gdb(
    output_dir: str | PathLike, file_name: str, **layers: gpd.GeoDataFrame
) -> Path:
    """utility to write an Esri File Geodatabase that functions correctly as a tableau data source.

    output_dir: ie path/to/directory
    filename: ie geography (don't include extentions)
    **layers: each gdf to write as a layer within this archive. gdf's will be simplified to work within tableau.
    user can provide one or many layers

    >>> write_tableau_gdb("path/to/file", my_layer=gdf)  # doctest: ignore
    will write to 'path/to/file.zip' an Esri File Geodatabase with one layer named 'my_layer'
    """
    if not layers:
        raise ValueError("you must provide named layers to write as keyword arguments")
    output_path = Path(output_dir) / f"{file_name}.gdb.zip"
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        temp_file = Path(tmp) / f"{file_name}.gdb"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*OpenFileGDB does not support open option DRIVER.*"
            )
            for layer_name, gdf in layers.items():
                gdf.to_file(
                    temp_file,
                    layer=layer_name,
                    driver="OpenFileGDB",
                    index=False,
                    engine="pyogrio",
                )
        shutil.make_archive(
            str(Path(output_dir) / f"{file_name}.gdb"), "zip", temp_file
        )
    assert output_path.exists()
    return output_path


def setup_geodb(overwrite: bool = False) -> Path:
    """setup the file geodatabase for this project.

    overwrite: bool,

    if true will overwrite the geodatabase by downloading fresh data

    if false will read data that already exists or download data if it doesn't exist"""
    output_path = OUTPUT_DIR / "places.gdb.zip"
    if overwrite or (not output_path.exists()):
        write_tableau_gdb(
            OUTPUT_DIR,
            "places",
            lsoa=get_surrey_lsoas(),
            nspl=gpd.GeoDataFrame(get_nspl().to_pandas()),
        )
    assert output_path.exists()
    return output_path


def read_gdb(layer: str = "lsoa") -> gpd.GeoDataFrame:
    """most likely point of entry for analysis tasks that read geographical data."""
    return gpd.read_file(
        setup_geodb(),
        layer=layer,
        engine="pyogrio",
    )


if __name__ == "__main__":
    # integration test will setup the data for use in any environment (live or development)
    # testing will automatically overwrite the existing data and should be done on every update
    # testing is as simple as running this as a script
    setup_geodb(overwrite=True)
    if read_gdb("lsoa").empty:
        raise ValueError("layer 'lsoa' appears corrupted in geodatabase")

    if read_gdb("nspl").empty:
        raise ValueError("layer 'nspl' appears corrupted in geodatabase")
