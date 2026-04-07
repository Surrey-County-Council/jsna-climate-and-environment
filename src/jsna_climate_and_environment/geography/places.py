import warnings
from typing import Literal
from jsna_climate_and_environment.config import OUTPUT_DIR
import shutil
import tempfile
from os import PathLike
from pathlib import Path
import functools
from jsna_climate_and_environment.geography.esri_api import (
    read_boundary_dataset,
    read_csv_lookup,
)
import geopandas as gpd
import polars as pl

BRITISH_NATIONAL_GRID = "EPSG:27700"
SURREY_COUNTY_CODE = "E10000030"

SURREY_EAST = (
    "E07000207",  # "Elmbridge",
    "E07000208",  # "Epsom and Ewell",
    "E07000210",  # "Mole Valley",
    "E07000211",  # "Reigate and Banstead",
    "E07000215",  # "Tandridge",
)
SURREY_WEST = (
    "E07000209",  # "Guildford",
    "E07000212",  # "Runnymede",
    "E07000213",  # "Spelthorne",
    "E07000214",  # "Surrey Heath",
    "E07000216",  # "Waverley",
    "E07000217",  # "Woking",
)

SURREY_DISTRICTS = (*SURREY_EAST, *SURREY_WEST)


# boundary reading functions are set up to facilitate ease of updates
# they are cached in memory to prevent multiple downloads
# the high level design is to specify the dataset name, the filter and the year as function arguments with
# default values which work together
# for future runs changing the default values should ensure the functions continue to work
@functools.cache
def get_surrey_county(
    dataset: str = "Counties_December_2024_Boundaries_EN_BSC",
    where: str = f"CTY24CD='{SURREY_COUNTY_CODE}'",
    year: int = 24,
) -> gpd.GeoDataFrame:
    gdf = read_boundary_dataset(dataset, layer_queries={0: {"where": where}})
    gdf["county_code"] = gdf[f"CTY{year}CD"]
    gdf["county_name"] = gdf[f"CTY{year}NM"]
    return gdf[["county_code", "county_name", "source_url", "geometry"]]


@functools.cache
def get_surrey_districts(
    dataset: str = "LAD_MAY_2025_UK_BFE_V2",
    where: str = f"LAD25CD in {SURREY_DISTRICTS}",
    year: int = 25,
) -> gpd.GeoDataFrame:
    gdf = read_boundary_dataset(dataset, layer_queries={0: {"where": where}})
    gdf["district_code"] = gdf[f"LAD{year}CD"]
    gdf["district_name"] = gdf[f"LAD{year}NM"]
    return gdf[["district_code", "district_name", "source_url", "geometry"]]


def get_surrey_lgr() -> gpd.GeoDataFrame:
    surrey_districts = get_surrey_districts()
    surrey_lgr = gpd.GeoDataFrame(
        data={
            "lgr_name": ["East Surrey", "West Surrey"],
            "lgr_code": ["SCC_EAST", "SCC_WEST"],
            "source_url": surrey_districts["source_url"][:2],
        },
        geometry=[
            surrey_districts[
                surrey_districts["district_code"].isin(SURREY_EAST)
            ].geometry.union_all(),
            surrey_districts[
                surrey_districts["district_code"].isin(SURREY_WEST)
            ].geometry.union_all(),
        ],
    )
    surrey_lgr.set_crs(BRITISH_NATIONAL_GRID, inplace=True)
    return surrey_lgr


# lookup functions are more challenging to update so a filter and year are not specified
# this function is only guaranteed to work with the dataset set as a default
@functools.cache
def get_nspl(
    dataset: Literal[
        "National_Statistics_Postcode_Lookup_(February_2026)_for_the_UK_(Hosted_Table)"
    ] = "National_Statistics_Postcode_Lookup_(February_2026)_for_the_UK_(Hosted_Table)",
) -> pl.DataFrame:
    return (
        read_csv_lookup(dataset)
        .select(
            country_code="ctry25cd",
            region_code="rgn25cd",
            county_code="cty25cd",
            lgr_code=pl.when(pl.col("lad25cd").is_in(SURREY_EAST))
            .then(pl.lit("SCC_EAST"))
            .when(pl.col("lad25cd").is_in(SURREY_WEST))
            .then(pl.lit("SCC_WEST")),
            district_code="lad25cd",
            msoa21_code="msoa21cd",
            lsoa21_code="lsoa21cd",
            source_url="source_url",
        )
        .filter(county_code=SURREY_COUNTY_CODE)
        .unique()
    )


# census boundaries change less frequently but are versioned
def get_surrey_lsoas(
    dataset: Literal[
        "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V10"
    ] = "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V10",
) -> gpd.GeoDataFrame:
    gdf = read_boundary_dataset(
        dataset,
        layer_queries={0: {"where": f"LSOA21CD in {tuple(get_nspl()['lsoa21_code'])}"}},
    ).explode(index_parts=False)  # tableau doesn't like multipolygons
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf["lsoa21_code"] = gdf["LSOA21CD"]
    gdf["lsoa21_name"] = gdf["LSOA21NM"]
    return gdf[["lsoa21_code", "lsoa21_name", "source_url", "geometry"]]


def get_surrey_msoas(
    dataset: Literal[
        "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V8"
    ] = "Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFE_V8",
) -> gpd.GeoDataFrame:
    gdf = read_boundary_dataset(
        dataset,
        layer_queries={0: {"where": f"MSOA21CD in {tuple(get_nspl()['msoa21_code'])}"}},
    ).explode(index_parts=False)  # tableau doesn't like multipolygons
    assert isinstance(gdf, gpd.GeoDataFrame)
    gdf["msoa21_code"] = gdf["MSOA21CD"]
    gdf["msoa21_name"] = gdf["MSOA21NM"]
    return gdf[["msoa21_code", "msoa21_name", "source_url", "geometry"]]


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


# this creates tests and type hint support for each layer we manually add to the geodatabase
IMPLEMENTED_LAYERS = Literal["linking", "lsoa", "msoa", "lgr", "districts", "county"]


# a manual check that the IMPLEMENTED_LAYERS above are all present is required
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
            linking=gpd.GeoDataFrame(get_nspl().to_pandas()),
            lsoa=get_surrey_lsoas(),
            msoa=get_surrey_msoas(),
            lgr=get_surrey_lgr(),
            districts=get_surrey_districts(),
            county=get_surrey_county(),
        )
    assert output_path.exists()
    return output_path


def read_gdb(
    layer: IMPLEMENTED_LAYERS,
) -> gpd.GeoDataFrame:
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

    # test all implemented layers can be read
    for layer in IMPLEMENTED_LAYERS.__args__:
        if read_gdb(layer).empty:
            raise ValueError(f"layer '{layer}' appears corrupted in geodatabase")
