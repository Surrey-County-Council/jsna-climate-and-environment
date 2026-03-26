import re
import fsspec
from typing import Literal
import httpx
from datetime import date
from bs4 import BeautifulSoup
import polars as pl
import pendulum


class ApiError(Exception):
    pass


def get_link(
    url_date: str,
    file_name: Literal["gp-reg-pat-prac-all.zip", "gp-reg-pat-prac-map.zip"],
) -> str:
    response = httpx.get(
        f"https://digital.nhs.uk/data-and-information/publications/statistical/patients-registered-at-a-gp-practice/{url_date}"
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content)
    html_parts = soup.find_all(
        class_="nhsd-a-box-link",
        onclick="Publication",
        href=re.compile(f".+{file_name}"),
    )
    if len(html_parts) != 1:
        raise ApiError(
            f"searching for {file_name} returned {len(html_parts)} results. Expected 1"
        )
    try:
        return str(html_parts[0]["href"])
    except KeyError:
        raise ApiError("href tag not found for link to file")


def get_gp_lookup(release_date: date) -> pl.DataFrame:
    url_date = release_date.strftime("%B-%Y").lower()
    zip_name = "gp-reg-pat-prac-map.zip"
    csv_name = "gp-reg-pat-prac-map.csv"
    url = get_link(url_date, zip_name)
    with fsspec.open(f"zip://{csv_name}::{url}") as file:
        return pl.read_csv(file)


def to_pcd2(postcode_col: str | pl.Expr) -> pl.Expr:
    """the pcd2 standard format is used in postcode mappings to ensure postcodes are all 8 characters"""
    if isinstance(postcode_col, str):
        postcode_col = pl.col(postcode_col)
    postcode_col = postcode_col.str.replace_all(r"\s", "")
    outward = postcode_col.str.slice(0, postcode_col.str.len_chars() - 3).str.pad_end(5)
    inward = postcode_col.str.slice(-3, 3)
    return pl.concat_str(outward, inward)


def get_surrey_health_lookup(release_date: date | None = None) -> pl.DataFrame:
    if release_date is None:
        release_date = pendulum.today().subtract(weeks=2)
    # select the quarter for LSOA files
    months_past_quarter = pendulum.Duration(months=release_date.month % 3)
    release_date = release_date - months_past_quarter

    df = get_gp_lookup(release_date)
    return (
        df.rename(str.lower)
        .with_columns(
            extract_date=pl.col("extract_date").str.to_date("%d%b%Y"),
            pcd2=to_pcd2("practice_postcode"),
        )
        .filter(
            pl.col("sub_icb_location_name").str.contains("[Ss]urrey|[Ff]rimley")
            | pl.col("icb_name").str.contains("[Ss]urrey|[Ff]rimley")
        )
    )
