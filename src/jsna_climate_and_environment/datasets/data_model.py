from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    dataset: str
    """name of the dataset at its source location"""
    source: str
    """usually a url"""
    name_at_source: str
    """name of the indicator derrived from the source data"""
    indicator_name: str = Field(default_factory=lambda data: data["name_at_source"])
    """name of the indicator in the dashboard. Change this if the name_at_source requires a rename"""
    measure_name: str
    """name of the measure (used to distinguish different types of indicator with the same name or dataset, often displayed on an axis), 
    ie: temperature, number of people, x_domain etc."""
    description: str
    """more in depth description of the indicator and what it measures"""
    rationalle: str
    """Analytical notes and a rationalle for inclusion. Why the indicator is important and what it shows."""
    caveats: str | None = None
    """Important caveats to be aware of when interpreting the data"""


@runtime_checkable
class HasDataset(Protocol):
    dataset: str


@runtime_checkable
class HasSource(Protocol):
    source: str
