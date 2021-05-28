import streamlit as st
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import sqlite3


# * ################### CONSTANTS #####################
@dataclass
class C:
    """Class containing constants"""

    filepath: Path = Path("data.db")  # filepath to the database
    pop: str = "population"  # population database table name
    pop_pred: str = "population_prediction"  # population prediction database table name
    prod: str = "production"  # production database table name
    trade: str = "trade"  # population database table name


# * ################ HELP FUNCTIONS ####################
# load data function is cached to speed up the webapp
@st.cache(show_spinner=False)
def load_data() -> dict:
    """Loads data from database and returns the tables as a dictionary.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of pandas dataframes containing table data
    """
    with sqlite3.connect(C.filepath) as conn:
        df_dict = {
            C.pop: pd.read_sql(f"SELECT * FROM {C.pop}", conn),
            C.pop_pred: pd.read_sql(f"SELECT * FROM {C.pop_pred}", conn),
            C.prod: pd.read_sql(f"SELECT * FROM {C.prod}", conn),
            C.trade: pd.read_sql(f"SELECT * FROM {C.trade}", conn),
        }
    return df_dict


@st.cache(show_spinner=False)
def filter_df(
    df, country_filter: list = [], element_filter: list = [], item_filter: list = []
) -> pd.DataFrame:
    """Filters the given DataFrame by country, element and item

    Args:
        df (pd.DataFrame): DataFrame containing the data to filter
        country_filter (list, optional): List of countries to filter for. Defaults to [].
        element_filter (list, optional): List of elements to filter for. Defaults to [].
        item_filter (list, optional): List of items to filter for. Defaults to [].

    Returns:
        pd.DataFrame: Filtered copy of the given DataFrame
    """

    df_filtered = df.copy()

    if country_filter:
        df_filtered = df_filtered[df_filtered.Area.isin(country_filter)]

    if element_filter:
        df_filtered = df_filtered[df_filtered.Element.isin(element_filter)]

    if item_filter:
        df_filtered = df_filtered[df_filtered.Item.isin(item_filter)]

    return df_filtered


@st.cache(show_spinner=False)
def unit_extraction(df_dict):
    """Extract element units from dataframe

    Args:
        df_dict (pd.DataFrame): DataFrame to extract units from

    Returns:
        dict[str, str]: dictionary mapping elements to units
    """
    unit_dict = {}
    for df in df_dict.values():
        for element in df.Element.unique():
            if element not in unit_dict.keys():
                unit = df[df.Element == element].Unit.values[0]
                unit_dict[element] = unit

    return unit_dict