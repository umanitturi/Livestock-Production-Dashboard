import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from webapp_func import C, filter_df


# * ######################## DATA SELECTION ##########################
def data_selection(df_dict):
    """Gets user input to select data from the given dataframes

    Args:
        df_dict (dict[str, pd.DataFrame]): Dictionary of dataframes for data selection

    Returns:
        str, str, str, str: Strings with Area, Item, Population and Production/Trade selection
    """
    col1, col2 = st.beta_columns(2)

    with col1:
        area_list = df_dict[C.pop].Area.unique()
        area_selection = st.multiselect("Countries: ", area_list)

        item_list = df_dict[C.prod].Item.unique()
        item_selection = st.multiselect("Food item: ", item_list)

    with col2:
        pop_element_list = df_dict[C.pop].Element.unique()
        pop_element_selection = st.multiselect("Population: ", pop_element_list)

        element_list = list(df_dict[C.prod].Element.unique())
        element_list += list(df_dict[C.trade].Element.unique())
        element_selection = st.multiselect("Trade, Production: ", element_list)

    return area_selection, item_selection, pop_element_selection, element_selection


# * ######################## DATA PLOTTING ##########################
def data_plot(
    df_dict, area_selection, item_selection, pop_element_selection, element_selection
):
    """Plot the selected data

    Args:
        df_dict (dict[str, pd.DataFrame]): Dictionary of dataframes for data selection
        area_selection (str): Area selection
        item_selection (str): Item selection
        pop_element_selection (str): Population element selection
        element_selection (str): Production/Trade element selection
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # create production plots
    if "Production" in element_selection:
        df_prod_filtered = filter_df(
            df_dict[C.prod],
            country_filter=area_selection,
            item_filter=item_selection,
        )
        for country in df_prod_filtered.Area.unique():
            for item in df_prod_filtered.Item.unique():
                df_plot = filter_df(
                    df_prod_filtered,
                    country_filter=[country],
                    item_filter=[item],
                )
                fig.add_scatter(
                    x=df_plot.Year,
                    y=df_plot.Value,
                    name=f"Production: {country} - {item}",
                    mode="lines",
                    legendgroup="Production",
                    secondary_y=False,
                )

    # create trade plots
    if "Import" in element_selection or "Export" in element_selection:
        df_trade_filtered = filter_df(
            df_dict[C.trade],
            country_filter=area_selection,
            item_filter=item_selection,
            element_filter=element_selection,
        )
        for country in df_trade_filtered.Area.unique():
            for item in df_trade_filtered.Item.unique():
                for element in df_trade_filtered.Element.unique():
                    df_plot = filter_df(
                        df_trade_filtered,
                        country_filter=[country],
                        item_filter=[item],
                        element_filter=[element],
                    )
                    fig.add_scatter(
                        x=df_plot.Year,
                        y=df_plot.Value,
                        name=f"{element}: {country} - {item}",
                        mode="lines",
                        legendgroup="Trade",
                        secondary_y=False,
                    )

    # create population plots
    if pop_element_selection:
        df_pop_filtered = filter_df(
            df_dict[C.pop],
            country_filter=area_selection,
            element_filter=pop_element_selection,
        )
        for country in df_pop_filtered.Area.unique():
            for element in df_pop_filtered.Element.unique():
                df_plot = filter_df(
                    df_pop_filtered,
                    country_filter=[country],
                    element_filter=[element],
                )
                fig.add_scatter(
                    x=df_plot.Year,
                    y=df_plot.Value,
                    name=f"Population: {country} - {element}",
                    line=dict(dash="dash"),
                    mode="lines",
                    legendgroup="Population",
                    secondary_y=True,
                )

    # change figure options
    fig.update_xaxes(title={"text": "Time (years)"})
    fig.update_yaxes(
        title={"text": "Production, Import, Export (tonnes)"}, secondary_y=False
    )
    fig.update_yaxes(title={"text": "Population"}, secondary_y=True)

    fig.update_layout(
        autosize=True,
        width=1200,
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.layout.template = "simple_white"

    st.plotly_chart(fig)


def exploration_app(df_dict):
    """Part of the app that allows for plotting time series data

    Args:
        df_dict (dict[str, pd.DataFrame]): DataFrame containing the data to plot
    """

    # get user input for data selection
    (
        area_selection,
        item_selection,
        pop_element_selection,
        element_selection,
    ) = data_selection(df_dict)

    # plot the selected data
    data_plot(
        df_dict,
        area_selection,
        item_selection,
        pop_element_selection,
        element_selection,
    )
