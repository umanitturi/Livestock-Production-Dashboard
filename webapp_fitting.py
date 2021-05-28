import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from webapp_func import C, filter_df, unit_extraction


# * ######################## DATA SELECTION ##########################
def data_selection(df_dict):
    """Get user input and select data from dataframe dictionary

    Args:
        df_dict (dict[str, pd.DataFrame]): Dictionary of Dataframes containing data to select from

    Returns:
        list[pd.DataFrame]: list of filtered dataframes
        string: Element to plot on the x axis
        string: Element to plot on the y axis

    """

    # show data selection widgets in two columns
    col1, col2 = st.beta_columns(2)

    # get list of available areas and get user selection
    with col1:
        area_list = df_dict[C.pop].Area.unique()
        ger_index = int(np.where(area_list == "Germany")[0])
        area_selection = st.selectbox("Countries: ", area_list, index=ger_index)

    # filter data with selected area
    df_filtered_pop = filter_df(df_dict[C.pop], country_filter=[area_selection])
    df_filtered_prod = filter_df(df_dict[C.prod], country_filter=[area_selection])
    df_filtered_trade = filter_df(df_dict[C.trade], country_filter=[area_selection])

    # get list of available food items and get user selection
    with col1:
        item_list = df_dict[C.prod].Item.unique()
        milk_index = int(np.where(item_list == "Milk, whole fresh cow")[0])
        item_selection = st.selectbox("Food item: ", item_list, index=milk_index)

    # filter data with selected food item
    df_filtered_prod = filter_df(df_filtered_prod, item_filter=[item_selection])
    df_filtered_trade = filter_df(df_filtered_trade, item_filter=[item_selection])

    # get list of available data elements and get user selection for x and y axes
    with col2:
        element_list = ["Year"]
        element_list += list(df_filtered_pop.Element.unique())
        element_list += list(df_filtered_prod.Element.unique())
        element_list += list(df_filtered_trade.Element.unique())

        year_index = element_list.index("Year")
        x_element_selection = st.selectbox("x-axis: ", element_list, index=year_index)
        prod_index = element_list.index("Production")
        y_element_selection = st.selectbox("y-axis: ", element_list, index=prod_index)

    df_filtered_list = [df_filtered_pop, df_filtered_prod, df_filtered_trade]

    return df_filtered_list, x_element_selection, y_element_selection


# * ######################## DATA PREPARATION ##########################
def data_preparation(df_filtered_list, x_element_selection, y_element_selection):
    """Prepares data for fitting and plotting

    Args:
        df_filtered_list (list[pd.DataFrame]): List of dataframes to prepare
        x_element_selection (str): String with selected element for x-axis
        y_element_selection (str): String with selected element for y-axis

    Returns:
        pd.DataFrame: Merged dataframe with prepared data
    """
    # pivot tables to merge the three datasets
    df_filtered_pop = (
        df_filtered_list[0]
        .pivot(index=["Area", "Year"], columns="Element", values="Value")
        .reset_index()
    )
    df_filtered_prod = (
        df_filtered_list[1]
        .pivot(index=["Area", "Item", "Year"], columns="Element", values="Value")
        .reset_index()
    )
    df_filtered_trade = (
        df_filtered_list[2]
        .pivot(index=["Area", "Item", "Year"], columns="Element", values="Value")
        .reset_index()
    )

    # merge the three datasets by year, area and item
    df_filtered = df_filtered_pop.merge(df_filtered_prod, on=["Year", "Area"])
    df_filtered = df_filtered.merge(df_filtered_trade, on=["Year", "Area", "Item"])

    # select only the columns needed for the plot
    if x_element_selection != y_element_selection:
        return df_filtered[[x_element_selection, y_element_selection]]
    else:  # if x and y are the same, make sure only one column is in the table
        return df_filtered[[x_element_selection]]


# * ######################## DATA FITTING ##########################
def data_fitting(df, x_element_selection, y_element_selection):
    """Fit data from DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_element_selection (str): Column name to use for x values
        y_element_selection (str): Column name to use for y values

    Returns:
        np.array: Array of x-values for fitted line
        np.array: Array of y-values for fitted line
        np.array: Array of x-values for predicted line
        np.array: Array of y-values for predicted line
    """
    # save minimum and maximum of data as integers
    fit_minimum = int(df[x_element_selection].min())
    fit_maximum = int(df[x_element_selection].max())

    # calculate reasonable stepsize for fitting/predictoin range sliders
    if x_element_selection != "Year":
        step_size = (fit_maximum - fit_minimum) // 1000
    else:
        step_size = 1

    # create three columns and show the fit selection slider in the middle
    col1, col2, col3 = st.beta_columns(3)
    with col2:
        # slider for selecting the fitting range
        fit_range = st.slider(
            "x fitting range: ",
            min_value=fit_minimum,
            max_value=fit_maximum,
            value=(fit_minimum, fit_maximum),
            step=step_size,
        )

    # slider for selecting the prediction range (full width)
    range_extension = fit_maximum - fit_minimum
    prediction_range = st.slider(
        "x prediction range: ",
        min_value=fit_minimum - range_extension,
        max_value=fit_maximum + range_extension,
        value=(fit_maximum, fit_maximum + range_extension),
        step=step_size,
    )

    # get X and y variables to fit
    df_fit = df[
        (df[x_element_selection] >= fit_range[0])
        & (df[x_element_selection] <= fit_range[1])
    ]
    X = df_fit[x_element_selection].to_numpy()
    y = df_fit[y_element_selection].to_numpy()

    # fit data
    lreg = LinearRegression()
    lreg.fit(np.array([X]).reshape(-1, 1), y)

    # data for plotting fitted line
    X_fit = np.linspace(fit_range[0], fit_range[1], 50)
    y_fit = lreg.predict(np.array([X_fit]).reshape(-1, 1))

    # data for plotting prediction
    X_pred = np.linspace(prediction_range[0], prediction_range[1], 50)
    y_pred = lreg.predict(np.array([X_pred]).reshape(-1, 1))

    return X_fit, y_fit, X_pred, y_pred


# * ######################## DATA PLOTTING ##########################
def data_plot(
    df_plot,
    unit_dict,
    X_fit,
    y_fit,
    X_pred,
    y_pred,
    x_element_selection,
    y_element_selection,
):
    """Plot data, fit and prediction

    Args:
        df_plot (pd.DataFrame): DataFrame to plot
        unit_dict ([type]): Dictionary with units for the elements
        X_fit (np.array): Array of x-values for fitted line
        y_fit (np.array): Array of y-values for fitted line
        X_pred (np.array): Array of x-values for predicted line
        y_pred (np.array): Array of y-values for predicted line
        x_element_selection (str): Column name to use for x values
        y_element_selection (str): Column name to use for y values
    """

    fig = make_subplots()

    # plot data
    fig.add_scatter(
        x=df_plot[x_element_selection],
        y=df_plot[y_element_selection],
        name="Data",
        mode="markers",
        marker=dict(color="black"),
    )

    # plot fitted line
    fig.add_scatter(
        x=X_fit, y=y_fit, mode="lines", name="Fit", line=dict(color="green")
    )

    # plot predicted line
    fig.add_scatter(
        x=X_pred,
        y=y_pred,
        mode="lines",
        name="Prediction",
        line=dict(dash="dash", color="red"),
    )

    fig.update_layout(
        autosize=True,
        width=1200,
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if x_element_selection != "Year":
        x_unit = unit_dict[x_element_selection]
        fig.update_xaxes(title={"text": f"{x_element_selection} ({x_unit})"})
    else:
        fig.update_xaxes(title={"text": x_element_selection})

    if y_element_selection != "Year":
        y_unit = unit_dict[y_element_selection]
        fig.update_yaxes(title={"text": f"{y_element_selection} ({y_unit})"})
    else:
        fig.update_yaxes(title={"text": y_element_selection})

    fig.layout.template = "simple_white"

    st.plotly_chart(fig)


# * ######################## MAIN FUNCTION  ##########################
def fitting_app(df_dict):
    """Part of the app that allows for fitting two data columns against each other

    Args:
        df_dict (dict[pd.DataFrame]): Dictionary with DataFrames to analyse
    """

    # get data selection to work with
    df_filtered_list, x_element_selection, y_element_selection = data_selection(df_dict)

    # prepare data for fitting and plotting
    df_plot = data_preparation(
        df_filtered_list, x_element_selection, y_element_selection
    )

    # fit data and get line data to plot
    X_fit, y_fit, X_pred, y_pred = data_fitting(
        df_plot, x_element_selection, y_element_selection
    )

    # get unit dictionary and plot data, fit and prediction
    unit_dict = unit_extraction(df_dict)
    data_plot(
        df_plot,
        unit_dict,
        X_fit,
        y_fit,
        X_pred,
        y_pred,
        x_element_selection,
        y_element_selection,
    )