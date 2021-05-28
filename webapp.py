import streamlit as st
from webapp_func import load_data
from webapp_exploration import exploration_app
from webapp_fitting import fitting_app


#! run in console with: streamlit run webapp.py


# configure streamlit to use the whole page
st.set_page_config(page_title="Lifestock produce dashboard ", layout="wide")


# print title and subtitle
st.title("Lifestock produce dashboard")
st.markdown(
    "### Food production, import/export and population analysis\n"
    + "Data taken from [Food and Agriculture Organization of the "
    + "United Nations](http://www.fao.org/faostat/en/#data)."
)

# load data
with st.spinner("loading data..."):
    df_dict = load_data()

# show options and get user input
option = st.radio("Select option: ", ["Exploration", "Fitting"])

# if user chooses exploration, start the exploration part of the app
if option == "Exploration":
    exploration_app(df_dict)
# if the user choooses fittin, start the fitting part of the app
else:
    fitting_app(df_dict)
