import streamlit as st

st.set_page_config(
    page_title="CS6303 Project",
    page_icon="🧠",
    layout="wide"
)

pg = st.navigation([
    st.Page("pages/st_customer.py", title="Customer View", icon=":material/support_agent:"),
    st.Page("pages/st_employee.py", title="Employee View", icon=":material/badge:"),
])
pg.run()