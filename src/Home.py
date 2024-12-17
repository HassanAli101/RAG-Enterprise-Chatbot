import streamlit as st
from employee_interface.StreamlitFrontend import chatbot_page

# Set page configuration (only once, at the top of the main file)
st.set_page_config(
    page_title="Employee Assistant App",
    page_icon="🏢",
    layout="wide"
)

# Main page
st.title("Welcome to the Employee Assistant App! 🏢")

# Navigation
page = st.selectbox("Navigate to:", ["Home", "Chatbot"])

if page == "Chatbot":
    chatbot_page()
else:
    st.write("This is the home page! Choose 'Chatbot' from the navigation menu to start.")
