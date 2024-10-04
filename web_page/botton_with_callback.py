import streamlit as st

def func():
    print('I was pressed!')

st.button("Reset", type="primary", on_click=func)
if st.button("Say hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")

