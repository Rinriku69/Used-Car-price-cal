import streamlit as st
from joblib import load
import pandas as pd

if "pred" not in st.session_state:
    st.session_state["pred"] = None


@st.cache_resource
def load_model():
    pipe = load("trained_model.joblib")

    return pipe


def calculate_price(pipe):
    miles = st.session_state["miles"]
    year = st.session_state["year"]
    make = st.session_state["make"]
    model = st.session_state["model"]
    engine_size = st.session_state["engine_size"]
    state = st.session_state["state"]

    # ...
    x_pred = pd.DataFrame(
        [[miles, year, make, model, engine_size, state]],
        columns=["miles", "year", "make", "model", "engine_size", "state"],
    )

    pred = pipe.predict(x_pred)
    pred = round(pred[0], 2)

    st.session_state["pred"] = pred


if __name__ == "__main__":
    st.header("Price Calculation")
    pipe = load_model()
    with st.form(key="Submit_form"):
        col1, col2, col3 = st.columns(3)
        col1.number_input("Miles", value=0.00, step=100.00, key="miles")
        col1.selectbox(
            "Model",
            options=[
                "Prius",
                "Highlander",
                "Civic",
                "Accord",
                "Corolla",
                "Ridgeline",
                "Odyssey",
                "CR-V",
                "Pilot",
                "Camry Solara",
                "Matrix",
                "RAV4",
                "Rav4",
                "HR-V",
                "Fit",
                "Yaris",
                "Yaris iA",
                "Tacoma",
                "Camry",
                "Avalon",
                "Venza",
                "Sienna",
                "Passport",
                "Accord Crosstour",
                "Crosstour",
                "Element",
                "Tundra",
                "Sequoia",
                "Corolla Hatchback",
                "4Runner",
                "Echo",
                "Tercel",
                "MR2 Spyder",
                "FJ Cruiser",
                "Corolla iM",
                "C-HR",
                "Civic Hatchback",
                "86",
                "S2000",
                "Supra",
                "Insight",
                "Clarity",
                "CR-Z",
                "Prius Prime",
                "Prius Plug-In",
                "Prius c",
                "Prius C",
                "Prius v",
            ],
            key="model",
        )
        col2.number_input(
            "Year", min_value=1997, max_value=2025, value=2001, key="year"
        )
        col2.number_input(
            "Engine size",
            min_value=1.3,
            max_value=5.7,
            value=1.3,
            step=0.1,
            key="engine_size",
        )

        col3.selectbox("Make", options=["toyota", "honda"], key="make")
        col3.selectbox(
            "State",
            options=[
                "BC",
                "ON",
                "AB",
                "QC",
                "NS",
                "MB",
                "SK",
                "NB",
                "NL",
                "PE",
                "YT",
                "OH",
                "NC",
                "SC",
            ],
            key="state",
        )

        st.form_submit_button(
            "Calculate",
            type="primary",
            on_click=calculate_price,
            kwargs={"pipe": pipe},
        )

    st.divider()

    if st.session_state["pred"] is not None:
        st.write(f"The calculate price is {st.session_state['pred']}")
    else:
        st.write("Calculate The Price")

    st.write(st.session_state)
