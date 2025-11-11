import requests
import streamlit as st
import pandas as pd
from datetime import date

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fuel Price Optimization",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Train Model", "Daily Price Recommendation"],
)


def call_train():
    with st.spinner("Training model on historical data..."):
        resp = requests.post(f"{BACKEND_URL}/train")
    resp.raise_for_status()
    return resp.json()


def call_recommend(payload: dict):
    resp = requests.post(f"{BACKEND_URL}/recommend_price", json=payload)
    resp.raise_for_status()
    return resp.json()


if page == "Overview":
    st.title("⛽ Fuel Price Optimization – Classical ML App")

    st.markdown(
        """
        This app implements a **classical machine learning system** to optimize
        daily fuel prices for a retail petrol company.
        """
    )

elif page == "Train Model":
    st.title("📈 Train Demand Model")

    st.markdown(
        """Click the button below to train/retrain the model on historical data."""
    )

    if st.button("Train / Retrain Model"):
        try:
            result = call_train()
            st.success("Model trained successfully!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{result['r2']:.3f}")
            col2.metric("RMSE", f"{result['rmse']:.2f}")
            col3.metric("MAE", f"{result['mae']:.2f}")
            col4.metric("# Samples", result["n_samples"])
        except Exception as e:
            st.error(f"Training failed: {e}")

elif page == "Daily Price Recommendation":
    st.title("💰 Daily Price Recommendation")

    with st.form("price_form"):
        col1, col2 = st.columns(2)

        with col1:
            today_date = st.date_input("Date", value=date.today())
            last_price = st.number_input(
                "Yesterday's company price",
                value=94.45,
                step=0.1,
                format="%.2f",
            )
            cost = st.number_input(
                "Today's cost",
                value=85.77,
                step=0.1,
                format="%.2f",
            )

        with col2:
            comp1 = st.number_input(
                "Competitor 1 price",
                value=95.01,
                step=0.1,
                format="%.2f",
            )
            comp2 = st.number_input(
                "Competitor 2 price",
                value=95.70,
                step=0.1,
                format="%.2f",
            )
            comp3 = st.number_input(
                "Competitor 3 price",
                value=95.21,
                step=0.1,
                format="%.2f",
            )

        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        payload = {
            "date": today_date.isoformat(),
            "price": float(last_price),
            "cost": float(cost),
            "comp1_price": float(comp1),
            "comp2_price": float(comp2),
            "comp3_price": float(comp3),
        }

        try:
            result = call_recommend(payload)

            st.subheader("Recommended Price")
            st.metric(
                label="Recommended Retail Price",
                value=f"{result['recommended_price']:.2f}",
            )

            col1, col2 = st.columns(2)
            col1.metric(
                "Expected Volume (L)",
                f"{result['expected_volume']:.0f}",
            )
            col2.metric(
                "Expected Profit",
                f"{result['expected_profit']:.2f}",
            )

            candidates = pd.DataFrame(result["candidates"]).sort_values("price")

            st.subheader("Profit vs Price")
            st.line_chart(
                data=candidates.set_index("price")[["profit"]],
            )

            with st.expander("Candidate Price Table"):
                st.dataframe(candidates)

        except Exception as e:
            st.error(f"Failed to get recommendation: {e}")
