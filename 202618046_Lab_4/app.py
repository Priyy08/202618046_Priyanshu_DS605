
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# NYC AIRBNB PRICE PREDICTOR
# Task 3 - Streamlit Application
# ============================================================

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "airbnb_price_prediction_pipeline.pkl"


# ------------------------------------------------------------
# Load trained pipeline
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def prepare_input(
    neighbourhood_group,
    neighbourhood,
    latitude,
    longitude,
    room_type,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,
    availability_365,
    last_review_year,
    last_review_month
):
    """
    Creates the exact feature structure expected by the
    trained Airbnb preprocessing + model pipeline.
    """

    # Cap minimum nights at the same threshold used during Task 1.
    # The Task 1 notebook used the 99th percentile of the cleaned
    # training dataset. The value below corresponds to the NYC
    # Airbnb dataset and should match the value used during training.
    minimum_nights_upper = 365

    minimum_nights_capped = min(
        float(minimum_nights),
        minimum_nights_upper
    )

    # Review indicators
    has_review = int(number_of_reviews > 0)
    has_last_review = int(last_review_year > 0)

    # Approximate NYC center used during feature engineering
    nyc_lat = 40.7128
    nyc_lon = -74.0060

    distance_from_nyc_center = np.sqrt(
        (latitude - nyc_lat) ** 2 +
        (longitude - nyc_lon) ** 2
    )

    input_data = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights_capped": minimum_nights_capped,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count":
            calculated_host_listings_count,
        "availability_365": availability_365,
        "last_review_year": last_review_year,
        "last_review_month": last_review_month,
        "has_review": has_review,
        "has_last_review": has_last_review,
        "distance_from_nyc_center":
            distance_from_nyc_center
    }])

    return input_data


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("🏠 NYC Airbnb Price Predictor")
st.markdown(
    """
    ### Estimate the nightly price of an Airbnb listing in New York City

    Enter the listing's location, room type, stay requirements,
    review activity and availability. The trained machine-learning
    pipeline will estimate the expected **nightly price in USD**.
    """
)

st.divider()


# ------------------------------------------------------------
# Model availability check
# ------------------------------------------------------------
if model is None:
    st.error(
        "The trained model file was not found. "
        "Place 'airbnb_price_prediction_pipeline.pkl' "
        "in the same folder as app.py."
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar - Model information
# ------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Model Information")

    st.success("Model loaded successfully")

    st.markdown(
        """
        **Algorithm:** HistGradient Boosting Regressor

        **Target:** Log-transformed Airbnb price

        **Output:** Estimated nightly price

        **Evaluation metrics from Task 2:**
        - MAE: $45.02
        - RMSE: $87.28
        - R²: 0.4541
        """
    )

    st.divider()

    st.caption(
        "The saved pipeline performs preprocessing and prediction "
        "using the same workflow used during model training."
    )


# ------------------------------------------------------------
# Input sections
# ------------------------------------------------------------
st.subheader("📍 Listing Location")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ],
        index=0
    )

with col2:
    neighbourhood = st.text_input(
        "Neighbourhood",
        value="Midtown",
        help="Enter the NYC neighbourhood name."
    )


col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=40.45,
        max_value=40.95,
        value=40.7549,
        step=0.0001,
        format="%.4f"
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-74.30,
        max_value=-73.65,
        value=-73.9840,
        step=0.0001,
        format="%.4f"
    )


st.subheader("🛏️ Property Details")

col1, col2 = st.columns(2)

with col1:
    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

with col2:
    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=3650,
        value=3,
        step=1
    )


st.subheader("⭐ Reviews & Host Information")

col1, col2 = st.columns(2)

with col1:
    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=1000,
        value=50,
        step=1
    )

with col2:
    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=100.0,
        value=1.5,
        step=0.1,
        format="%.1f"
    )


col1, col2 = st.columns(2)

with col1:
    calculated_host_listings_count = st.number_input(
        "Host's Number of Listings",
        min_value=1,
        max_value=500,
        value=1,
        step=1
    )

with col2:
    availability_365 = st.slider(
        "Availability (days/year)",
        min_value=0,
        max_value=365,
        value=200
    )


st.subheader("📅 Review History")

has_previous_reviews = number_of_reviews > 0

if has_previous_reviews:
    col1, col2 = st.columns(2)

    with col1:
        last_review_year = st.number_input(
            "Last Review Year",
            min_value=2010,
            max_value=2026,
            value=2019,
            step=1
        )

    with col2:
        last_review_month = st.slider(
            "Last Review Month",
            min_value=1,
            max_value=12,
            value=6
        )
else:
    last_review_year = 0
    last_review_month = 0

    st.info(
        "No reviews entered, so the application automatically "
        "sets the review-date indicators to 0."
    )


st.divider()


# ------------------------------------------------------------
# Prediction button
# ------------------------------------------------------------
predict_button = st.button(
    "💰 Estimate Nightly Price",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Validation
    if not neighbourhood.strip():
        st.warning("Please enter a neighbourhood.")
        st.stop()

    if number_of_reviews == 0:
        reviews_per_month = 0.0
        last_review_year = 0
        last_review_month = 0

    if number_of_reviews > 0 and reviews_per_month < 0:
        st.warning(
            "Reviews per month cannot be negative."
        )
        st.stop()

    # Prepare exact model input
    input_data = prepare_input(
        neighbourhood_group=neighbourhood_group,
        neighbourhood=neighbourhood.strip(),
        latitude=latitude,
        longitude=longitude,
        room_type=room_type,
        minimum_nights=minimum_nights,
        number_of_reviews=number_of_reviews,
        reviews_per_month=reviews_per_month,
        calculated_host_listings_count=
            calculated_host_listings_count,
        availability_365=availability_365,
        last_review_year=last_review_year,
        last_review_month=last_review_month
    )

    try:
        # Model predicts log(1 + price)
        predicted_log_price = model.predict(input_data)[0]

        # Convert prediction back to USD
        predicted_price = np.expm1(predicted_log_price)

        # Prevent displaying a negative price due to model behaviour
        predicted_price = max(0.0, predicted_price)

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------
        st.success("Prediction generated successfully!")

        st.markdown("## 💵 Estimated Nightly Price")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Estimated Price",
                f"${predicted_price:,.0f} / night"
            )

        with result_col2:
            st.metric(
                "Room Type",
                room_type
            )

        with result_col3:
            st.metric(
                "Location",
                neighbourhood_group
            )

        st.divider()

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({
            "Input": [
                "Neighbourhood Group",
                "Neighbourhood",
                "Room Type",
                "Minimum Nights",
                "Number of Reviews",
                "Reviews per Month",
                "Host Listings",
                "Availability (days)",
                "Latitude",
                "Longitude"
            ],
            "Value": [
                neighbourhood_group,
                neighbourhood.strip(),
                room_type,
                minimum_nights,
                number_of_reviews,
                reviews_per_month,
                calculated_host_listings_count,
                availability_365,
                latitude,
                longitude
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "This is an ML-based estimate and should not be interpreted "
            "as a guaranteed market price."
        )

    except Exception as e:
        st.error(
            "Prediction failed. Please make sure that the saved "
            "pipeline was created using the same feature names and "
            "preprocessing workflow as Task 2."
        )

        with st.expander("Technical error"):
            st.exception(e)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()

st.caption(
    "NYC Airbnb Price Prediction • Machine Learning Task 3"
)
