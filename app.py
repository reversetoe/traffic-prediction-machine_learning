import streamlit as st
import pandas as pd
import joblib
import os
import sklearn.compose._column_transformer
from datetime import datetime

# ===============================
# sklearn compatibility patch
# ===============================
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Traffic Prediction System",
    page_icon="🚦",
    layout="wide"
)

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>

.main {
    background-color: #f5f7f9;
}

/* Enhanced Typography Target Rule for Bright, Bold Attribute Labels */
div[data-testid="stWidgetLabel"] p, 
label[data-testid="stWidgetLabel"] p,
.stSelectbox label, 
.stSlider label, 
.stNumberInput label {
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.5);
}

.stButton>button {
    width: 100%;
    border-radius: 6px;
    height: 3em;
    background-color: #007bff;
    color: white;
    font-weight: bold;
    font-size:16px;
}

.prediction-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    background-color: #ffffff;
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_assets():

    if not os.path.exists("traffic_model.pkl") or \
       not os.path.exists("preprocessor.pkl"):

        return None, None

    return (
        joblib.load("traffic_model.pkl"),
        joblib.load("preprocessor.pkl")
    )

model, preprocessor = load_assets()

if model is None:
    st.error("⚠ Model files missing")
    st.stop()

# ===============================
# HEADER
# ===============================
st.title("🚦 Smart City Traffic Prediction")

st.markdown("""
Predict traffic density using environmental and traffic parameters.
""")

st.divider()

# ===============================
# DROPDOWN DATA
# ===============================

area_options = [
    "Electronic City",
    "Hebbal",
    "Indiranagar",
    "Jayanagar",
    "Koramangala",
    "M.G. Road",
    "Whitefield",
    "Yeshwanthpur"
]

road_options = [
    "100 Feet Road",
    "Anil Kumble Circle",
    "Ballari Road",
    "CMH Road",
    "Hebbal Flyover",
    "Hosur Road",
    "ITPL Main Road",
    "Jayanagar 4th Block",
    "Marathahalli Bridge",
    "Sarjapur Road",
    "Silk Board Junction",
    "Sony World Junction",
    "South End Circle",
    "Trinity Circle",
    "Tumkur Road",
    "Yeshwanthpur Circle"
]

weather_options = [
    "Clear",
    "Rainy",
    "Foggy",
    "Cloudy"
]

# ===============================
# INPUT FORM
# ===============================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📍 Location & Environment")

    area_name = st.selectbox(
        "Area Name",
        area_options
    )

    road_name = st.selectbox(
        "Road / Intersection",
        road_options
    )

    weather = st.selectbox(
        "Weather Conditions",
        weather_options
    )

    # ===============================
    # DATE
    # ===============================

    st.subheader("🗓 Date Information")

    d1, d2, d3 = st.columns(3)

    day = d1.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=datetime.now().day
    )

    month = d2.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=datetime.now().month
    )

    day_of_week = d3.selectbox(
        "Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x:
        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )

    # ===============================
    # TIME
    # ===============================

    st.subheader("⏰ Time Information")

    t1, t2, t3 = st.columns(3)

    hour = t1.selectbox(
        "Hour",
        list(range(1,13))
    )

    minute = t2.selectbox(
        "Minute",
        list(range(60))
    )

    am_pm = t3.selectbox(
        "AM / PM",
        ["AM", "PM"]
    )

    # Convert to 24-hour format
    hour_24 = hour

    if am_pm == "PM" and hour != 12:
        hour_24 += 12

    if am_pm == "AM" and hour == 12:
        hour_24 = 0

with col2:

    st.subheader("📊 Traffic Metrics")

    utilization = st.slider(
        "Road Capacity Utilization",
        0.0,
        1.0,
        0.5
    )

    public_transport = st.slider(
        "Public Transport Usage",
        0.0,
        1.0,
        0.3
    )

    st.subheader("🚶 Pedestrian & Incident Data")

    incidents = st.number_input(
        "Incident Reports",
        min_value=0,
        max_value=100,
        value=0
    )

    pedestrians = st.number_input(
        "Pedestrian/Cyclist Count",
        min_value=0,
        max_value=500,
        value=50
    )

# ===============================
# RESULT AREA
# ===============================

result_area = st.empty()

st.markdown("---")

# ===============================
# PREDICTION
# ===============================

if st.button("🚀 Predict Traffic Volume"):

    try:

        # ===============================
        # CREATE INPUT DATAFRAME
        # ===============================

        input_data = pd.DataFrame({
            "Area Name":[area_name],
            "Road/Intersection Name":[road_name],
            "Weather Conditions":[weather],
            "Road Capacity Utilization":[utilization],
            "Incident Reports":[incidents],
            "Public Transport Usage":[public_transport],
            "Pedestrian and Cyclist Count":[pedestrians],
            "Day":[day],
            "Month":[month],
            "DayOfWeek":[day_of_week],
            "Hour":[hour_24],
            "Minute":[minute]
        })

        # ===============================
        # PREPROCESS + PREDICT
        # ===============================

        processed = preprocessor.transform(input_data)

        prediction = float(
            model.predict(processed)[0]
        )

        # ===============================
        # SAVE DATA TO CSV
        # ===============================

        save_data = pd.DataFrame({
            "Area":[area_name],
            "Road":[road_name],
            "Weather":[weather],
            "Utilization":[utilization],
            "PublicTransport":[public_transport],
            "Incidents":[incidents],
            "Pedestrians":[pedestrians],
            "Hour":[hour_24],
            "Minute":[minute],
            "Prediction":[prediction],
            "Timestamp":[
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ]
        })

        save_data.to_csv(
            "traffic_logs.csv",
            mode="a",
            header=not os.path.exists("traffic_logs.csv"),
            index=False
        )

        # ===============================
        # AUTO SCROLL
        # ===============================

        st.markdown("""
        <script>
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
        </script>
        """, unsafe_allow_html=True)
        
        # ===============================
        # RESULT DISPLAY
        # ===============================

        st.subheader("📈 Prediction Result")

        left, center, right = st.columns([1,2,1])

        with center:

            # Fixed: Prevented multi-line indented strings from being interpreted as a markdown code block
            html_card = f"""
<div style="padding:30px; border-radius:15px; text-align:center; background-color:white; box-shadow:0 6px 15px rgba(0,0,0,0.15); margin-top:20px;">
    <p style="font-size:20px; color:#666; margin-bottom:10px;">
        Estimated Traffic Volume
    </p>
    <h1 style="color:#007bff; font-size:60px; margin:0;">
        {prediction:,.2f}
    </h1>
    <p style="color:#888; margin-top:10px;">
        Vehicles per Hour
    </p>
</div>
"""
            st.markdown(html_card, unsafe_allow_html=True)

            # ===============================
            # TRAFFIC LEVEL
            # ===============================

            if prediction < 3000:
                st.success("✅ Low Traffic — smooth flow expected")
            elif prediction <= 5000:
                st.warning("🟠 Moderate Traffic — slowdowns possible")
            else:
                st.error("🔴 High Traffic — congestion likely")

            st.info("💾 Prediction saved successfully to traffic_logs.csv")

    except Exception as e:
        st.error("Prediction failed")
        st.write(e)