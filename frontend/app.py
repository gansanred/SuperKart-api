import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# GLOBAL CONFIGURATION & OPTIMIZED MEMORY LOADING
# =====================================================================
st.set_page_config(
    page_title="Retail Revenue Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "product_sales_pipeline_v1.joblib"

@st.cache_resource(show_spinner="Waking up serialized prediction matrices...")
def load_production_pipeline(path):
    """
    Leverages Streamlit resource caching to store the heavy LightGBM
    and transformer pipeline objects directly inside system memory frames.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"⚠️ Critical Pipeline Error: '{path}' was not found in the running path folder root directory.")
        return None

# Load the cached model resource object globally
sales_pipeline = load_production_pipeline(MODEL_PATH)

# =====================================================================
# LAYOUT RENDERING & BRAND HEADER HOOKS
# =====================================================================
st.title("📊 Store Product Sales Forecast Engine")
st.markdown(
    """
    Welcome to the Retail Inference portal. Use the navigation panels below to input distinct item parameters
    interactively, or upload high-throughput inventory array matrices for rapid bulk scoring loops.
    """
)

# Set up left sidebar navigation lane flags
st.sidebar.image("https://icons8.com", width=120)
st.sidebar.markdown("## 🧭 Application Controls")
app_mode = st.sidebar.radio("Select Operational Task Type:", ["Single Item Calculator", "Bulk Batch Transformer"])

# =====================================================================
# CONDITION BLOCKS 1: INDIVIDUAL POINT REALTIME CALCULATIONS
# =====================================================================
if app_mode == "Single Item Calculator":
    st.subheader("🛒 Single Transaction Online Evaluation")

    if sales_pipeline is None:
        st.warning("Please ensure your trained 'product_sales_pipeline_v1.joblib' is located within the app directory to continue feature processing.")
    else:
        # Construct form grid boundaries using layout vectors
        with st.form("inference_input_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📦 Product Categorization Vectors")
                product_id = st.text_input("Product Identifier ID", value="FD6114", help="Unique alphanumeric item code string identification.")
                product_weight = st.number_input("Product Weight Index", min_value=0.0, max_value=100.0, value=12.66, step=0.01)
                product_sugar_content = st.selectbox("Sugar Formulation Grouping", ["Low Sugar", "Regular", "No Sugar"])
                product_allocated_area = st.number_input("Allocated Display Surface Ratio", min_value=0.000, max_value=1.000, value=0.027, step=0.001, format="%.4f")
                product_type = st.selectbox("Product Core Super-Category", [
                    "Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
                    "Snack Foods", "Meat", "Household", "Hard Drinks", "Fruits and Vegetables",
                    "Soft Drinks", "Breads", "Others", "Starchy Foods", "Breakfast", "Seafood"
                ])
                product_mrp = st.number_input("Maximum Retail Price (MRP Index value)", min_value=0.0, value=117.08, step=0.01)

            with col2:
                st.markdown("#### 🏢 Store Matrix Parameters")
                store_id = st.text_input("Store Location Outlet ID", value="OUT004")
                store_establishment_year = st.number_input("Outlet Initialization Year", min_value=1950, max_value=2026, value=2009, step=1)
                store_size = st.selectbox("Outlet Real Estate Footprint Square Class", ["Small", "Medium", "High"])
                store_location_city_type = st.selectbox("Urban Density Demographics Class", ["Tier 1", "Tier 2", "Tier 3"])
                store_type = st.selectbox("Store Format Framework", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

            # Submit action button tracking boundary state changes
            submit_calculation = st.form_submit_button("Generate Sales Revenue Forecast", type="primary")

        if submit_calculation:
            # Map input parameters instantly to a row collection framework tracking target schema structure
            sample_dataframe = pd.DataFrame([{
                'Product_Id': product_id,
                'Product_Weight': product_weight,
                'Product_Sugar_Content': product_sugar_content,
                'Product_Allocated_Area': product_allocated_area,
                'Product_Type': product_type,
                'Product_MRP': product_mrp,
                'Store_Id': store_id,
                'Store_Establishment_Year': store_establishment_year,
                'Store_Size': store_size,
                'Store_Location_City_Type': store_location_city_type,
                'Store_Type': store_type
            }])

            # Evaluate frame target constraints instantly using cached memory layers
            try:
                raw_predicted_output = sales_pipeline.predict(sample_dataframe)[0]
                sanitized_revenue_metric = max(0.0, float(raw_predicted_output))

                # Visual output banner displaying localized metrics cleanly
                st.success(f"📈 **Predicted Product Store Sales Total Valuation:** `${sanitized_revenue_metric:,.2f}`")

                # Metric dashboard layout widgets for visual tracking enhancements
                m1, m2 = st.columns(2)
                m1.metric(label="Calculated Revenue Yield", value=f"${sanitized_revenue_metric:,.2f}")
                m2.metric(label="Unit Marginal Ceiling Value (MRP)", value=f"${product_mrp:,.2f}")

            except Exception as execution_error:
                st.error(f"An anomaly emerged during transformer execution pipelines: {str(execution_error)}")

# =====================================================================
# CONDITION BLOCKS 2: HIGH-THROUGHPUT BULK TRANSFORMS
# =====================================================================
else:
    st.subheader("📂 Bulk Inventory Batch Prediction Layer")
    st.markdown("Upload raw system `.csv` inventories matching your validation framework columns to execute vectorized batch transforms.")

    csv_file_layer = st.file_uploader("Upload Target Inventory Data File Arrays:", type=["csv"])

    if csv_file_layer is not None:
        try:
            # Direct internal parsing to memory matrix dataframe
            batch_evaluation_frame = pd.read_csv(csv_file_layer)

            # Column matching presence sanity checking pass
            required_identity_columns = ['Product_Id', 'Product_MRP', 'Store_Id']
            missing_structure_keys = [col for col in required_identity_columns if col not in batch_evaluation_frame.columns]

            if missing_structure_keys:
                st.error(f"❌ Input payload file missing crucial formatting column structures: {missing_structure_keys}")
            elif sales_pipeline is None:
                st.error("Cannot process batch arrays because model target pipeline artifact files are unallocated.")
            else:
                st.info(f"Successfully tracked file frame containing `{batch_evaluation_frame.shape[0]}` rows and `{batch_evaluation_frame.shape[1]}` tracking features.")

                if st.button("Execute Vectorized Batch Inference Loop", type="primary"):
                    with st.spinner("Executing optimized model pipeline scoring operations across file matrices..."):
                        # Extract row index predictions tracking raw LightGBM target structures
                        computed_batch_predictions = sales_pipeline.predict(batch_evaluation_frame)

                        # Clip potential negative algorithmic anomalies and inject results back into frame matrix
                        batch_evaluation_frame['Predicted_Sales_Total'] = np.clip(computed_batch_predictions, 0, None).round(2)

                        st.success("🎯 Vectorized dataset processing phase finalized successfully!")

                        # Displaying interactive tabular datasets mapping columns
                        display_cols = ['Product_Id', 'Product_Type', 'Product_MRP', 'Store_Id', 'Predicted_Sales_Total']
                        filtered_output_view = batch_evaluation_frame[[col for col in display_cols if col in batch_evaluation_frame.columns]]

                        st.dataframe(filtered_output_view, use_container_width=True)

                        # Render safe dynamic data file downloads payload options
                        exported_csv_data = batch_evaluation_frame.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Annotated Forecast Spreadsheet Records",
                            data=exported_csv_data,
                            file_name="sales_forecast_batch_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as system_anomaly:
            st.error(f"A structural file parsing exception truncated evaluation passes: {str(system_anomaly)}")

# Footer branding layout elements
st.sidebar.markdown("---")
