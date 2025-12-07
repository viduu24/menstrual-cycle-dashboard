import streamlit as st

# Test imports one by one
try:
    from utils.data_loader import load_data, load_models
    st.success("✅ utils.data_loader imported successfully")
except Exception as e:
    st.error(f"❌ Error importing utils.data_loader: {e}")

try:
    from pages import page_readme
    st.success("✅ page_readme imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_readme: {e}")

try:
    from pages import page_data_description
    st.success("✅ page_data_description imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_data_description: {e}")

try:
    from pages import page_missingness
    st.success("✅ page_missingness imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_missingness: {e}")

try:
    from pages import page_cleaning
    st.success("✅ page_cleaning imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_cleaning: {e}")

try:
    from pages import page_information
    st.success("✅ page_information imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_information: {e}")

try:
    from pages import page_graphs
    st.success("✅ page_graphs imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_graphs: {e}")

try:
    from pages import page_ml_models
    st.success("✅ page_ml_models imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_ml_models: {e}")

try:
    from pages import page_guide
    st.success("✅ page_guide imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_guide: {e}")

try:
    from pages import page_predictions
    st.success("✅ page_predictions imported successfully")
except Exception as e:
    st.error(f"❌ Error importing page_predictions: {e}")

st.markdown("---")
st.markdown("### If all imports are ✅, replace app.py with the fixed version")
st.markdown("### If any imports show ❌, that page file has an error")
