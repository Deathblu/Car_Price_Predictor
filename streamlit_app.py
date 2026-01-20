import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('final_model.pkl')

model = load_model()

# Title
st.title("🚗 Car Price Predictor")
st.markdown("Enter car specifications to get an estimated price using Gradient Boosting")

# Categorical feature options
categorical_features = {
    'Fuel_type': ['gas', 'diesel'],
    'make': ['std', 'turbo'],
    'num_of_doors': ['two', 'four'],
    'aspiration': ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
    'wheel_base': ['rwd', 'fwd', '4wd'],
    'engine_location': ['front', 'rear'],
    'body_style': ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 
                    'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury', 
                    'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 
                    'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'],
    'engine_type': ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf'],
    'num_of_cylinders': ['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'],
    'fuel_system': ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi']
}

# Create two columns for layout
col1, col2 = st.columns(2)

# Numerical inputs
with col1:
    st.subheader("📊 Numerical Features")
    
    symboling = st.number_input("Symboling", min_value=-3, max_value=3, value=0)
    normalized_losses = st.number_input("Normalized Losses", min_value=0.0, value=120.0)
    drive_wheels = st.number_input("Drive Wheels", min_value=0.0, value=100.0)
    length = st.number_input("Length (inches)", min_value=0.0, value=170.0)
    width = st.number_input("Width (inches)", min_value=0.0, value=65.0)
    height = st.number_input("Height (inches)", min_value=0.0, value=54.0)
    curb_weight = st.number_input("Curb Weight (lbs)", min_value=0.0, value=2500.0)
    engine_size = st.number_input("Engine Size (cc)", min_value=0.0, value=2000.0)

with col2:
    st.subheader("⚙️ More Numerical Features")
    
    bore = st.number_input("Bore", min_value=0.0, value=3.2, step=0.1)
    stroke = st.number_input("Stroke", min_value=0.0, value=3.0, step=0.1)
    compression_ratio = st.number_input("Compression Ratio", min_value=0.0, value=9.0, step=0.1)
    horsepower = st.number_input("Horsepower (bhp)", min_value=0.0, value=120.0)
    peak_rpm = st.number_input("Peak RPM", min_value=0.0, value=5000.0)
    city_mpg = st.number_input("City MPG", min_value=0.0, value=25.0)
    highway_mpg = st.number_input("Highway MPG", min_value=0.0, value=30.0)

# Categorical inputs
st.subheader("🏷️ Categorical Features")

cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

with cat_col1:
    fuel_type = st.selectbox("Fuel Type", categorical_features['Fuel_type'])
    make = st.selectbox("Make", categorical_features['make'])

with cat_col2:
    num_of_doors = st.selectbox("Number of Doors", categorical_features['num_of_doors'])
    aspiration = st.selectbox("Aspiration", categorical_features['aspiration'])

with cat_col3:
    wheel_base = st.selectbox("Wheel Base", categorical_features['wheel_base'])
    engine_location = st.selectbox("Engine Location", categorical_features['engine_location'])

with cat_col4:
    body_style = st.selectbox("Body Style", categorical_features['body_style'])
    engine_type = st.selectbox("Engine Type", categorical_features['engine_type'])

cat_col5, cat_col6 = st.columns(2)

with cat_col5:
    num_of_cylinders = st.selectbox("Number of Cylinders", categorical_features['num_of_cylinders'])

with cat_col6:
    fuel_system = st.selectbox("Fuel System", categorical_features['fuel_system'])

# Create one-hot encoding function
def create_one_hot(feature_name, value, options):
    encoding = {}
    for option in options:
        encoding[f"{feature_name}_{option}"] = 1 if value == option else 0
    return encoding

# Predict button
st.markdown("---")
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    # Create one-hot encodings
    fuel_encoding = create_one_hot('Fuel_type', fuel_type, categorical_features['Fuel_type'])
    make_encoding = create_one_hot('make', make, categorical_features['make'])
    doors_encoding = create_one_hot('num_of_doors', num_of_doors, categorical_features['num_of_doors'])
    aspiration_encoding = create_one_hot('aspiration', aspiration, categorical_features['aspiration'])
    wheel_encoding = create_one_hot('wheel_base', wheel_base, categorical_features['wheel_base'])
    location_encoding = create_one_hot('engine_location', engine_location, categorical_features['engine_location'])
    body_encoding = create_one_hot('body_style', body_style, categorical_features['body_style'])
    engine_encoding = create_one_hot('engine_type', engine_type, categorical_features['engine_type'])
    cylinders_encoding = create_one_hot('num_of_cylinders', num_of_cylinders, categorical_features['num_of_cylinders'])
    fuel_sys_encoding = create_one_hot('fuel_system', fuel_system, categorical_features['fuel_system'])
    
    # Combine all features in exact order (74 total)
    features = [
        # Numerical (15)
        symboling, normalized_losses, drive_wheels, length, width,
        height, curb_weight, engine_size, bore, stroke,
        compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg,
        # Fuel_type (2)
        fuel_encoding['Fuel_type_diesel'], fuel_encoding['Fuel_type_gas'],
        # make (2)
        make_encoding['make_std'], make_encoding['make_turbo'],
        # num_of_doors (2)
        doors_encoding['num_of_doors_four'], doors_encoding['num_of_doors_two'],
        # aspiration (5)
        aspiration_encoding['aspiration_convertible'], aspiration_encoding['aspiration_hardtop'],
        aspiration_encoding['aspiration_hatchback'], aspiration_encoding['aspiration_sedan'],
        aspiration_encoding['aspiration_wagon'],
        # wheel_base (3)
        wheel_encoding['wheel_base_4wd'], wheel_encoding['wheel_base_fwd'],
        wheel_encoding['wheel_base_rwd'],
        # engine_location (2)
        location_encoding['engine_location_front'], location_encoding['engine_location_rear'],
        # body_style (22)
        body_encoding['body_style_alfa-romero'], body_encoding['body_style_audi'],
        body_encoding['body_style_bmw'], body_encoding['body_style_chevrolet'],
        body_encoding['body_style_dodge'], body_encoding['body_style_honda'],
        body_encoding['body_style_isuzu'], body_encoding['body_style_jaguar'],
        body_encoding['body_style_mazda'], body_encoding['body_style_mercedes-benz'],
        body_encoding['body_style_mercury'], body_encoding['body_style_mitsubishi'],
        body_encoding['body_style_nissan'], body_encoding['body_style_peugot'],
        body_encoding['body_style_plymouth'], body_encoding['body_style_porsche'],
        body_encoding['body_style_renault'], body_encoding['body_style_saab'],
        body_encoding['body_style_subaru'], body_encoding['body_style_toyota'],
        body_encoding['body_style_volkswagen'], body_encoding['body_style_volvo'],
        # engine_type (6)
        engine_encoding['engine_type_dohc'], engine_encoding['engine_type_l'],
        engine_encoding['engine_type_ohc'], engine_encoding['engine_type_ohcf'],
        engine_encoding['engine_type_ohcv'], engine_encoding['engine_type_rotor'],
        # num_of_cylinders (7)
        cylinders_encoding['num_of_cylinders_eight'], cylinders_encoding['num_of_cylinders_five'],
        cylinders_encoding['num_of_cylinders_four'], cylinders_encoding['num_of_cylinders_six'],
        cylinders_encoding['num_of_cylinders_three'], cylinders_encoding['num_of_cylinders_twelve'],
        cylinders_encoding['num_of_cylinders_two'],
        # fuel_system (8)
        fuel_sys_encoding['fuel_system_1bbl'], fuel_sys_encoding['fuel_system_2bbl'],
        fuel_sys_encoding['fuel_system_4bbl'], fuel_sys_encoding['fuel_system_idi'],
        fuel_sys_encoding['fuel_system_mfi'], fuel_sys_encoding['fuel_system_mpfi'],
        fuel_sys_encoding['fuel_system_spdi'], fuel_sys_encoding['fuel_system_spfi']
    ]
    
    # Make prediction
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        
        # Display result
        st.success("✅ Prediction Complete!")
        st.markdown("### Estimated Car Price")
        st.markdown(f"## ${prediction[0]:,.2f}")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.write(f"**Features count:** {len(features)} (Expected: 74)")
            st.write(f"**Body Style:** {body_style}")
            st.write(f"**Engine:** {engine_size}cc, {horsepower} bhp, {num_of_cylinders} cylinders")
            st.write(f"**Fuel:** {fuel_type}, {fuel_system}")
            st.write(f"**Efficiency:** {city_mpg} city / {highway_mpg} highway MPG")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.write(f"Feature count: {len(features)} (Expected: 74)")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a **Gradient Boosted Tree** model to predict car prices.")
    
    st.header("📈 Model Info")
    st.write("- **Features:** 74 total")
    st.write("- **Numerical:** 15")
    st.write("- **Categorical:** 59 (one-hot encoded)")
    
    st.header("🎯 Quick Tips")
    st.info("""
    - Fill in realistic values
    - Check fuel economy matches engine size
    - Higher horsepower = higher price
    - Luxury brands (BMW, Mercedes) cost more
    """)
    
    st.header("💡 Example Cars")
    if st.button("Load Toyota Camry"):
        st.rerun()