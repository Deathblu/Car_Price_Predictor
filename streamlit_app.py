# import streamlit as st
# import joblib
# import numpy as np

# # Page config
# st.set_page_config(
#     page_title="Car Price Predictor",
#     page_icon="🚗",
#     layout="wide"
# )

# # Load model
# @st.cache_resource
# def load_model():
#     return joblib.load('final_model.pkl')

# model = load_model()

# # Title
# st.title("🚗 Car Price Predictor")
# st.markdown("Enter car specifications to get an estimated price using Gradient Boosting")

# # Categorical feature options
# categorical_features = {
#     'Fuel_type': ['gas', 'diesel'],
#     'make': ['std', 'turbo'],
#     'num_of_doors': ['two', 'four'],
#     'aspiration': ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
#     'wheel_base': ['rwd', 'fwd', '4wd'],
#     'engine_location': ['front', 'rear'],
#     'body_style': ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 
#                     'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury', 
#                     'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 
#                     'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'],
#     'engine_type': ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf'],
#     'num_of_cylinders': ['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'],
#     'fuel_system': ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi']
# }

# # Create two columns for layout
# col1, col2 = st.columns(2)

# # Numerical inputs
# with col1:
#     st.subheader("📊 Numerical Features")
    
#     symboling = st.number_input("Symboling", min_value=-3, max_value=3, value=0)
#     normalized_losses = st.number_input("Normalized Losses", min_value=0.0, value=120.0)
#     drive_wheels = st.number_input("Drive Wheels", min_value=0.0, value=100.0)
#     length = st.number_input("Length (inches)", min_value=0.0, value=170.0)
#     width = st.number_input("Width (inches)", min_value=0.0, value=65.0)
#     height = st.number_input("Height (inches)", min_value=0.0, value=54.0)
#     curb_weight = st.number_input("Curb Weight (lbs)", min_value=0.0, value=2500.0)
#     engine_size = st.number_input("Engine Size (cc)", min_value=0.0, value=2000.0)

# with col2:
#     st.subheader("⚙️ More Numerical Features")
    
#     bore = st.number_input("Bore", min_value=0.0, value=3.2, step=0.1)
#     stroke = st.number_input("Stroke", min_value=0.0, value=3.0, step=0.1)
#     compression_ratio = st.number_input("Compression Ratio", min_value=0.0, value=9.0, step=0.1)
#     horsepower = st.number_input("Horsepower (bhp)", min_value=0.0, value=120.0)
#     peak_rpm = st.number_input("Peak RPM", min_value=0.0, value=5000.0)
#     city_mpg = st.number_input("City MPG", min_value=0.0, value=25.0)
#     highway_mpg = st.number_input("Highway MPG", min_value=0.0, value=30.0)

# # Categorical inputs
# st.subheader("🏷️ Categorical Features")

# cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

# with cat_col1:
#     fuel_type = st.selectbox("Fuel Type", categorical_features['Fuel_type'])
#     make = st.selectbox("Make", categorical_features['make'])

# with cat_col2:
#     num_of_doors = st.selectbox("Number of Doors", categorical_features['num_of_doors'])
#     aspiration = st.selectbox("Aspiration", categorical_features['aspiration'])

# with cat_col3:
#     wheel_base = st.selectbox("Wheel Base", categorical_features['wheel_base'])
#     engine_location = st.selectbox("Engine Location", categorical_features['engine_location'])

# with cat_col4:
#     body_style = st.selectbox("Body Style", categorical_features['body_style'])
#     engine_type = st.selectbox("Engine Type", categorical_features['engine_type'])

# cat_col5, cat_col6 = st.columns(2)

# with cat_col5:
#     num_of_cylinders = st.selectbox("Number of Cylinders", categorical_features['num_of_cylinders'])

# with cat_col6:
#     fuel_system = st.selectbox("Fuel System", categorical_features['fuel_system'])

# # Create one-hot encoding function
# def create_one_hot(feature_name, value, options):
#     encoding = {}
#     for option in options:
#         encoding[f"{feature_name}_{option}"] = 1 if value == option else 0
#     return encoding

# # Predict button
# st.markdown("---")
# if st.button("🔮 Predict Price", type="primary", use_container_width=True):
#     # Create one-hot encodings
#     fuel_encoding = create_one_hot('Fuel_type', fuel_type, categorical_features['Fuel_type'])
#     make_encoding = create_one_hot('make', make, categorical_features['make'])
#     doors_encoding = create_one_hot('num_of_doors', num_of_doors, categorical_features['num_of_doors'])
#     aspiration_encoding = create_one_hot('aspiration', aspiration, categorical_features['aspiration'])
#     wheel_encoding = create_one_hot('wheel_base', wheel_base, categorical_features['wheel_base'])
#     location_encoding = create_one_hot('engine_location', engine_location, categorical_features['engine_location'])
#     body_encoding = create_one_hot('body_style', body_style, categorical_features['body_style'])
#     engine_encoding = create_one_hot('engine_type', engine_type, categorical_features['engine_type'])
#     cylinders_encoding = create_one_hot('num_of_cylinders', num_of_cylinders, categorical_features['num_of_cylinders'])
#     fuel_sys_encoding = create_one_hot('fuel_system', fuel_system, categorical_features['fuel_system'])
    
#     # Combine all features in exact order (74 total)
#     features = [
#         # Numerical (15)
#         symboling, normalized_losses, drive_wheels, length, width,
#         height, curb_weight, engine_size, bore, stroke,
#         compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg,
#         # Fuel_type (2)
#         fuel_encoding['Fuel_type_diesel'], fuel_encoding['Fuel_type_gas'],
#         # make (2)
#         make_encoding['make_std'], make_encoding['make_turbo'],
#         # num_of_doors (2)
#         doors_encoding['num_of_doors_four'], doors_encoding['num_of_doors_two'],
#         # aspiration (5)
#         aspiration_encoding['aspiration_convertible'], aspiration_encoding['aspiration_hardtop'],
#         aspiration_encoding['aspiration_hatchback'], aspiration_encoding['aspiration_sedan'],
#         aspiration_encoding['aspiration_wagon'],
#         # wheel_base (3)
#         wheel_encoding['wheel_base_4wd'], wheel_encoding['wheel_base_fwd'],
#         wheel_encoding['wheel_base_rwd'],
#         # engine_location (2)
#         location_encoding['engine_location_front'], location_encoding['engine_location_rear'],
#         # body_style (22)
#         body_encoding['body_style_alfa-romero'], body_encoding['body_style_audi'],
#         body_encoding['body_style_bmw'], body_encoding['body_style_chevrolet'],
#         body_encoding['body_style_dodge'], body_encoding['body_style_honda'],
#         body_encoding['body_style_isuzu'], body_encoding['body_style_jaguar'],
#         body_encoding['body_style_mazda'], body_encoding['body_style_mercedes-benz'],
#         body_encoding['body_style_mercury'], body_encoding['body_style_mitsubishi'],
#         body_encoding['body_style_nissan'], body_encoding['body_style_peugot'],
#         body_encoding['body_style_plymouth'], body_encoding['body_style_porsche'],
#         body_encoding['body_style_renault'], body_encoding['body_style_saab'],
#         body_encoding['body_style_subaru'], body_encoding['body_style_toyota'],
#         body_encoding['body_style_volkswagen'], body_encoding['body_style_volvo'],
#         # engine_type (6)
#         engine_encoding['engine_type_dohc'], engine_encoding['engine_type_l'],
#         engine_encoding['engine_type_ohc'], engine_encoding['engine_type_ohcf'],
#         engine_encoding['engine_type_ohcv'], engine_encoding['engine_type_rotor'],
#         # num_of_cylinders (7)
#         cylinders_encoding['num_of_cylinders_eight'], cylinders_encoding['num_of_cylinders_five'],
#         cylinders_encoding['num_of_cylinders_four'], cylinders_encoding['num_of_cylinders_six'],
#         cylinders_encoding['num_of_cylinders_three'], cylinders_encoding['num_of_cylinders_twelve'],
#         cylinders_encoding['num_of_cylinders_two'],
#         # fuel_system (8)
#         fuel_sys_encoding['fuel_system_1bbl'], fuel_sys_encoding['fuel_system_2bbl'],
#         fuel_sys_encoding['fuel_system_4bbl'], fuel_sys_encoding['fuel_system_idi'],
#         fuel_sys_encoding['fuel_system_mfi'], fuel_sys_encoding['fuel_system_mpfi'],
#         fuel_sys_encoding['fuel_system_spdi'], fuel_sys_encoding['fuel_system_spfi']
#     ]
    
#     # Make prediction
#     try:
#         features_array = np.array(features).reshape(1, -1)
#         prediction = model.predict(features_array)
        
#         # Display result
#         st.success("✅ Prediction Complete!")
#         st.markdown("### Estimated Car Price")
#         st.markdown(f"## ${prediction[0]:,.2f}")
        
#         # Show input summary
#         with st.expander("📋 Input Summary"):
#             st.write(f"**Features count:** {len(features)} (Expected: 74)")
#             st.write(f"**Body Style:** {body_style}")
#             st.write(f"**Engine:** {engine_size}cc, {horsepower} bhp, {num_of_cylinders} cylinders")
#             st.write(f"**Fuel:** {fuel_type}, {fuel_system}")
#             st.write(f"**Efficiency:** {city_mpg} city / {highway_mpg} highway MPG")
        
#     except Exception as e:
#         st.error(f"❌ Prediction failed: {str(e)}")
#         st.write(f"Feature count: {len(features)} (Expected: 74)")

# # Sidebar with info
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.write("This app uses a **Gradient Boosted Tree** model to predict car prices.")
    
#     st.header("📈 Model Info")
#     st.write("- **Features:** 74 total")
#     st.write("- **Numerical:** 15")
#     st.write("- **Categorical:** 59 (one-hot encoded)")
    
#     st.header("🎯 Quick Tips")
#     st.info("""
#     - Fill in realistic values
#     - Check fuel economy matches engine size
#     - Higher horsepower = higher price
#     - Luxury brands (BMW, Mercedes) cost more
#     """)
    
#     st.header("💡 Example Cars")
#     if st.button("Load Toyota Camry"):
#         st.rerun()

import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-price {
        font-size: 60px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .price-container {
        background: linear-gradient(135deg, #667eea22 0%, #764ba255 100%);
        border-radius: 20px;
        padding: 30px;
        border: 2px solid #667eea;
        margin: 20px 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        margin: 5px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('final_model.pkl')

model = load_model()

# Title with better styling
st.markdown("<h1 style='text-align: center; color: #667eea;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Get instant AI-powered price estimates using advanced Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

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

# Quick presets for demo
st.sidebar.header("🎯 Quick Presets")
preset = st.sidebar.selectbox("Load Example Car:", 
    ["Custom", "Toyota Camry", "BMW 3-Series", "Honda Civic", "Mercedes E-Class", "VW GTI"])

# Preset values
presets = {
    "Toyota Camry": {
        'symboling': 0, 'normalized_losses': 122, 'drive_wheels': 102.4, 'length': 175.6,
        'width': 66.5, 'height': 54.1, 'curb_weight': 2755, 'engine_size': 1998,
        'bore': 3.19, 'stroke': 3.03, 'compression_ratio': 9.5, 'horsepower': 140,
        'peak_rpm': 6000, 'city_mpg': 24, 'highway_mpg': 30,
        'fuel_type': 'gas', 'make': 'std', 'num_of_doors': 'four', 'aspiration': 'sedan',
        'wheel_base': 'fwd', 'engine_location': 'front', 'body_style': 'toyota',
        'engine_type': 'ohc', 'num_of_cylinders': 'four', 'fuel_system': 'mpfi'
    },
    "BMW 3-Series": {
        'symboling': 2, 'normalized_losses': 164, 'drive_wheels': 103.3, 'length': 176.8,
        'width': 64.8, 'height': 54.3, 'curb_weight': 2970, 'engine_size': 2494,
        'bore': 3.46, 'stroke': 3.10, 'compression_ratio': 8.0, 'horsepower': 189,
        'peak_rpm': 5200, 'city_mpg': 16, 'highway_mpg': 25,
        'fuel_type': 'gas', 'make': 'std', 'num_of_doors': 'four', 'aspiration': 'sedan',
        'wheel_base': 'rwd', 'engine_location': 'front', 'body_style': 'bmw',
        'engine_type': 'ohc', 'num_of_cylinders': 'six', 'fuel_system': 'mpfi'
    },
    "Honda Civic": {
        'symboling': 1, 'normalized_losses': 94, 'drive_wheels': 96.5, 'length': 163.4,
        'width': 63.9, 'height': 54.5, 'curb_weight': 2215, 'engine_size': 1488,
        'bore': 2.91, 'stroke': 3.03, 'compression_ratio': 9.2, 'horsepower': 100,
        'peak_rpm': 5500, 'city_mpg': 31, 'highway_mpg': 38,
        'fuel_type': 'gas', 'make': 'std', 'num_of_doors': 'four', 'aspiration': 'hatchback',
        'wheel_base': 'fwd', 'engine_location': 'front', 'body_style': 'honda',
        'engine_type': 'ohc', 'num_of_cylinders': 'four', 'fuel_system': 'mpfi'
    }
}

# Load preset if selected
if preset != "Custom" and preset in presets:
    preset_data = presets[preset]
else:
    preset_data = None

# Create two columns for layout
col1, col2 = st.columns(2)

# Numerical inputs
with col1:
    st.subheader("📊 Numerical Features")
    
    symboling = st.number_input("Symboling", min_value=-3, max_value=3, 
                                value=preset_data['symboling'] if preset_data else 0)
    normalized_losses = st.number_input("Normalized Losses", min_value=0.0, 
                                       value=float(preset_data['normalized_losses']) if preset_data else 120.0)
    drive_wheels = st.number_input("Drive Wheels", min_value=0.0, 
                                   value=float(preset_data['drive_wheels']) if preset_data else 100.0)
    length = st.number_input("Length (inches)", min_value=0.0, 
                            value=float(preset_data['length']) if preset_data else 170.0)
    width = st.number_input("Width (inches)", min_value=0.0, 
                           value=float(preset_data['width']) if preset_data else 65.0)
    height = st.number_input("Height (inches)", min_value=0.0, 
                            value=float(preset_data['height']) if preset_data else 54.0)
    curb_weight = st.number_input("Curb Weight (lbs)", min_value=0.0, 
                                 value=float(preset_data['curb_weight']) if preset_data else 2500.0)
    engine_size = st.number_input("Engine Size (cc)", min_value=0.0, 
                                 value=float(preset_data['engine_size']) if preset_data else 2000.0)

with col2:
    st.subheader("⚙️ More Numerical Features")
    
    bore = st.number_input("Bore", min_value=0.0, 
                          value=float(preset_data['bore']) if preset_data else 3.2, step=0.1)
    stroke = st.number_input("Stroke", min_value=0.0, 
                            value=float(preset_data['stroke']) if preset_data else 3.0, step=0.1)
    compression_ratio = st.number_input("Compression Ratio", min_value=0.0, 
                                       value=float(preset_data['compression_ratio']) if preset_data else 9.0, step=0.1)
    horsepower = st.number_input("Horsepower (bhp)", min_value=0.0, 
                                value=float(preset_data['horsepower']) if preset_data else 120.0)
    peak_rpm = st.number_input("Peak RPM", min_value=0.0, 
                              value=float(preset_data['peak_rpm']) if preset_data else 5000.0)
    city_mpg = st.number_input("City MPG", min_value=0.0, 
                              value=float(preset_data['city_mpg']) if preset_data else 25.0)
    highway_mpg = st.number_input("Highway MPG", min_value=0.0, 
                                 value=float(preset_data['highway_mpg']) if preset_data else 30.0)

# Categorical inputs
st.subheader("🏷️ Categorical Features")

cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

with cat_col1:
    fuel_type = st.selectbox("Fuel Type", categorical_features['Fuel_type'],
                            index=categorical_features['Fuel_type'].index(preset_data['fuel_type']) if preset_data else 0)
    make = st.selectbox("Make", categorical_features['make'],
                       index=categorical_features['make'].index(preset_data['make']) if preset_data else 0)

with cat_col2:
    num_of_doors = st.selectbox("Number of Doors", categorical_features['num_of_doors'],
                               index=categorical_features['num_of_doors'].index(preset_data['num_of_doors']) if preset_data else 1)
    aspiration = st.selectbox("Aspiration", categorical_features['aspiration'],
                             index=categorical_features['aspiration'].index(preset_data['aspiration']) if preset_data else 2)

with cat_col3:
    wheel_base = st.selectbox("Wheel Base", categorical_features['wheel_base'],
                             index=categorical_features['wheel_base'].index(preset_data['wheel_base']) if preset_data else 1)
    engine_location = st.selectbox("Engine Location", categorical_features['engine_location'],
                                  index=categorical_features['engine_location'].index(preset_data['engine_location']) if preset_data else 0)

with cat_col4:
    body_style = st.selectbox("Body Style", categorical_features['body_style'],
                             index=categorical_features['body_style'].index(preset_data['body_style']) if preset_data else 19)
    engine_type = st.selectbox("Engine Type", categorical_features['engine_type'],
                              index=categorical_features['engine_type'].index(preset_data['engine_type']) if preset_data else 2)

cat_col5, cat_col6 = st.columns(2)

with cat_col5:
    num_of_cylinders = st.selectbox("Number of Cylinders", categorical_features['num_of_cylinders'],
                                   index=categorical_features['num_of_cylinders'].index(preset_data['num_of_cylinders']) if preset_data else 0)

with cat_col6:
    fuel_system = st.selectbox("Fuel System", categorical_features['fuel_system'],
                              index=categorical_features['fuel_system'].index(preset_data['fuel_system']) if preset_data else 0)

# Create one-hot encoding function
def create_one_hot(feature_name, value, options):
    encoding = {}
    for option in options:
        encoding[f"{feature_name}_{option}"] = 1 if value == option else 0
    return encoding

# Predict button
st.markdown("---")
if st.button("🔮 Predict Price Now"):
    with st.spinner("🤖 AI is analyzing your car specifications..."):
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
        
        # Combine all features
        features = [
            symboling, normalized_losses, drive_wheels, length, width,
            height, curb_weight, engine_size, bore, stroke,
            compression_ratio, horsepower, peak_rpm, city_mpg, highway_mpg,
            fuel_encoding['Fuel_type_diesel'], fuel_encoding['Fuel_type_gas'],
            make_encoding['make_std'], make_encoding['make_turbo'],
            doors_encoding['num_of_doors_four'], doors_encoding['num_of_doors_two'],
            aspiration_encoding['aspiration_convertible'], aspiration_encoding['aspiration_hardtop'],
            aspiration_encoding['aspiration_hatchback'], aspiration_encoding['aspiration_sedan'],
            aspiration_encoding['aspiration_wagon'],
            wheel_encoding['wheel_base_4wd'], wheel_encoding['wheel_base_fwd'],
            wheel_encoding['wheel_base_rwd'],
            location_encoding['engine_location_front'], location_encoding['engine_location_rear'],
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
            engine_encoding['engine_type_dohc'], engine_encoding['engine_type_l'],
            engine_encoding['engine_type_ohc'], engine_encoding['engine_type_ohcf'],
            engine_encoding['engine_type_ohcv'], engine_encoding['engine_type_rotor'],
            cylinders_encoding['num_of_cylinders_eight'], cylinders_encoding['num_of_cylinders_five'],
            cylinders_encoding['num_of_cylinders_four'], cylinders_encoding['num_of_cylinders_six'],
            cylinders_encoding['num_of_cylinders_three'], cylinders_encoding['num_of_cylinders_twelve'],
            cylinders_encoding['num_of_cylinders_two'],
            fuel_sys_encoding['fuel_system_1bbl'], fuel_sys_encoding['fuel_system_2bbl'],
            fuel_sys_encoding['fuel_system_4bbl'], fuel_sys_encoding['fuel_system_idi'],
            fuel_sys_encoding['fuel_system_mfi'], fuel_sys_encoding['fuel_system_mpfi'],
            fuel_sys_encoding['fuel_system_spdi'], fuel_sys_encoding['fuel_system_spfi']
        ]
        
        try:
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            
            # Display beautiful result
            st.markdown('<div class="price-container">', unsafe_allow_html=True)
            st.markdown("### 💰 Estimated Market Value")
            
            # Format price nicely
            if prediction >= 1000000:
                price_str = f"\u20B9{prediction/1000000:.2f}M"
            elif prediction >= 1000:
                price_str = f"\u20B9{prediction/1000:.1f}K"
            else:
                price_str = f"\u20B9{prediction:.0f}"
            
            st.markdown(f'<div class="big-price">{price_str}</div>', unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #666; font-size: 18px;'>Exact: ${prediction:,.2f}</p>", unsafe_allow_html=True)
            
            # Price range estimate (±10%)
            lower = prediction * 0.9
            upper = prediction * 1.1
            st.markdown(f"<p style='text-align: center; color: #888;'>Typical range: \u20B9{lower:,.0f} - \u20B9{upper:,.0f}</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Key features summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Engine Power", f"{int(horsepower)} HP", f"{num_of_cylinders} cyl")
            with col_b:
                st.metric("Fuel Economy", f"{int(city_mpg)} / {int(highway_mpg)} MPG", "City / Highway")
            with col_c:
                st.metric("Weight", f"{int(curb_weight)} lbs", f"{engine_size:.0f}cc engine")
            
            # Confidence indicator
            st.markdown("---")
            st.success("✅ Prediction completed successfully using Gradient Boosted Tree model")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.header("📊 Model Stats")
    st.metric("Accuracy", "MAPE = 9.65%", "Low Rate of Error")
    st.metric("Features", "74 total", "15 numerical + 59 categorical")
    
    st.markdown("---")
    st.header("💡 Tips")
    st.info("""
    **For accurate predictions:**
    - Use realistic combinations
    - Match engine size to horsepower
    - Consider brand positioning
    - Check fuel economy is reasonable
    """)
    
    st.markdown("---")
    st.caption("Powered by Gradient Boosting 🚀")
