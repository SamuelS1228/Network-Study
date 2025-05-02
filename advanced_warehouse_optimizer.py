import streamlit as st
import pandas as pd
import numpy as np
import io
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from geopy.geocoders import Nominatim
import time
import pydeck as pdk

# Set page configuration
st.set_page_config(
    page_title="Warehouse Location Optimizer",
    page_icon="📦",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #5a9;
    }
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# App title and intro
st.title("📦 Warehouse Location Optimizer")
st.markdown("""
This tool helps you find optimal warehouse locations based on your store sales and locations.
Upload an Excel file with zip codes and yearly sales data, or use the sample data to explore.
""")

# Haversine formula to calculate distance between two points on Earth
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3959  # Radius of Earth in miles
    
    return c * r

# Function to geocode zip codes
@st.cache_data
def geocode_zip_code(zip_code):
    """Geocode a single zip code and return lat/lng"""
    try:
        geolocator = Nominatim(user_agent=f"warehouse_optimizer_{zip_code}")
        location = geolocator.geocode(f"{zip_code}, USA", timeout=10)
        
        if location:
            return {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'address': location.address
            }
        return None
    except Exception:
        return None

# Function to process batch of zip codes with a progress bar
def geocode_zip_codes(zip_codes, progress_bar=None, status=None):
    results = {}
    total_zips = len(zip_codes)
    
    for i, zip_code in enumerate(zip_codes):
        if progress_bar:
            progress_bar.progress((i + 1) / total_zips)
        if status:
            status.text(f"Processing {i+1}/{total_zips} zip codes...")
        
        # Check if we already have this zip code geocoded
        if zip_code in results:
            continue
            
        result = geocode_zip_code(zip_code)
        if result:
            results[zip_code] = result
        
        # Respect usage limits with a small delay
        time.sleep(0.2)
    
    return results

# Function to add geographic coordinates to the dataframe
def add_coordinates(df):
    progress_text = "Geocoding zip codes..."
    progress_bar = st.progress(0)
    
    # Get unique zip codes to reduce API calls
    unique_zips = df['zip_code'].unique().tolist()
    total_zips = len(unique_zips)
    
    # Create a status placeholder
    status = st.empty()
    status.text(f"Processing 0/{total_zips} zip codes...")
    
    # Get geocode results
    zip_data = geocode_zip_codes(unique_zips, progress_bar, status)
    
    # Create lists to store the results
    latitudes = []
    longitudes = []
    city_names = []
    state_names = []
    
    # Process each store
    for idx, row in df.iterrows():
        zip_code = row['zip_code']
        
        if zip_code in zip_data:
            latitudes.append(zip_data[zip_code]['latitude'])
            longitudes.append(zip_data[zip_code]['longitude'])
            
            # Parse address for city and state
            address_parts = zip_data[zip_code]['address'].split(',')
            city_names.append(address_parts[-5].strip() if len(address_parts) > 5 else "Unknown")
            state_names.append(address_parts[-3].strip() if len(address_parts) > 3 else "Unknown")
        else:
            # Skip records without geo data
            latitudes.append(None)
            longitudes.append(None)
            city_names.append("Unknown")
            state_names.append("Unknown")
    
    # Add the data to the dataframe
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    df['city'] = city_names
    df['state'] = state_names
    
    # Remove rows with missing coordinates
    valid_df = df.dropna(subset=['latitude', 'longitude'])
    
    # Clear the progress indicators
    progress_bar.empty()
    status.empty()
    
    if len(valid_df) == 0:
        st.error("❌ Error: No valid coordinates found for any zip codes.")
        return None
    
    st.success(f"✅ Successfully geocoded {valid_df.shape[0]} out of {df.shape[0]} zip codes.")
    return valid_df

# Sample data with coordinates
def generate_sample_data():
    # Sample locations across the US
    locations = [
        {'zip_code': '10001', 'city': 'New York', 'state': 'NY', 'latitude': 40.7506, 'longitude': -73.9971, 'yearly_sales': 2500000},
        {'zip_code': '90210', 'city': 'Beverly Hills', 'state': 'CA', 'latitude': 34.0901, 'longitude': -118.4065, 'yearly_sales': 4200000},
        {'zip_code': '60611', 'city': 'Chicago', 'state': 'IL', 'latitude': 41.8952, 'longitude': -87.6217, 'yearly_sales': 1800000},
        {'zip_code': '75201', 'city': 'Dallas', 'state': 'TX', 'latitude': 32.7845, 'longitude': -96.7967, 'yearly_sales': 3100000},
        {'zip_code': '33131', 'city': 'Miami', 'state': 'FL', 'latitude': 25.7602, 'longitude': -80.1959, 'yearly_sales': 2700000},
        {'zip_code': '98101', 'city': 'Seattle', 'state': 'WA', 'latitude': 47.6101, 'longitude': -122.3423, 'yearly_sales': 1950000},
        {'zip_code': '02108', 'city': 'Boston', 'state': 'MA', 'latitude': 42.3582, 'longitude': -71.0637, 'yearly_sales': 1650000},
        {'zip_code': '80202', 'city': 'Denver', 'state': 'CO', 'latitude': 39.7534, 'longitude': -104.9999, 'yearly_sales': 1400000},
        {'zip_code': '97205', 'city': 'Portland', 'state': 'OR', 'latitude': 45.5202, 'longitude': -122.6834, 'yearly_sales': 950000},
        {'zip_code': '85004', 'city': 'Phoenix', 'state': 'AZ', 'latitude': 33.4521, 'longitude': -112.0747, 'yearly_sales': 1250000},
        {'zip_code': '63101', 'city': 'St. Louis', 'state': 'MO', 'latitude': 38.6289, 'longitude': -90.1928, 'yearly_sales': 1100000},
        {'zip_code': '30303', 'city': 'Atlanta', 'state': 'GA', 'latitude': 33.7604, 'longitude': -84.3915, 'yearly_sales': 2200000},
        {'zip_code': '19103', 'city': 'Philadelphia', 'state': 'PA', 'latitude': 39.9516, 'longitude': -75.1680, 'yearly_sales': 1850000},
        {'zip_code': '89101', 'city': 'Las Vegas', 'state': 'NV', 'latitude': 36.1719, 'longitude': -115.1400, 'yearly_sales': 3200000},
        {'zip_code': '48201', 'city': 'Detroit', 'state': 'MI', 'latitude': 42.3486, 'longitude': -83.0567, 'yearly_sales': 950000}
    ]
    
    df = pd.DataFrame(locations)
    return df

# Function to optimize warehouse locations
def optimize_warehouses(df, num_warehouses, progress_bar=None):
    # Extract coordinates and sales data
    coords = df[['latitude', 'longitude']].values
    sales = df['yearly_sales'].values
    
    if num_warehouses == 1:
        # For a single warehouse, use weighted average of coordinates
        total_sales = sales.sum()
        weighted_lat = np.sum(coords[:, 0] * sales) / total_sales
        weighted_lng = np.sum(coords[:, 1] * sales) / total_sales
        
        warehouse_locations = np.array([[weighted_lat, weighted_lng]])
        
    else:
        # For multiple warehouses, use weighted K-means clustering
        # Initialize with K-means to get starting points
        kmeans = KMeans(n_clusters=num_warehouses, random_state=42, n_init=10)
        kmeans.fit(coords)
        warehouse_locations = kmeans.cluster_centers_.copy()
        
        # Iterative optimization to account for sales weights
        max_iterations = 50
        for iteration in range(max_iterations):
            if progress_bar:
                progress_bar.progress((iteration + 1) / max_iterations)
                
            # Assign each store to the nearest warehouse
            distances = cdist(coords, warehouse_locations)
            assignments = np.argmin(distances, axis=1)
            
            # Calculate new warehouse locations as weighted average of assigned stores
            new_locations = np.zeros_like(warehouse_locations)
            for i in range(num_warehouses):
                mask = (assignments == i)
                if np.sum(mask) > 0:
                    weights = sales[mask] / np.sum(sales[mask])
                    new_locations[i, 0] = np.sum(coords[mask, 0] * weights)
                    new_locations[i, 1] = np.sum(coords[mask, 1] * weights)
                else:
                    new_locations[i] = warehouse_locations[i]
            
            # Check for convergence
            if np.allclose(warehouse_locations, new_locations, atol=1e-5):
                break
                
            warehouse_locations = new_locations
    
    # Create a DataFrame for warehouse locations
    warehouse_df = pd.DataFrame(warehouse_locations, columns=['latitude', 'longitude'])
    warehouse_df['id'] = range(1, num_warehouses + 1)
    
    # Assign each store to a warehouse
    distances = cdist(coords, warehouse_locations)
    df['assigned_warehouse'] = np.argmin(distances, axis=1) + 1
    
    # Calculate weighted distance for each store (using haversine distance for accuracy)
    for i in range(df.shape[0]):
        min_dist_idx = np.argmin(distances[i])
        df.loc[df.index[i], 'distance_to_warehouse_miles'] = haversine_distance(
            df.iloc[i]['latitude'], 
            df.iloc[i]['longitude'],
            warehouse_locations[min_dist_idx, 0],
            warehouse_locations[min_dist_idx, 1]
        )
    
    # Calculate total weighted distance
    df['weighted_distance'] = df['distance_to_warehouse_miles'] * df['yearly_sales']
    
    return warehouse_df, df

# Function to create a pydeck map
def create_map(store_df, warehouse_df):
    # Create a color palette for warehouses
    warehouse_colors = {
        1: [255, 0, 0],    # Red
        2: [0, 128, 255],  # Blue
        3: [0, 204, 0],    # Green
        4: [255, 128, 0],  # Orange
        5: [204, 0, 204],  # Purple
        6: [255, 255, 0],  # Yellow
        7: [0, 204, 204],  # Teal
        8: [255, 0, 255],  # Magenta
        9: [102, 102, 0],  # Olive
        10: [153, 51, 255] # Violet
    }
    
    # Create a column with hex colors for each warehouse
    color_column = []
    for i, row in store_df.iterrows():
        wh_id = int(row['assigned_warehouse'])
        if wh_id in warehouse_colors:
            color_column.append(warehouse_colors[wh_id])
        else:
            color_column.append([128, 128, 128])  # Gray for any warehouse > 10
    
    store_df['color'] = color_column
    
    # Create views for both stores and warehouses
    view_state = pdk.ViewState(
        latitude=store_df['latitude'].mean(),
        longitude=store_df['longitude'].mean(),
        zoom=3,
        pitch=0
    )
    
    # Create a layer for the stores
    store_layer = pdk.Layer(
        'ScatterplotLayer',
        data=store_df,
        get_position=['longitude', 'latitude'],
        get_color='color',
        get_radius=15000,  # Size based on radius in meters
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=6,
        radius_min_pixels=3,
        radius_max_pixels=30,
        line_width_min_pixels=1
    )
    
    # Create a layer for the warehouses
    warehouse_layer = pdk.Layer(
        'ScatterplotLayer',
        data=warehouse_df,
        get_position=['longitude', 'latitude'],
        get_color=[255, 255, 255],  # White
        get_radius=25000,  # Size based on radius in meters
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=6,
        radius_min_pixels=5,
        radius_max_pixels=40,
        line_width_min_pixels=2
    )
    
    # Set tooltip
    tooltip = {
        "html": "<b>Store Info:</b> <br/>"
                "<b>City:</b> {city}, {state} <br/>"
                "<b>Zip Code:</b> {zip_code} <br/>"
                "<b>Sales:</b> ${yearly_sales} <br/>"
                "<b>Assigned Warehouse:</b> {assigned_warehouse} <br/>"
                "<b>Distance to Warehouse:</b> {distance_to_warehouse_miles} miles",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    # Create the deck
    r = pdk.Deck(
        map_style='light',
        initial_view_state=view_state,
        layers=[store_layer, warehouse_layer],
        tooltip=tooltip
    )
    
    return r

# Function to process uploaded file
def process_file(uploaded_file):
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check if the dataframe has at least 2 columns
        if df.shape[1] < 2:
            st.error("❌ Error: The uploaded file must have at least 2 columns.")
            return None
            
        # Rename columns for clarity
        df.columns = ['zip_code', 'yearly_sales'] + list(df.columns[2:])
        
        # Data validation
        df['zip_code'] = df['zip_code'].astype(str).str.strip()
        df['zip_code'] = df['zip_code'].str.extract('(\d{5})', expand=False)
        df = df.dropna(subset=['zip_code', 'yearly_sales'])
        
        # Check if we have valid data after cleaning
        if df.shape[0] == 0:
            st.error("❌ Error: No valid data found after processing.")
            return None
            
        st.success(f"✅ Successfully loaded data with {df.shape[0]} store locations.")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

# Generate a sample Excel file
def generate_sample_excel():
    df = generate_sample_data()
    
    # Only keep essential columns for the example
    sample_df = df[['zip_code', 'yearly_sales']].copy()
    
    # Create a downloadable Excel file
    output = io.BytesIO()
    sample_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    
    return output

# Main application logic
def main():
    # Sidebar for controls
    st.sidebar.title("Configuration")
    
    # File upload section
    st.header("Step 1: Upload Your Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        Upload an Excel file with two columns:
        - Column A: Zip codes (5-digit US zip codes)
        - Column B: Yearly sales (numeric values)
        """)
        
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    with col2:
        st.markdown("Don't have a file? Download a sample:")
        
        # Generate and provide download of sample data
        sample_excel = generate_sample_excel()
        st.download_button(
            label="⬇️ Download Sample Excel File",
            data=sample_excel,
            file_name="warehouse_optimizer_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Or use sample data directly
        use_sample = st.checkbox("Use sample data instead", value=False)
    
    # Process the data
    if uploaded_file is not None:
        # Process the uploaded file
        df = process_file(uploaded_file)
        if df is None:
            return
    elif use_sample:
        # Use the sample data
        df = generate_sample_data()
        st.success("✅ Using sample data with 15 store locations across the US.")
    else:
        st.info("Please upload a file or use the sample data to continue.")
        return
    
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Add geocoding if needed
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.header("Step 2: Geocode Store Locations")
        st.info("Your data needs geocoding to convert zip codes to coordinates.")
        
        if st.button("Geocode Store Locations"):
            geo_df = add_coordinates(df)
            
            if geo_df is None:
                return
                
            # Store the geocoded dataframe in session state
            st.session_state.geo_df = geo_df
    else:
        # Data already has coordinates
        st.session_state.geo_df = df
    
    # Only show optimization if we have geocoded data
    if 'geo_df' in st.session_state:
        st.header("Step 3: Optimize Warehouse Locations")
        
        # Display a summary of the data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stores", f"{st.session_state.geo_df.shape[0]}")
        with col2:
            st.metric("Total Sales", f"${st.session_state.geo_df['yearly_sales'].sum():,.2f}")
        with col3:
            st.metric("Avg. Sales/Store", f"${st.session_state.geo_df['yearly_sales'].mean():,.2f}")
        
        # Warehouse configuration
        num_warehouses = st.sidebar.slider(
            "Number of Warehouses", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Select the number of warehouses to optimize for"
        )
        
        # Run optimization
        if st.button("Run Optimization"):
            with st.spinner("Optimizing warehouse locations..."):
                progress_bar = st.progress(0)
                
                warehouse_df, updated_df = optimize_warehouses(
                    st.session_state.geo_df.copy(), 
                    num_warehouses,
                    progress_bar
                )
                
                # Store results in session state
                st.session_state.warehouse_df = warehouse_df
                st.session_state.updated_df = updated_df
                
                # Clear the progress bar
                progress_bar.empty()
            
            st.success("✅ Optimization complete!")
            
            # Show metrics
            avg_distance = updated_df['distance_to_warehouse_miles'].mean()
            total_weighted_distance = updated_df['weighted_distance'].sum()
            max_distance = updated_df['distance_to_warehouse_miles'].max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Distance", f"{avg_distance:.2f} miles")
            with col2:
                st.metric("Max Distance", f"{max_distance:.2f} miles")
            with col3:
                st.metric("Weighted Distance", f"{total_weighted_distance:,.2f}")
    
    # Visualize results if we have completed optimization
    if 'warehouse_df' in st.session_state and 'updated_df' in st.session_state:
        st.header("Step 4: Results Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Interactive Map", "Warehouse Details"])
        
        with tab1:
            # Create the map visualization
            st.subheader("Store and Warehouse Locations")
            
            map_chart = create_map(
                st.session_state.updated_df,
                st.session_state.warehouse_df
            )
            
            # Display the map
            st.pydeck_chart(map_chart)
            
            st.info("💡 **Tip:** Scroll to zoom, click and drag to pan, hover over points for details.")
            
            # Add a download button for the updated data
            csv_data = st.session_state.updated_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_data,
                file_name="warehouse_optimization_results.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("Warehouse Details")
            
            # Show summary by warehouse
            for wh_id in range(1, num_warehouses + 1):
                warehouse_info = st.session_state.warehouse_df[st.session_state.warehouse_df['id'] == wh_id].iloc[0]
                assigned_stores = st.session_state.updated_df[st.session_state.updated_df['assigned_warehouse'] == wh_id]
                
                # Display summarized info
                with st.expander(f"Warehouse #{wh_id}", expanded=(wh_id == 1)):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"**Location:** ({warehouse_info['latitude']:.4f}, {warehouse_info['longitude']:.4f})")
                        st.markdown(f"**Number of stores:** {len(assigned_stores)}")
                        st.markdown(f"**Total yearly sales:** ${assigned_stores['yearly_sales'].sum():,.2f}")
                    
                    with col2:
                        st.markdown(f"**Average distance:** {assigned_stores['distance_to_warehouse_miles'].mean():.2f} miles")
                        st.markdown(f"**Maximum distance:** {assigned_stores['distance_to_warehouse_miles'].max():.2f} miles")
                        st.markdown(f"**Total weighted distance:** {assigned_stores['weighted_distance'].sum():,.2f}")
                    
                    # Display the list of assigned stores
                    st.subheader(f"Stores assigned to Warehouse #{wh_id}")
                    st.dataframe(
                        assigned_stores[['zip_code', 'city', 'state', 'yearly_sales', 'distance_to_warehouse_miles']]
                        .sort_values('distance_to_warehouse_miles')
                    )

# Add information about deployment
st.sidebar.markdown("---")
st.sidebar.header("About This App")
st.sidebar.info(
    "This app is hosted on Streamlit Community Cloud. "
    "The source code is available on GitHub. "
    "No installation required!"
)

# Run the app
if __name__ == "__main__":
    main()
