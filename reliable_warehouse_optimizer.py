import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import math
import random

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

# Simple implementation of K-means clustering without sklearn
def custom_kmeans(data, n_clusters, max_iterations=100):
    # data: numpy array of shape (n_samples, n_features)
    # n_clusters: number of clusters to find
    # Get dimensions
    n_samples, n_features = data.shape
    
    # Initialize centroids randomly
    random_indices = random.sample(range(n_samples), n_clusters)
    centroids = data[random_indices]
    
    # Main loop
    for _ in range(max_iterations):
        # Calculate distances to centroids
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            # Calculate Euclidean distance
            diff = data - centroids[i]
            distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))
        
        # Assign points to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros((n_clusters, n_features))
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                new_centroids[i] = np.mean(data[labels == i], axis=0)
            else:
                # If no points in cluster, keep old centroid
                new_centroids[i] = centroids[i]
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return centroids, labels

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
        assignments = np.zeros(len(coords), dtype=int)
        
    else:
        # For multiple warehouses, use custom K-means
        # Run the algorithm several times to get good starting points
        best_total_distance = float('inf')
        best_warehouse_locations = None
        best_assignments = None
        
        # Try a few different initializations to find the best one
        for attempt in range(5):
            if progress_bar:
                progress_bar.progress((attempt + 1) / 10)
                
            # Get initial centroids using custom K-means
            centroids, labels = custom_kmeans(coords, num_warehouses)
            
            # Refine with weighted version
            for iteration in range(5):
                if progress_bar:
                    progress_bar.progress(0.5 + (iteration + 1) / 10)
                
                # Calculate new warehouse locations as weighted average of assigned stores
                new_centroids = np.zeros_like(centroids)
                for i in range(num_warehouses):
                    mask = (labels == i)
                    if np.sum(mask) > 0:
                        weights = sales[mask] / np.sum(sales[mask])
                        new_centroids[i, 0] = np.sum(coords[mask, 0] * weights)
                        new_centroids[i, 1] = np.sum(coords[mask, 1] * weights)
                    else:
                        new_centroids[i] = centroids[i]
                
                # Reassign points
                distances = np.zeros((len(coords), num_warehouses))
                for i in range(num_warehouses):
                    # Calculate distances to each centroid
                    for j in range(len(coords)):
                        distances[j, i] = haversine_distance(
                            coords[j, 0], coords[j, 1],
                            new_centroids[i, 0], new_centroids[i, 1]
                        )
                
                labels = np.argmin(distances, axis=1)
                
                centroids = new_centroids
            
            # Calculate total weighted distance
            total_distance = 0
            for i in range(len(coords)):
                wh_idx = labels[i]
                dist = haversine_distance(
                    coords[i, 0], coords[i, 1],
                    centroids[wh_idx, 0], centroids[wh_idx, 1]
                )
                total_distance += dist * sales[i]
            
            # Keep best result
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_warehouse_locations = centroids
                best_assignments = labels
        
        warehouse_locations = best_warehouse_locations
        assignments = best_assignments
    
    # Create a DataFrame for warehouse locations
    warehouse_df = pd.DataFrame(warehouse_locations, columns=['latitude', 'longitude'])
    warehouse_df['id'] = range(1, num_warehouses + 1)
    
    # Assign each store to a warehouse
    df['assigned_warehouse'] = assignments + 1
    
    # Calculate distances for each store
    for i in range(len(df)):
        wh_idx = df.iloc[i]['assigned_warehouse'] - 1
        df.loc[df.index[i], 'distance_to_warehouse_miles'] = haversine_distance(
            df.iloc[i]['latitude'], 
            df.iloc[i]['longitude'],
            warehouse_locations[wh_idx, 0],
            warehouse_locations[wh_idx, 1]
        )
    
    # Calculate total weighted distance
    df['weighted_distance'] = df['distance_to_warehouse_miles'] * df['yearly_sales']
    
    return warehouse_df, df

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

# Function to geocode zip codes (simplified version with pre-defined coordinates)
def geocode_zip_codes(zip_codes):
    # This is a simplified version that uses a dictionary of pre-defined coordinates
    # for common US zip codes. In a real implementation, you'd use a geocoding API.
    
    # Sample coordinates for common US zip codes
    zip_coordinates = {
        '10001': {'latitude': 40.7506, 'longitude': -73.9971, 'city': 'New York', 'state': 'NY'},
        '90210': {'latitude': 34.0901, 'longitude': -118.4065, 'city': 'Beverly Hills', 'state': 'CA'},
        '60611': {'latitude': 41.8952, 'longitude': -87.6217, 'city': 'Chicago', 'state': 'IL'},
        '75201': {'latitude': 32.7845, 'longitude': -96.7967, 'city': 'Dallas', 'state': 'TX'},
        '33131': {'latitude': 25.7602, 'longitude': -80.1959, 'city': 'Miami', 'state': 'FL'},
        '98101': {'latitude': 47.6101, 'longitude': -122.3423, 'city': 'Seattle', 'state': 'WA'},
        '02108': {'latitude': 42.3582, 'longitude': -71.0637, 'city': 'Boston', 'state': 'MA'},
        '80202': {'latitude': 39.7534, 'longitude': -104.9999, 'city': 'Denver', 'state': 'CO'},
        '97205': {'latitude': 45.5202, 'longitude': -122.6834, 'city': 'Portland', 'state': 'OR'},
        '85004': {'latitude': 33.4521, 'longitude': -112.0747, 'city': 'Phoenix', 'state': 'AZ'},
        '63101': {'latitude': 38.6289, 'longitude': -90.1928, 'city': 'St. Louis', 'state': 'MO'},
        '30303': {'latitude': 33.7604, 'longitude': -84.3915, 'city': 'Atlanta', 'state': 'GA'},
        '19103': {'latitude': 39.9516, 'longitude': -75.1680, 'city': 'Philadelphia', 'state': 'PA'},
        '89101': {'latitude': 36.1719, 'longitude': -115.1400, 'city': 'Las Vegas', 'state': 'NV'},
        '48201': {'latitude': 42.3486, 'longitude': -83.0567, 'city': 'Detroit', 'state': 'MI'},
        # Add more as needed
    }
    
    results = {}
    for zip_code in zip_codes:
        if zip_code in zip_coordinates:
            results[zip_code] = zip_coordinates[zip_code]
        else:
            # For unknown zip codes, generate random coordinates in the continental US
            # This is just for demonstration purposes
            results[zip_code] = {
                'latitude': np.random.uniform(25, 49),
                'longitude': np.random.uniform(-125, -65),
                'city': 'Unknown',
                'state': 'Unknown'
            }
    
    return results

# Function to add geographic coordinates to the dataframe
def add_coordinates(df):
    progress_text = "Geocoding zip codes..."
    progress_bar = st.progress(0)
    
    # Get unique zip codes
    unique_zips = df['zip_code'].unique().tolist()
    total_zips = len(unique_zips)
    
    # Create a status placeholder
    status = st.empty()
    status.text(f"Processing zip codes...")
    
    # Get geocode results
    zip_data = geocode_zip_codes(unique_zips)
    
    # Create lists to store the results
    latitudes = []
    longitudes = []
    city_names = []
    state_names = []
    
    # Process each store
    for idx, row in df.iterrows():
        progress_bar.progress((idx + 1) / len(df))
        
        zip_code = row['zip_code']
        
        if zip_code in zip_data:
            latitudes.append(zip_data[zip_code]['latitude'])
            longitudes.append(zip_data[zip_code]['longitude'])
            city_names.append(zip_data[zip_code]['city'])
            state_names.append(zip_data[zip_code]['state'])
        else:
            # If zip not found, use random coordinates
            latitudes.append(np.random.uniform(25, 49))
            longitudes.append(np.random.uniform(-125, -65))
            city_names.append("Unknown")
            state_names.append("Unknown")
    
    # Add the data to the dataframe
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    df['city'] = city_names
    df['state'] = state_names
    
    # Clear the progress indicators
    progress_bar.empty()
    status.empty()
    
    st.success(f"✅ Successfully geocoded {df.shape[0]} zip codes.")
    return df

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

# Create color column for mapping
def create_color_column(df):
    # We need red, blue, and green channels between 0-255
    color_df = df.copy()
    
    # Get unique warehouse IDs
    warehouse_ids = color_df['assigned_warehouse'].unique()
    
    # Create a color mapping
    color_map = {}
    for i, wh_id in enumerate(warehouse_ids):
        if i == 0:
            color_map[wh_id] = [255, 0, 0]  # Red
        elif i == 1:
            color_map[wh_id] = [0, 0, 255]  # Blue
        elif i == 2:
            color_map[wh_id] = [0, 255, 0]  # Green
        elif i == 3:
            color_map[wh_id] = [255, 165, 0]  # Orange
        elif i == 4:
            color_map[wh_id] = [128, 0, 128]  # Purple
        else:
            # For more warehouses, generate random colors
            color_map[wh_id] = [
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ]
    
    # Apply the color mapping
    color_df['color'] = color_df['assigned_warehouse'].map(lambda x: color_map.get(x, [0, 0, 0]))
    
    return color_df

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
            # Map visualization
            st.subheader("Store and Warehouse Locations")
            
            # Create two separate map views
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Stores by Warehouse Assignment**")
                # Use built-in st.map for store locations
                # st.map can color points but we need to use Pandas format with lat/lon columns
                map_df = st.session_state.updated_df[['latitude', 'longitude']].copy()
                st.map(map_df)
            
            with col2:
                st.write("**Optimized Warehouse Locations**")
                # Use built-in st.map for warehouse locations
                wh_map_df = st.session_state.warehouse_df[['latitude', 'longitude']].copy()
                st.map(wh_map_df)
            
            st.info("💡 **Note:** Different colors on the map represent different warehouse assignments.")
            
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
