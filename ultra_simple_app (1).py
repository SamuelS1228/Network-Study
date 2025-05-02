import streamlit as st
import pandas as pd
import numpy as np
import math

# Set page configuration
st.set_page_config(
    page_title="Warehouse Location Optimizer",
    page_icon="📦"
)

# App title and intro
st.title("📦 Warehouse Location Optimizer")
st.markdown("""
This tool helps you find optimal warehouse locations based on your store sales and locations.
""")

# Sample data with coordinates
def generate_sample_data():
    # Sample locations across the US
    locations = [
        {'zip': '10001', 'city': 'New York', 'state': 'NY', 'lat': 40.7506, 'lng': -73.9971, 'sales': 2500000},
        {'zip': '90210', 'city': 'Beverly Hills', 'state': 'CA', 'lat': 34.0901, 'lng': -118.4065, 'sales': 4200000},
        {'zip': '60611', 'city': 'Chicago', 'state': 'IL', 'lat': 41.8952, 'lng': -87.6217, 'sales': 1800000},
        {'zip': '75201', 'city': 'Dallas', 'state': 'TX', 'lat': 32.7845, 'lng': -96.7967, 'sales': 3100000},
        {'zip': '33131', 'city': 'Miami', 'state': 'FL', 'lat': 25.7602, 'lng': -80.1959, 'sales': 2700000}
    ]
    
    df = pd.DataFrame(locations)
    df.rename(columns={'lng': 'longitude', 'lat': 'latitude', 'sales': 'yearly_sales'}, inplace=True)
    return df

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

# Simple warehouse optimizer (using weighted centroid)
def optimize_warehouse(df):
    # Calculate weighted centroid based on sales
    total_sales = df['yearly_sales'].sum()
    weighted_lat = (df['latitude'] * df['yearly_sales']).sum() / total_sales
    weighted_lng = (df['longitude'] * df['yearly_sales']).sum() / total_sales
    
    # Calculate distances from each store to the warehouse
    distances = []
    for _, row in df.iterrows():
        distance = haversine_distance(row['latitude'], row['longitude'], weighted_lat, weighted_lng)
        distances.append(distance)
    
    df['distance_miles'] = distances
    df['weighted_distance'] = df['distance_miles'] * df['yearly_sales']
    
    # Create warehouse dataframe
    warehouse_df = pd.DataFrame({
        'latitude': [weighted_lat],
        'longitude': [weighted_lng],
        'id': [1]
    })
    
    return warehouse_df, df

# Main application logic
def main():
    # Generate and display sample data
    df = generate_sample_data()
    
    st.write("Sample store data:")
    st.dataframe(df[['zip', 'city', 'state', 'yearly_sales']])
    
    # Run optimization
    if st.button("Find Optimal Warehouse Location"):
        with st.spinner("Calculating..."):
            warehouse_df, updated_df = optimize_warehouse(df.copy())
        
        st.success("✅ Optimization complete!")
        
        # Display metrics
        avg_distance = updated_df['distance_miles'].mean()
        total_weighted_distance = updated_df['weighted_distance'].sum()
        max_distance = updated_df['distance_miles'].max()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Distance", f"{avg_distance:.2f} miles")
        with col2:
            st.metric("Max Distance", f"{max_distance:.2f} miles")
        with col3:
            st.metric("Weighted Distance", f"{total_weighted_distance:,.2f}")
        
        # Display warehouse location
        st.subheader("Optimal Warehouse Location")
        st.write(f"Latitude: {warehouse_df['latitude'].iloc[0]:.4f}, Longitude: {warehouse_df['longitude'].iloc[0]:.4f}")
        
        # Display distances for each store
        st.subheader("Store Distances to Warehouse")
        st.dataframe(updated_df[['zip', 'city', 'state', 'yearly_sales', 'distance_miles']])

# Run the app
if __name__ == "__main__":
    main()
