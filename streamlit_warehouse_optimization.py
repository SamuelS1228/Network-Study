# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

# Set page configuration
st.set_page_config(
    page_title="Warehouse Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This application helps you determine the optimal locations for warehouses based on your store locations 
and their sales volume. Upload your store data, select the number of warehouses you want to deploy, 
and see the optimized warehouse locations and metrics.
""")

# Sidebar
st.sidebar.header("Configuration")

# Load sample data
@st.cache_data
def load_sample_data():
    # Generate 100 random US store locations
    np.random.seed(42)
    num_stores = 100
    
    # Continental US bounds
    lat_min, lat_max = 24.396308, 49.384358
    lon_min, lon_max = -125.0, -66.93457
    
    latitudes = np.random.uniform(lat_min, lat_max, num_stores)
    longitudes = np.random.uniform(lon_min, lon_max, num_stores)
    sales = np.random.uniform(10000, 1000000, num_stores).astype(int)
    
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Sales': sales
    })
    
    return df

# Download sample data as CSV
def download_sample_csv():
    df = load_sample_data()
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_store_data.csv">Download sample CSV</a>'
    return href

# Option to use sample data
use_sample = st.sidebar.checkbox("Use sample data")

if not use_sample:
    st.sidebar.markdown("## Upload Store Data")
    st.sidebar.markdown("Upload a CSV file with columns: Latitude, Longitude, and Sales")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    st.sidebar.markdown(download_sample_csv(), unsafe_allow_html=True)
else:
    uploaded_file = None

# Number of warehouses slider
num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=10, value=3)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    # Distance weight vs. sales weight
    distance_weight = st.slider(
        "Distance Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Lower values prioritize minimizing distance, higher values prioritize serving high-sales stores"
    )
    sales_weight = 1.0 - distance_weight
    
    st.markdown(f"Sales Weight: {sales_weight:.2f}")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Clustering Algorithm",
        ["K-Means", "Weighted K-Means"],
        help="K-Means optimizes for distance only. Weighted K-Means accounts for store sales."
    )
    
    # Max iterations
    max_iter = st.slider("Max Iterations", min_value=10, max_value=500, value=100)

# Load data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = load_sample_data()
    return data

# Main function for optimizing warehouse locations
def optimize_warehouse_locations(data, num_warehouses, algorithm, max_iter, sales_weight):
    X = data[['Latitude', 'Longitude']].values
    
    if algorithm == "K-Means":
        # Use regular K-means
        kmeans = KMeans(n_clusters=num_warehouses, random_state=42, max_iter=max_iter)
        data['Cluster'] = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
    else:  # Weighted K-means
        # Normalize sales for weights
        weights = data['Sales'] / data['Sales'].max()
        
        # Initialize clusters randomly
        centroids = X[np.random.choice(len(X), num_warehouses, replace=False)]
        
        # Iterate to find optimal locations
        for _ in range(max_iter):
            # Calculate distances from each point to each centroid
            distances = cdist(X, centroids)
            
            # Assign points to closest centroid
            labels = np.argmin(distances, axis=1)
            
            # Update centroids based on weighted mean
            new_centroids = np.zeros_like(centroids)
            for i in range(num_warehouses):
                if np.sum(labels == i) > 0:
                    # Combine distance and sales weights
                    cluster_weights = weights[labels == i] ** sales_weight
                    # Update with weighted mean
                    new_centroids[i] = np.average(X[labels == i], axis=0, weights=cluster_weights)
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        # Assign final clusters
        distances = cdist(X, centroids)
        data['Cluster'] = np.argmin(distances, axis=1)
        centers = centroids
        
    # Create result dataframe with warehouse locations
    warehouse_df = pd.DataFrame(centers, columns=['Latitude', 'Longitude'])
    warehouse_df['Warehouse_ID'] = warehouse_df.index
    
    # Calculate distance from each store to its assigned warehouse
    for i, row in data.iterrows():
        cluster_id = row['Cluster']
        warehouse = warehouse_df.iloc[cluster_id]
        # Haversine distance calculation (approximate)
        R = 6371  # Earth radius in km
        lat1, lon1 = np.radians(row['Latitude']), np.radians(row['Longitude'])
        lat2, lon2 = np.radians(warehouse['Latitude']), np.radians(warehouse['Longitude'])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        data.loc[i, 'Distance_km'] = R * c
    
    # Calculate metrics per warehouse
    metrics = []
    for i in range(num_warehouses):
        cluster_stores = data[data['Cluster'] == i]
        metrics.append({
            'Warehouse_ID': i,
            'Latitude': warehouse_df.iloc[i]['Latitude'],
            'Longitude': warehouse_df.iloc[i]['Longitude'],
            'Num_Stores': len(cluster_stores),
            'Total_Sales': cluster_stores['Sales'].sum(),
            'Avg_Distance_km': cluster_stores['Distance_km'].mean(),
            'Max_Distance_km': cluster_stores['Distance_km'].max(),
            'Min_Distance_km': cluster_stores['Distance_km'].min()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    return data, warehouse_df, metrics_df

# Create the map visualization
def create_map(store_data, warehouse_data):
    # Center map on the mean lat/long
    center_lat = store_data['Latitude'].mean()
    center_lon = store_data['Longitude'].mean()
    
    # Create the folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")
    
    # Add store markers with clusters
    store_cluster = MarkerCluster(name="Stores").add_to(m)
    
    # Color palette for clusters
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple'
    ]
    
    # Add store markers
    for i, row in store_data.iterrows():
        cluster_id = int(row['Cluster'])
        color = colors[cluster_id % len(colors)]
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"Store {i}<br>Sales: ${int(row['Sales']):,}<br>Warehouse: {cluster_id}",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(store_cluster)
    
    # Add warehouse markers
    for i, row in warehouse_data.iterrows():
        warehouse_id = int(row['Warehouse_ID'])
        color = colors[warehouse_id % len(colors)]
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Warehouse {warehouse_id}",
            icon=folium.Icon(color=color, icon='industry', prefix='fa')
        ).add_to(m)
        
        # Draw lines from warehouse to each of its stores
        cluster_stores = store_data[store_data['Cluster'] == warehouse_id]
        for _, store in cluster_stores.iterrows():
            folium.PolyLine(
                locations=[
                    [row['Latitude'], row['Longitude']],
                    [store['Latitude'], store['Longitude']]
                ],
                color=color,
                weight=1,
                opacity=0.3
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Create charts for metrics visualization
def create_charts(store_data, metrics_df):
    # Create a figure with subplots
    charts = []
    
    # 1. Bar chart for number of stores per warehouse
    fig_stores = px.bar(
        metrics_df, 
        x='Warehouse_ID', 
        y='Num_Stores',
        title='Number of Stores per Warehouse',
        labels={'Warehouse_ID': 'Warehouse', 'Num_Stores': 'Number of Stores'},
        color='Warehouse_ID',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    charts.append(fig_stores)
    
    # 2. Bar chart for total sales per warehouse
    fig_sales = px.bar(
        metrics_df, 
        x='Warehouse_ID', 
        y='Total_Sales',
        title='Total Sales per Warehouse',
        labels={'Warehouse_ID': 'Warehouse', 'Total_Sales': 'Total Sales ($)'},
        color='Warehouse_ID',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_sales.update_layout(yaxis_tickformat=',.0f')
    charts.append(fig_sales)
    
    # 3. Average distance from stores to warehouses
    fig_distance = px.bar(
        metrics_df, 
        x='Warehouse_ID', 
        y='Avg_Distance_km',
        title='Average Distance to Stores (km)',
        labels={'Warehouse_ID': 'Warehouse', 'Avg_Distance_km': 'Average Distance (km)'},
        color='Warehouse_ID',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    charts.append(fig_distance)
    
    # 4. Histogram of distances
    fig_hist = px.histogram(
        store_data, 
        x='Distance_km',
        nbins=20,
        title='Distribution of Store-to-Warehouse Distances',
        labels={'Distance_km': 'Distance (km)', 'count': 'Number of Stores'},
        color='Cluster',
        marginal='box'
    )
    charts.append(fig_hist)
    
    # 5. Scatter plot of store sales vs distance
    fig_scatter = px.scatter(
        store_data,
        x='Distance_km',
        y='Sales',
        color='Cluster',
        title='Store Sales vs. Distance to Warehouse',
        labels={'Distance_km': 'Distance to Warehouse (km)', 'Sales': 'Store Sales ($)'},
        opacity=0.7,
        size='Sales',
        size_max=15
    )
    fig_scatter.update_layout(yaxis_tickformat=',.0f')
    charts.append(fig_scatter)
    
    return charts

# Main app logic
try:
    # Load data
    data = load_data(uploaded_file)
    
    # Check if data has the required columns
    required_cols = ['Latitude', 'Longitude', 'Sales']
    if not all(col in data.columns for col in required_cols):
        st.error(f"The uploaded file must contain the columns: {', '.join(required_cols)}")
        st.stop()
    
    # Display raw data
    with st.expander("Raw Store Data"):
        st.dataframe(data)
        
    # Run optimization when button is clicked
    if st.button("Optimize Warehouse Locations"):
        with st.spinner("Optimizing warehouse locations..."):
            # Run the optimization
            store_data, warehouse_data, metrics_df = optimize_warehouse_locations(
                data, 
                num_warehouses, 
                algorithm, 
                max_iter, 
                sales_weight
            )
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Map View", "Charts", "Detailed Results"])
            
            with tab1:
                st.subheader("Optimized Warehouse Locations")
                
                # Display the map
                m = create_map(store_data, warehouse_data)
                folium_static(m, width=1000, height=600)
                
                # Legend
                st.markdown("""
                **Legend:**
                - Markers with industry icon (🏭): Warehouses
                - Colored circles: Stores (color indicates warehouse assignment)
                - Lines: Connection between stores and their assigned warehouse
                """)
            
            with tab2:
                st.subheader("Performance Metrics")
                
                # Create and display charts
                charts = create_charts(store_data, metrics_df)
                
                # Display charts in a 2-column layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(charts[0], use_container_width=True)
                    st.plotly_chart(charts[2], use_container_width=True)
                
                with col2:
                    st.plotly_chart(charts[1], use_container_width=True)
                    st.plotly_chart(charts[3], use_container_width=True)
                
                # Full width for the last chart
                st.plotly_chart(charts[4], use_container_width=True)
            
            with tab3:
                st.subheader("Warehouse Details")
                
                # Display warehouse metrics
                st.dataframe(metrics_df.style.format({
                    'Total_Sales': '${:,.2f}', 
                    'Avg_Distance_km': '{:.2f}', 
                    'Max_Distance_km': '{:.2f}', 
                    'Min_Distance_km': '{:.2f}'
                }))
                
                # Display the optimized warehouse locations
                st.subheader("Warehouse Coordinates")
                st.dataframe(warehouse_data[['Warehouse_ID', 'Latitude', 'Longitude']])
                
                # Download results as CSV
                csv_warehouse = warehouse_data.to_csv(index=False)
                b64_warehouse = base64.b64encode(csv_warehouse.encode()).decode()
                href_warehouse = f'<a href="data:file/csv;base64,{b64_warehouse}" download="optimized_warehouses.csv">Download warehouse locations CSV</a>'
                st.markdown(href_warehouse, unsafe_allow_html=True)
                
                # Download full results as CSV
                csv_stores = store_data.to_csv(index=False)
                b64_stores = base64.b64encode(csv_stores.encode()).decode()
                href_stores = f'<a href="data:file/csv;base64,{b64_stores}" download="optimized_store_assignments.csv">Download store assignments CSV</a>'
                st.markdown(href_stores, unsafe_allow_html=True)
                
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your data format and try again.")
