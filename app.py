import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

st.title("Segmentasi Mahasiswa Baru App")

menu = ["Upload Data", "Data Preprocessing", "Data Visualization", "Clustering", "Evaluation"]
choice = st.sidebar.selectbox("Select a Menu", menu)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if choice == "Upload Data":
    st.header("Upload your data")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df
            st.write("Uploaded Data")
            st.write(df)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    features = ['PRODUCT', 'PRICE', 'PLACE', 'PROMOTION', 'PEOPLE', 'PROCESS', 'PHYSICAL EVIDENCE']
    
    # Ensure the selected features exist in the data
    features = [feature for feature in features if feature in df.columns]
    df = df[features]

    if choice == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        st.write("## Handle Duplicate Rows")
        if st.checkbox("Show Duplicate Rows"):
            st.write(df[df.duplicated()])
        
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state['data'] = df  # Update session state
            st.success("Duplicates removed")

        st.write("## Handle Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(df.isnull().sum())
        
        st.write("Choose a method to handle missing values and then click 'Handle Missing Values'")
        method = st.selectbox("Select a method to handle missing values", ["Drop rows with missing values", "Fill missing values with mean", "Fill missing values with median", "Fill missing values with specific value"])
        if method == "Fill missing values with specific value":
            specific_value = st.number_input("Enter the specific value to fill NaNs", value=0)

        if st.button("Handle Missing Values"):
            if method == "Drop rows with missing values":
                df = df.dropna()
            elif method == "Fill missing values with mean":
                df = df.fillna(df.mean())
            elif method == "Fill missing values with median":
                df = df.fillna(df.median())
            elif method == "Fill missing values with specific value":
                df = df.fillna(specific_value)
            
            st.session_state['data'] = df  # Update session state
            st.session_state['missing_values'] = df.isnull().sum()
            st.success("Missing values handled")

        st.write("## Handle Outliers")
        if st.checkbox("Show Boxplot for Outliers"):
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            column = st.selectbox("Select a column", numeric_columns)
            fig, ax = plt.subplots()
            sns.boxplot(df[column], ax=ax)
            st.pyplot(fig)

        if st.button("Handle Outliers with Median"):
            outlier_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for col in outlier_columns:
                median = df[col].median()
                std_dev = df[col].std()

                upper_bound = median + 2 * std_dev
                lower_bound = median - 2 * std_dev

                # Create masks for the outliers
                upper_mask = df[col] > upper_bound
                lower_mask = df[col] < lower_bound

                # Replace outliers with median
                df.loc[upper_mask, col] = median
                df.loc[lower_mask, col] = median

            st.session_state['data'] = df  # Update session state
            st.success("Outliers handled with median")

        st.write("## Scaling Data")
        if st.button("Scale Data"):
            try:
                scaler = StandardScaler()
                df[df.columns] = scaler.fit_transform(df[df.columns])
                st.session_state['data'] = df  # Update session state
                st.success("Data scaled successfully!")
                st.write(df)
            except Exception as e:
                st.error(f"Error: {e}")

        st.session_state['data'] = df
    
    elif choice == "Data Visualization":
        st.header("Data Visualization")
        
        st.write("## Correlation Heatmap")
        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        st.write("## Bar Plot")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        column = st.selectbox("Select a feature for bar plot", numeric_columns)

        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    elif choice == "Clustering":
        st.header("Clustering")
        
        st.write("## Elbow Method to Determine Optimal k")
        
        # Function to calculate inertia
        def calculate_inertia(df, k_range):
            inertia_values = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
                kmeans.fit(df)
                inertia_values.append(kmeans.inertia_)
            return inertia_values

        # Function to plot elbow
        def plot_elbow(k_range, inertia_values):
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertia_values, marker='o')
            plt.title('Elbow Method for Optimal k')
            plt.xlabel('Number of clusters k')
            plt.ylabel('Inertia (Sum of squared distances)')
            plt.xticks(k_range)
            plt.grid(True)
            st.pyplot(plt)

        k_range = range(1, 11)
        inertia_values = calculate_inertia(df, k_range)
        plot_elbow(k_range, inertia_values)

        k = st.number_input("Select the number of clusters (k)", min_value=2, max_value=10)
        
        def clustering(data, optimal_k):
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(data)

            inertia_scores = []
            random_states = []

            # Finding the best random state based on inertia scores
            for i in range(10):
                kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=i)
                kmeans.fit(pca_data)
                labels = kmeans.labels_

                # Store inertia scores and random states
                inertia_scores.append(kmeans.inertia_)
                random_states.append(i)

            # Fit final model with the best random state
            best_index = inertia_scores.index(min(inertia_scores))
            best_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=1000, n_init=10, random_state=random_states[best_index])
            best_kmeans.fit(pca_data)

            column_names = data.columns
            cluster_data = pd.DataFrame(data, columns=column_names)
            cluster_data['Cluster'] = best_kmeans.labels_

            return cluster_data, pca_data, best_kmeans

        if st.button("Run KMeans"):
            try:
                if df.isnull().sum().sum() == 0:
                    cluster_data, pca_data, best_kmeans = clustering(df, k)
                    st.session_state['cluster_data'] = cluster_data
                    st.session_state['pca_data'] = pca_data
                    st.session_state['best_kmeans'] = best_kmeans
                    st.success(f"KMeans clustering completed with {k} clusters")

                    # Show the cluster assignment
                    st.write("Clustered Data")
                    st.write(cluster_data[features + ['Cluster']].head())
                else:
                    st.error("Data contains NaN values. Please handle missing values in the 'Data Preprocessing' section.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.write("## Cluster Visualization")
        if 'cluster_data' in st.session_state:
            if st.checkbox("Show Cluster Scatter Plot"):
                try:
                    cluster_data = st.session_state['cluster_data']
                    pca_data = st.session_state['pca_data']
                    best_kmeans = st.session_state['best_kmeans']
                    df_pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
                    df_pca_df['Cluster'] = cluster_data['Cluster']

                    # Plotting the clusters and centroids
                    plt.figure(figsize=(10, 6))

                    for cluster in range(cluster_data['Cluster'].nunique()):
                        cluster_data_cluster = pca_data[cluster_data['Cluster'] == cluster]
                        scatter = plt.scatter(cluster_data_cluster[:, 0], cluster_data_cluster[:, 1], label=f'Cluster {cluster}')

                        # Adding centroid label
                        centroid = best_kmeans.cluster_centers_[cluster]
                        plt.scatter(centroid[0], centroid[1], s=150, c='red', marker='X')
                        plt.text(centroid[0], centroid[1], f'Centroid {cluster}', fontsize=12, ha='right')

                    plt.title('Clusters Visualized with PCA')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.colorbar(scatter)
                    plt.legend()
                    st.pyplot(plt)

                except KeyError as e:
                    st.error(f"Error: {e}")

        st.write("## Deskripsi Cluster")
        if 'cluster_data' in st.session_state:
            cluster_data = st.session_state['cluster_data']

            for cluster in sorted(cluster_data['Cluster'].unique()):
                st.write(f"**Cluster {cluster}**")
                cluster_subset = cluster_data[cluster_data['Cluster'] == cluster]
                cluster_description = cluster_subset[features].mean()
                st.write(cluster_description)
    
    elif choice == "Evaluation":
        st.header("Model Evaluation")
        if 'cluster_data' in st.session_state:
            cluster_data = st.session_state['cluster_data']
            pca_data = st.session_state['pca_data']
            best_kmeans = st.session_state['best_kmeans']
            
            # Calculate the silhouette score and DB index using the same clustering results
            silhouette_avg = silhouette_score(df, cluster_data['Cluster'])
            db_index = davies_bouldin_score(df, cluster_data['Cluster'])
            st.write(f"Silhouette Score: {silhouette_avg}")

            # Calculate silhouette scores and Davies-Bouldin indices for different k
            k_range = range(2, 11)
            silhouette_scores = []
            db_indices = []

            for k in k_range:
                # Re-clustering using the same process as before for each k
                kmeans = KMeans(n_clusters=k, random_state=best_kmeans.random_state)
                labels = kmeans.fit_predict(pca_data)
                silhouette_scores.append(silhouette_score(df, labels))
                db_indices.append(davies_bouldin_score(df, labels))

            # Plotting Silhouette Scores
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, marker='o', label='Silhouette Score')
            plt.title('Silhouette Score vs Number of Clusters')
            plt.xlabel('Number of clusters k')
            plt.ylabel('Silhouette Score')
            plt.xticks(k_range)
            plt.grid(True)
            st.pyplot(plt)

            # Plotting Davies-Bouldin Index
            st.write(f"Davies-Bouldin Index: {db_index}")
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, db_indices, marker='o', label='Davies-Bouldin Index')
            plt.title('Davies-Bouldin Index vs Number of Clusters')
            plt.xlabel('Number of clusters k')
            plt.ylabel('Davies-Bouldin Index')
            plt.xticks(k_range)
            plt.grid(True)
            st.pyplot(plt)

        else:
            st.write("Please perform clustering first")

else:
    st.write("No data available. Please upload a file in the 'Upload Data' section.")
