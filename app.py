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
    st.write("Silakan upload data yang ingin anda gunakan untuk melakukan segmentasi mahasiswa baru. Pastikan data yang diupload berformat CSV atau Excel (XLSX) dan minimal memiliki kolom sebagai berikut: PRODUCT, PRICE, PLACE, PROMOTION, PEOPLE, PROCESS, PHYSICAL EVIDENCE." 
             " Jika terdapat kolom lain selain yang disebutkan tadi, maka tidak masalah. Namun, pastikan kolom yang disebutkan tadi ada di dalam data yang diupload."
             " Untuk penulisan kolom, pastikan menggunakan huruf kapital dan tidak ada spasi.")
    st.write("**Contoh dataset yang benar:**")
    st.image("Contoh_Dataset.jpg", use_column_width=True)
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
        st.write("Centang Checkbox untuk menampilkan baris duplikat. Kemudian klik 'Remove Duplicates' untuk menghapus baris duplikat.")
        if st.checkbox("Show Duplicate Rows"):
            st.write(df[df.duplicated()])
        
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state['data'] = df  # Update session state
            st.success("Duplicates removed")
        st.write("Centang checkbox kembali untuk melakukan pengecekan setelah menghapus baris duplikat.")

        st.write("## Handle Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(df.isnull().sum())
        
        st.write("Pilih metode yang ingin dilakukan untk menghandle missing value lalu tekan tombol 'Handle Missing Values'")
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
        st.write("Centang checkbox untuk mengecek outliers di masing-masing kolom")
        # Fitur untuk menampilkan boxplot per kolom sesuai pilihan pengguna
        if st.checkbox("Show Boxplot for Outliers (Select a Column)"):
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            column = st.selectbox("Select a column", numeric_columns)
            fig, ax = plt.subplots()
            sns.boxplot(df[column], ax=ax)
            st.pyplot(fig)

        st.write("Centang checkbox untuk menampilkan semua boxplot dari seluruh kolom")
        # Fitur untuk menampilkan semua boxplot dari seluruh kolom numerik dalam satu gambar
        if st.checkbox("Show Boxplot for All Numeric Columns"):
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=df[numeric_columns], ax=ax)
                ax.set_title("Boxplot for All Numeric Columns")
                st.pyplot(fig)
            else:
                st.error("No numeric columns to display.")

        st.write("Lakukan pengecekan kembali setiap menghandle outliers. Jika masih terdapat outliers silakan lakukan penghandlean kembali")
        # Tombol untuk menangani outliers dengan median
        if st.button("Handle Outliers with Median"):
            outlier_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for col in outlier_columns:
                median = df[col].median()
                std_dev = df[col].std()

                upper_bound = median + 2 * std_dev
                lower_bound = median - 2 * std_dev

                # Buat mask untuk mendeteksi outliers
                upper_mask = df[col] > upper_bound
                lower_mask = df[col] < lower_bound

                # Gantikan outliers dengan median
                df.loc[upper_mask, col] = median
                df.loc[lower_mask, col] = median

            st.session_state['data'] = df  # Perbarui session state
            st.success("Outliers handled with median")
    
    elif choice == "Data Visualization":
        st.header("Data Visualization")
        
        st.write("## Correlation Heatmap")
        st.write("Heatmap di bawah menunjukkan korelasi antara kolom-kolom data. Semakin mendekati 1 maka korelasinya semakin kuat.")
        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        st.write("## Bar Plot")
        st.write("Menunjukkan banyak persebaran nilai tertentu dalam sebuah kolom menggunakan grafik batang")
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
        st.write("Pilih titik yang menjadi awal melandainya grafik sebagai titik siku pada grafik. Titik tersebut adalah jumlah cluster yang optimal untuk data Anda")
        
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

        st.write("Silakan input nilai klaster yang diinginkan. Nilai klaster dapat diganti sesuai dengan prefrensi Anda")
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
        st.write("Centang checkbox untuk menghasilkan visualisasi clustering dalam scatter plot")
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
                st.write(f"### Cluster {cluster}")
                cluster_subset = cluster_data[cluster_data['Cluster'] == cluster]

                # Calculate mean of each feature for the cluster
                cluster_mean = cluster_subset[features].mean()

                # Calculate the overall mean for the cluster (across all columns)
                overall_cluster_mean = cluster_mean.mean()
                
                # Find the column with the highest average value
                highest_avg_column = cluster_mean.idxmax()
                highest_avg_value = cluster_mean.max()

                # Display the insights for the cluster
                st.write(f"##### Insight untuk Cluster {cluster}:")
                st.write(f"- Rata-rata keseluruhan nilai di cluster ini adalah **{overall_cluster_mean:.2f}**.")
                st.write(f"- Kolom dengan rata-rata tertinggi adalah **{highest_avg_column}** dengan nilai **{highest_avg_value:.2f}**.")

            # Provide general overview of clusters
            st.write("### Overview of All Clusters:")
            overall_mean = cluster_data.groupby('Cluster').mean()
            st.write("Rata-rata keseluruhan per cluster:")
            st.write(overall_mean)
    
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
