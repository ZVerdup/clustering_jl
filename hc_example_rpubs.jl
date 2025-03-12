using Pkg
Pkg.add(["Clustering", "DataFrames", "Plots", "Distances"]) # Install necessary packages
using Clustering
using DataFrames
using Plots
using Distances
using Random

# 1. Generate or Load Data

# Option A: Generate synthetic data (for demonstration)
Random.seed!(123)
n_clusters = 3
n_points = 100
data = vcat(
    [rand(2) .+ [i, i] for i in 1:n_clusters]...
) * 5
data = transpose(hcat(data...))

# Option B: Load data from a file (replace with your actual file path)
# df = DataFrame(CSV.read("your_data.csv", DataFrame))
# data = Matrix(df[:, [:column1, :column2]])

# 2. Perform Hierarchical Clustering

# Choose a distance metric (see previous tutorial for options)
distance_metric = Euclidean() # Example: Euclidean distance

# Calculate the distance matrix
distance_matrix = pairwise(distance_metric, data, dims=1)

# Perform hierarchical clustering using a linkage method
# Common linkage methods: :single, :complete, :average, :ward
linkage_method = :complete # Example: Complete linkage

# Perform the clustering
tree = hclust(distance_matrix, method=linkage_method)

# 3. Analyze Results

# Cut the dendrogram to get cluster assignments
k = 3 # Desired number of clusters
cluster_assignments = cutree(tree, k=k)

# Dendrogram
p_dendrogram = plot(tree,
    xticks = false,
    xlabel = "Data Points",
    ylabel = "Distance",
    title = "Dendrogram",
    linecolor=:black,
    linewidth=1)
display(p_dendrogram)

# 4. Visualize Results

# Scatter plot of data points, colored by cluster assignment
p_scatter = scatter(data[:, 1], data[:, 2],
          group = cluster_assignments,
          xlabel = "X", ylabel = "Y",
          title = "Hierarchical Clustering (Complete Linkage)",
          markersize = 5,
          markerstrokewidth = 0,
          alpha = 0.8)

display(p_scatter)


# 5. Exploring Different Linkage Methods

# Compare the results with different linkage methods
linkage_methods = [:single, :complete, :average, :ward]
for method in linkage_methods
    tree = hclust(distance_matrix, method=method)
    cluster_assignments = cutree(tree, k=k)

    p = scatter(data[:, 1], data[:, 2],
          group = cluster_assignments,
          xlabel = "X", ylabel = "Y",
          title = "Hierarchical Clustering ($(method) Linkage)",
          markersize = 5,
          markerstrokewidth = 0,
          alpha = 0.8)
    display(p)
end

# 6. Using DataFrames (Optional)

# If you have your data in a DataFrame:
# df[:Cluster] = cluster_assignments
# groupby(df, :Cluster) # Analyze data within each cluster

#7. Plotting with Dendrogram and Clusters Side-by-Side
combined_plot = plot(p_dendrogram, p_scatter, layout = @layout [a{0.5w} b])
display(combined_plot)