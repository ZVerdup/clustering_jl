using Clustering, Random, Distances, StatsPlots

# Set seed for reproducibility
Random.seed!(42)

# Generate a random dataset with 1000 observations and 8 attributes
num_observations = 1000
num_attributes = 8
data = rand(num_observations, num_attributes)  # Random values between 0 and 1

# Compute the pairwise distance matrix (Euclidean distance)
distance_matrix = pairwise(Euclidean(), data', dims=1)

# Perform hierarchical clustering
hclust_result = hclust(distance_matrix, linkage=:ward)

# Cut the dendrogram into clusters (e.g., 5 clusters)
num_clusters = 5
cluster_assignments = cutree(hclust_result, k=num_clusters)

# Display the first few cluster assignments safely
println("First 10 cluster assignments: ", cluster_assignments[1:min(10, length(cluster_assignments))])

# Plot dendrogram using StatsPlots
dendrogram(hclust_result, xlabel="Observations", ylabel="Height", title="Hierarchical Clustering Dendrogram")
