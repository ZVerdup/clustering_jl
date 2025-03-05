using Distances
using Clustering
using StatsPlots

# Generate some sample data
data = rand(1000, 8)

# Compute the distance matrix
dist_matrix = pairwise(Euclidean(), data)

# Perform hierarchical clustering
hc = hclust(dist_matrix, linkage=:average)

# Plot the dendrogram
plot(hc, labels=1:size(data, 1), title="Dendrogram", xlabel="Sample Index", ylabel="Distance")