using CSV, DataFrames, XLSX, StatsBase, LinearAlgebra
using Plots, Clustering, MLJ, MLJBase, MultivariateStats, Statistics, Printf

# Function to plot dendrogram
function plot_dendrogram(model)
    n_samples = length(model.labels_)
    counts = zeros(Float64, size(model.children_, 1))
    for i in 1:size(model.children_, 1)
        current_count = 0
        for child_idx in model.children_[i, :]
            if child_idx < n_samples
                current_count += 1  # leaf node
            else
                current_count += counts[child_idx - n_samples + 1]
            end
        end
        counts[i] = current_count
    end
    linkage_matrix = hcat(model.children_, model.distances_, counts)
    plot(hclust(linkage_matrix), xticks=:none, yticks=:auto, legend=false)
end

# Function to convert array to DataFrame with named columns
function array_to_dataframe(array)
    if !(array isa AbstractArray)
        throw(ArgumentError("Input must be an array."))
    end
    num_columns = size(array, 2)
    column_names = ["PC$(i+1)" for i in 0:num_columns-1]
    return DataFrame(array, Symbol.(column_names))
end

# Define the parent directory and filename
initial_dir = raw"C:\Scripts\Julia\HRA_Clustering"
filename = "logs.csv"
full_path = joinpath(initial_dir, filename)
println(full_path)

# Read the data into a DataFrame
log_data = CSV.read(full_path, DataFrame; skipto=3)

# Display the data
header = names(log_data)
println(header)
println(log_data)

# Visualize the data
println("Visualizing data...")
# Define the number of subplots
num_plots = length(header) - 1  # Exclude the first column
plot_height = 8.0  # Height of the plot window in inches
plot_width = plot_height / 10  # Aspect ratio 1:10
# Create subplots
p = plot(layout = (1,num_plots), size=(Int(plot_width*100), Int(plot_height*100)))
# Loop through each column starting from the second
for (i, col) in enumerate(header[2:end])
    plot!(
        log_data[!, 1], log_data[!, col], 
        xlabel=string(header[1]), ylabel=string(col), 
        label="",   # No legend label
        xflip=true,  # Reverse the x-axis
        aspect_ratio = 1/10,
        subplot=i
    )
end
display(p)

# Prompt for resistivity variable
res_var = readline(stdin, "Enter the resistivity variable name: ")
if !(res_var in header)
    throw(ArgumentError("Invalid resistivity variable name."))
else
    res_values = log_data[!, res_var]
    println(res_values)
    println("Resistivity variable '$res_var' loaded successfully.")
end

# Apply log10 to resistivity variable
log_data[!, res_var] = log10.(log_data[!, res_var])

# Visualize log-transformed data
for col in header
    if eltype(log_data[!, col]) <: Number
        plot(log_data[!, col], title="Log-Transformed $col", label=col)
    end
end

# Standardize the data
scaled_log_data = (log_data .- mean(log_data, dims=1)) ./ std(log_data, dims=1)
scaled_log_data_df = DataFrame(scaled_log_data, names(log_data))
println(scaled_log_data_df)

# Perform PCA
pca_model = fit(PCA, scaled_log_data_df[:, 2:end]; maxoutdim=0.95)
pca_data = transform(pca_model, scaled_log_data_df[:, 2:end])
pca_data_df = array_to_dataframe(pca_data)
println(pca_model.principalvars / sum(pca_model.principalvars))

# Perform Hierarchical Clustering
println("Performing Hierarchical Clustering...")
hc_model = hclust(pca_data, linkage=:ward)
plot(hc_model, xticks=:none, yticks=:auto, legend=false, title="Hierarchical Clustering Dendrogram")

# Perform K-means Cluster Analysis
num_clusters = parse(Int, readline(stdin, "Enter the number of clusters to use: "))
start_time = time()
kmeans_model = kmeans(pca_data, num_clusters, maxiter=100, init=:kmeanspp)
elapsed_time = time() - start_time
println(@sprintf("Elapsed time: %.2f seconds", elapsed_time))
println(@sprintf("Minimum distance: %.2f", minimum(kmeans_model.assignments)))
