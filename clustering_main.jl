println("Loading packages...")
using CSV
println("Loaded CSV...")
using DataFrames
println("Loaded DataFrames...")
using StatsBase
println("Loaded StatsBase...") 
# using LinearAlgebra
using CairoMakie
println("Loaded CairoMakie...")
# using Plots 
using Clustering
println("Loaded Clustering...") 
using MultivariateStats
println("Loaded MultivariateStats...")
using Statistics
println("Loaded Statistics...") 
using Printf
println("Loaded Printf...")
# using MLJ, MLJBase 

# # # Wait for keyboard input to continue
# wait_for_key(prompt) = (println(stdout, prompt);
# 	read(stdin, 1); 
# 	while bytesavailable(stdin)>0 read(stdin, Char); end; 
# 	nothing)

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

# # Function to convert array to DataFrame with named columns
# function array_to_dataframe(array)
#     if !(array isa AbstractArray)
#         throw(ArgumentError("Input must be an array."))
#     end
#     num_columns = size(array, 2)
#     column_names = ["PC$(i+1)" for i in 0:num_columns-1]
#     return DataFrame(array, Symbol.(column_names))
# end

function zscore_transform(df)
    # Calculate the mean and standard deviation for each column
    zscored_df = df  # Initialize the z-scored DataFrame with the original dataframe
    means = 
    for col in names(zscored_df)
        means = mean(zscored_df[:,col])  # Mean for each column
        stds = std(zscored_df[:,col])    # Standard deviation for each column
        # Apply the z-score transformation
        zscored_df[:,col] = (zscored_df[:,col] .- means) ./ stds
    end
    print(means)  # Print the means used for transformation
    return zscored_df  # Return the z-scored DataFrame and the means and standard deviations used for transformation
end

# Define the parent directory and filename
# initial_dir = raw"C:\Scripts\Julia\HRA_Clustering"
initial_dir = raw"C:\Users\evank\Documents\Clustering\clustering_jl\clustering_jl"
filename = "logs.csv"
full_path = joinpath(initial_dir, filename)
println(full_path)

#wait_for_key("Ready to read data. Press ENTER to continue...")
println("Reading data...")
# Read the data into a DataFrame
log_data = CSV.read(full_path, DataFrame; skipto=3)

# Display the data
header = names(log_data)
println(header)
# println(log_data)

# Visualize the data
println("Visualizing data...")
# Define the number of subplots
num_plots = length(header) - 1  # Exclude the first column
plot_height = 1000  # Height of the plot window in inches
plot_width = 500  # Aspect ratio 1:10
# # Create subplots using Plots.jl, this doesn't look good yet
# p = plot(layout = (1,num_plots), size=(plot_height, plot_width))
# # Loop through each column starting from the second
# for i = 2:length(header)
#     x_min = minimum(log_data[:,i])
#     x_max = maximum(log_data[:,i])
#     dx=x_max-x_min
#     xtick_arr = x_min:dx/2:x_max
#     println(dx)
#     println(xtick_arr)
#     ticklabels = [@sprintf("%1.1f",x) for x in xtick_arr]
#     plot!(
#         log_data[:, i], log_data[:, 1], 
#         #xlabel=string(header[1]), ylabel=string(col), 
#         #label="",   # No legend label
#         #xflip=true,  # Reverse the x-axis
#         #aspect_ratio = 1/10,
#         yflip = true,  # Reverse the y-axis
#         xticks=(xtick_arr,ticklabels),
#         xrotation = 90,
#         subplot=i-1
#     )
# end
# display(p)
# Create subplots using Makie
f1 = Figure()
# Create a new axis for each subplot and plot the data
for j in 1:num_plots
    # Hide the y-axis for all but the first subplot
    if j==1  
        ylabelflag = true
    else
        ylabelflag = false
    end 
    ax = Axis(f1[1, j],
    #title="Log Data", 
    xlabel=header[j+1],
    ylabel=header[1],
    width = 100, 
    height = 900,
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yminorgridvisible = true,
    yticklabelsvisible = ylabelflag,
    ylabelvisible = ylabelflag,
    xaxisposition = :top,
    xticklabelrotation = 45,  
    yreversed = true, 
    )  
    lines!(ax, log_data[:, j+1], log_data[:, 1])  # Plot the data
end
Label(f1[0, :], "Log Data", fontsize = 20)
resize_to_layout!(f1)
save("log_data.png", f1)  # Save the figure to a file
println("Saved log data fig")
# display(f1)
# wait_for_key("Close Log Data figure then press ENTER to continue...")

# Prompt for resistivity variable, this doesn't work. It returns the rest of the code as a string.
print("Enter the resistivity variable name: ")
res_var = readline()
println("You entered: ", res_var)
res_flag = false
for i in eachindex(header)
    if contains(res_var,header[i])
        println("Resistivity variable '$res_var' found in header.", " Column: ", i)
        global res_flag = true
        break
    end
end
# Define res_var in case the column was not found
if res_flag == false
    println("Didn't find resistivity variable. Enter column number:")
    res_col_num = readline()
    res_col_num = parse(Int64,res_col_num)  # Convert the input to an integer
    res_var = header[res_col_num]  # Assign the resistivity variable name to the column number entered by the user
    println("Resistivity variable set to: column ", res_var)  # Print the resistivity variable name to the console for verification
end

# Copy log_data dataframe and apply log10 scaling to resistivity variable
log_data_logrt = log_data
log_data_logrt[:, res_var] = log10.(log_data[:, res_var])

# Visualize log-transformed data
println("Visualizing log-transformed data...")
f2 = Figure()
# Create a new axis for each subplot and plot the data
for j in 1:num_plots
    # Hide the y-axis for all but the first subplot
    if j==1  
        ylabelflag = true
    else
        ylabelflag = false
    end 
    ax = Axis(f2[1, j],
    #title="Log Data", 
    xlabel=header[j+1],
    ylabel=header[1],
    width = 100, 
    height = 900,
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yminorgridvisible = true,
    yticklabelsvisible = ylabelflag,
    ylabelvisible = ylabelflag,
    xaxisposition = :top,
    xticklabelrotation = 45,  
    yreversed = true, 
    )  
    lines!(ax, log_data_logrt[:, j+1], log_data_logrt[:, 1])  # Plot the data
end
Label(f2[0, :], rich("Log Data with Log", subscript("10"), " Resistivity"), fontsize = 20)
resize_to_layout!(f2)
save("logrt_data.png", f2)
println("Saved log resistivity figure")
# display(f2)
# wait_for_key("Close log standardized resistivity figure then press ENTER to continue...")

# # Standardize the data
# # Using StatsBase.jl for standardization
# # First convert the DataFrame to an Array for standardization
# array_logrt = Matrix(log_data_logrt)
# scaler = fit(ZScoreTransform, array_logrt)  # Returns a ZScoreTransform object
# standardized_data = StatsBase.transform(scaler, array_logrt)  # Returns an Array{Float64,1}
# # Convert back to datafram and visualize
# standardized_data_df = DataFrame(standardized_data, header)
# standardized_data_df[:,1] = log_data[:,1]  # Replace the first column with the original data (not standardized)
# Standarize the data using a custom function
println("Standardizing data with z-score transformation...")
function zscore_transform(df::DataFrame)
    cols = Dict()
    for (colname, col) in pairs(eachcol(df))
        if eltype(col) <: Number
            col_mean = mean(col)
            col_std = std(col)
            # println("Column: ", colname, " | Mean: ", col_mean, " | Std: ", col_std)
            cols[colname] = (col .- col_mean) ./ col_std
        end
    end
    return DataFrame(cols)
end
zscored_log_data_df = zscore_transform(log_data_logrt)  # Returns a DataFrame with z-scored data
# Replace the first column with the original data (not standardized)
zscored_log_data_df[:,1] = log_data[:,1]  # Replace the first column with the original data (not standardized)

# Visualize standardized data
println("Visualizing z-score standardized data...")
f3 = Figure()
# Create a new axis for each subplot and plot the data
for j in 1:num_plots
    # Hide the y-axis for all but the first subplot
    if j==1  
        ylabelflag = true
    else
        ylabelflag = false
    end 
    ax = Axis(f3[1, j],
    #title="Log Data", 
    xlabel=header[j+1],
    ylabel=header[1],
    width = 100, 
    height = 900,
    yminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yminorgridvisible = true,
    yticklabelsvisible = ylabelflag,
    ylabelvisible = ylabelflag,
    xaxisposition = :top,
    xticklabelrotation = 45,  
    yreversed = true, 
    )  
    lines!(ax, zscored_log_data_df[:, j+1], zscored_log_data_df[:, 1])  # Plot the data
end
Label(f3[0, :], "Z-score Standardized Log Data", fontsize = 20)
resize_to_layout!(f3)
save("zscaled_data.png", f3)
println("Saved scaled data fig")
#display(f3)

# # Perform PCA
println("Performing PCA...")
# pca_model = fit(PCA, zscored_log_data_df[:, 2:end]; maxoutdim=0.95)
# pca_data = transform(pca_model, scaled_log_data_df[:, 2:end])
# pca_data_df = array_to_dataframe(pca_data)
# println(pca_model.principalvars / sum(pca_model.principalvars))
function perform_pca(df::DataFrame, num_components::Int)
    matrix = Matrix(df[:,2:end])  # Convert DataFrame to a matrix
    pca_model = fit(PCA, matrix; maxoutdim=size(matrix,2))  # Fit PCA model
    transformed_data = predict(pca_model, matrix)  # Transform data
    return DataFrame(transformed_data', :auto)  # Convert back to DataFrame
end

function perform_pca(df::DataFrame)
    data = Matrix(df[:,2:end])'  # Convert DataFrame to a matrix
    pca_model = fit(PCA, data; maxoutdim=size(data,2))  # Fit PCA model
    explained_variance = pca_model.prinvars ./ sum(pca_model.prinvars)  # Calculate explained variance
    return explained_variance
end

function elbow_analysis(data::DataFrame)
    explained_variance = perform_pca(data)
    cumulative_variance = cumsum(explained_variance)
    num_components = 1:length(cumulative_variance)
    
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
        title = "Elbow Analysis", 
        xlabel = "Number of Components", 
        ylabel = "Cumulative Explained Variance"
    )
    
    lines!(ax, num_components, explained_variance, label = "Explained Variance")
    CairoMakie.scatter!(ax, num_components, explained_variance, color = :blue)
    
    # Add reference line at 90% explained variance (optional)
    hlines!(ax, [0.1], linestyle = :dash, color = :red, label = "90% Variance")
    
    axislegend(ax, position = :rt)
    save("elbow_analysis.png", fig)
    fig
end

# Perform PCA and plot elbow analysis
println("Performing PCA and plotting elbow analysis...")
elbow_analysis(zscored_log_data_df)
println("Saved elbow analysis figure")
println("Input a number of principal components to use for clustering...")
num_pcs = readline()  # Wait for user input to continue
println("User entered ", num_pcs, " principal components.")
num_pcs = parse(Int64, num_pcs)  # Convert the input to an integer
data = Matrix(zscored_log_data_df[:,2:end])'  # Convert DataFrame to a matrix
pca_model = fit(PCA, data; maxoutdim=num_pcs)  # Fit PCA model
pca_data = reconstruct(fit(pca_model,data),data)  # Transform data
pca_data_df = DataFrame(pca_data', :auto)  # Convert back to DataFrame
# # Perform Hierarchical Clustering
# println("Performing Hierarchical Clustering...")
# hc_model = hclust(pca_data, linkage=:ward)
# plot(hc_model, xticks=:none, yticks=:auto, legend=false, title="Hierarchical Clustering Dendrogram")

# # Perform K-means Cluster Analysis
# num_clusters = parse(Int, readline(stdin, "Enter the number of clusters to use: "))
# start_time = time()
# kmeans_model = kmeans(pca_data, num_clusters, maxiter=100, init=:kmeanspp)
# elapsed_time = time() - start_time
# println(@sprintf("Elapsed time: %.2f seconds", elapsed_time))
# println(@sprintf("Minimum distance: %.2f", minimum(kmeans_model.assignments)))
