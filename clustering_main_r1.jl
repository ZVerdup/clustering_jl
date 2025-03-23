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
using Distances
println("Loaded Distances...")
using StatsPlots
println("Loaded StatsPlots...")

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
    return num_components
end

function tracks_plot(data::DataFrame,savefile::String)
    println("Plotting tracks and saving png file...")
    header = names(data)
    num_plots = length(header) - 1  # Exclude the first column
    fig = Figure()
    for j in 1:num_plots
        # Hide the y-axis for all but the first subplot
        if j==1  
            ylabelflag = true
        else
            ylabelflag = false
        end 
        ax = Axis(fig[1, j],
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
        lines!(ax, data[:, j+1], data[:, 1])  # Plot the data
    end
    Label(fig[0, :], savefile, fontsize = 20)
    resize_to_layout!(fig)
    save(savefile, fig)  # Save the figure to a file
end

function single_bar_plot(log_data::DataFrame,grp::Vector,num::Int64,savefile::String)
    fhc = Figure(size = (75, 400), fontsize = 8)
    fig_title = string(num," Clusters")
    ax_Dbar = Axis(
        fhc[1,1],
        title = fig_title, 
        yticklabelsvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        ygridvisible = false,
        xticksvisible = false,
        xgridvisible = false,
    )

    barplot!(
    ax_Dbar,
    reverse(log_data[!,"DEPTH"]), grp,
    gap=0,
    color=grp,
    direction=:x,
    )

    CairoMakie.xlims!(ax_Dbar, 0, .5)
    CairoMakie.ylims!(ax_Dbar, log_data[!,"DEPTH"][1], log_data[!,"DEPTH"][end])

    save(savefile, fhc)
end

# Define the parent directory and filename
initial_dir = raw"C:\Users\evan.kias\Documents\Scripts\Julia\Clustering\clustering_jl"
# initial_dir = raw"C:\Users\evank\Documents\Clustering\clustering_jl\clustering_jl"
filename = "logs.csv"
full_path = joinpath(initial_dir, filename)
println(full_path)

#wait_for_key("Ready to read data. Press ENTER to continue...")
println("Reading data...")
# Read the data into a DataFrame
log_data = CSV.read(full_path, DataFrame; skipto=3)

# Visualize the data
println("Visualizing log data...")
tracks_plot(log_data,"log_data.png")
println("Saved log data fig")

# Prompt for resistivity variable
header = names(log_data)
println(header)
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
tracks_plot(log_data_logrt,"logrt_data.png")

# Standardize the data using z-score transformation
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
tracks_plot(zscored_log_data_df,"zscored_log_data.png")

# Perform PCA and plot elbow analysis
println("Performing PCA and plotting elbow analysis...")
(num_components) = elbow_analysis(zscored_log_data_df)
println("Saved elbow analysis figure")
println("Number of PCA components = ", string(length(num_components)))
println("Input a number of principal components to use for clustering...")
num_pcs = readline()  # Wait for user input to continue
println("User entered ", num_pcs, " principal components.")
num_pcs = parse(Int64, num_pcs)  # Convert the input to an integer
data = Matrix(zscored_log_data_df[:,2:end])'  # Convert DataFrame to a matrix
pca_model = fit(PCA, data; maxoutdim=num_pcs)  # Fit PCA model
println("PCA model fitted with ", num_pcs, " components.")
pca_pred = predict(pca_model,data)
println("PCA data predicted.")
pca_data = reconstruct(pca_model,pca_pred)  # Transform data
println("Z-score scaled PCA data reconstructed.")
pca_data_arr = hcat(log_data[!,"DEPTH"], pca_data')
pca_data_df = DataFrame(pca_data_arr, header)  # Convert back to DataFrame

# Visualize PCA Reconstructed data
println("Visualizing reconstructed PCA data...")
tracks_plot(pca_data_df,"recon_pca_data.png")

# Perform Hierarchical Clustering
println("Performing Hierarchical Clustering...")
start_time = time()
# Calculate the Euclidean distance matrix
distance_matrix = pairwise(Euclidean(), pca_data, dims=2)
hc_model = hclust(distance_matrix, linkage=:ward)
elapsed_time = time() - start_time
println(@sprintf("Elapsed time: %.2f seconds", elapsed_time))

# Plot the dendrogram tree
StatsPlots.plot(hc_model, xticks=:none, yticks=:auto, legend=false, title="Hierarchical Clustering Dendrogram")
savefig("hclust_result.png")      # saves the CURRENT_PLOT as a .png
println("Saved dendrogram...")

#Analyze and Plot the Data
# Cut the dendrogram into clusters and plot each (e.g., 5 clusters)
println("Plotting hierarchical clusters...")
for num in 2:length(num_components)
    clusterNums = cutree(hc_model, k=num)
    single_bar_plot(log_data,clusterNums,num,string("hc_",num,"_clusters.png"))
end

# Perform K-means Cluster Analysis
println("Performing K-means Cluster Analysis...")
start_time = time()
for num in 2:length(num_components)
    kmeans_model = kmeans(pca_data, num, maxiter=100, display=:iter)
    clusterNums = assignments(kmeans_model)
    single_bar_plot(log_data,clusterNums,num,string("kmeans_",num,"_clusters.png"))
end
elapsed_time = time() - start_time
println(@sprintf("Elapsed time: %.2f seconds", elapsed_time))
println(@sprintf("Minimum distance: %.2f", minimum(kmeans_model.assignments)))