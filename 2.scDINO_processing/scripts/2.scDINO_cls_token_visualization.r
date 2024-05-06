suppressPackageStartupMessages(suppressWarnings(library(ggplot2)))
suppressPackageStartupMessages(suppressWarnings(library(dplyr)))
suppressPackageStartupMessages(suppressWarnings(library(tidyr)))
suppressPackageStartupMessages(suppressWarnings(library(shiny)))

# set UMAP df path
umap_df_path <- file.path("..","..","1.scDINO_run/outputdir/test_run/CLS_features/CLS_features_annotated_umap.csv")
# load UMAP df
umap_df <- read.csv(umap_df_path)
head(umap_df,2)

# get all wells
unique(umap_df$Metadata_Well)

# select just one well
# umap_df <- umap_df %>% filter(Metadata_Well == "E-11")
# remove the T from the time column
umap_df$Metadata_Time <- gsub("T","",umap_df$Metadata_Time)
# convert to numeric
umap_df$Metadata_Time <- as.numeric(umap_df$Metadata_Time)


# plot UMAP
width <- 15
height <- 10
options(repr.plot.width = width, repr.plot.height = height)
umap_plot <- (
    ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Metadata_Time))
    + geom_point(size = 1)
    + theme_minimal()
    + theme(legend.position = "right")
)
umap_plot
