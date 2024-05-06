# Load necessary libraries
library(shiny)
library(ggplot2)
library(dplyr)


# Define server logic
server <- function(input, output, session) {
    umap_df <- read.csv("CLS_features_annotated_umap.csv")
    # Create well select input
    # Create well checkbox group input
    output$doseSelect <- renderUI({
    checkboxGroupInput("Doses", "Select Doses:", choices = unique(umap_df$Metadata_dose), selected = unique(umap_df$Metadata_dose))
    })

    # Create time checkbox group input
    output$timeSelect <- renderUI({
    checkboxGroupInput("times", "Select Time Points:", choices = unique(umap_df$Metadata_Time), selected = unique(umap_df$Metadata_Time))
    })

    output$umapPlot <- renderPlot({
    # Filter data based on selected wells and time points
    data <- dplyr::filter(umap_df, Metadata_dose %in% input$Doses & Metadata_Time %in% input$times)

    # Create plot
    ggplot(data, aes(x = UMAP1, y = UMAP2, color = as.factor(Metadata_Time))) +
        geom_point() +
        theme_minimal() +
        scale_color_discrete(name = "Time Point")
    })
    }

