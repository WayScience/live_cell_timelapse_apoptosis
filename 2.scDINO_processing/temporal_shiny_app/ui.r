# Load necessary libraries
library(shiny)


# Define UI
ui <- fluidPage(
  titlePanel("UMAP Plot"),

  sidebarLayout(
    sidebarPanel(
      uiOutput("doseSelect"),
      uiOutput("timeSelect")
    ),

    mainPanel(
      plotOutput("umapPlot")
    )
  )
)
