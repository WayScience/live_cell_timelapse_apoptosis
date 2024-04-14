# Load necessary libraries
library(shiny)


# Define UI
ui <- fluidPage(
  titlePanel("UMAP Plot"),

  sidebarLayout(
    sidebarPanel(
      uiOutput("wellSelect"),
      uiOutput("timeSelect")
    ),

    mainPanel(
      plotOutput("umapPlot")
    )
  )
)
