library(dplyr)
library(ggplot2)
library(rsconnect)

Sys.setenv(RSCONNECT_NAME='')
Sys.setenv(RSCONNECT_TOKEN='')
Sys.setenv(RSCONNECT_SECRET='')

Sys.getenv("RSCONNECT_NAME")

rsconnect::setAccountInfo(
  name = Sys.getenv("RSCONNECT_NAME"),
  token = Sys.getenv("RSCONNECT_TOKEN"),
  secret = Sys.getenv("RSCONNECT_SECRET")
)

rsconnect::deployApp(appDir = "../temporal_shiny_app", appName = "temporal_shiny_app")

