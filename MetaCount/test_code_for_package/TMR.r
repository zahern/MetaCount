library(readxl)
path = 'Expanded Sections (0.5 to 1.0km)_mod.xlsx'
data <- read_excel(path, sheet = "Data", skip = 3)
data <- data[-((nrow(data)-2):nrow(data)), ]
library(ggplot2)
library(pscl)
library(MASS)
library(boot)
library(dplyr)

#FSI crash costs per 1,000 tonne-km of payload

#1.	Using Hurdle models, develop quantitative relationships between FSI crash costs per 1,000 tonne-km and roadway AADT for roads gazetted for B-doubles for a 5-year analysis period:
 # a.	On single carriageway rural roads
data$`All Crash Types...245` = as.integer(as.numeric(as.character(data$`All Crash Types...245`)))
data$`All Crash Types...245` <- as.integer(data$`All Crash Types...245`/1000)
data[[exog]] <- as.integer(data[[exog]])
count_crash_types <- colnames(data)[245] 
offset_var <- colnames(data)[27]
exog <- colnames(data)[57]
# construct formula as a string
formula_str <- paste(count_crash_types, "~", log1p(exog))

# convert to formula object
formula_obj <- as.formula(formula_str)
m1 <- zeroinfl(formula_str, data = data, dist = "negbin")
m1 <- zeroinfl(data[[count_crash_types]] ~ log1p(data[[exog]]), dist = "negbin")
summary(m1)
