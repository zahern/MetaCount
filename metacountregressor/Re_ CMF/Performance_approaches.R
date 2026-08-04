
rm(list = ls())
require(dplyr)
require(ggplot2)
require(extrafont)

loadfonts(device = 'win')

# Comparison of models
washington.data <- read.csv2(file = 'C:\\Users\\gonza\\Desktop\\Pasantía QUT\\Lit review about crash modification factors and functions\\Update 20260403\\GA_CMF_AADT\\Ex-16-3.csv',
                             sep = ",")

washington.data <- washington.data %>% mutate(WIDTH_PER_LANE = WIDTH / (INCLANES + DECLANES),
                                              SLOPE_FLAT = case_when(
                                                SLOPE == 0 ~ 1,
                                                TRUE ~ 0
                                              ),
                                              AADTmaj = log(AADT),
                                              const = 1,
                                              SHOULDER_WIDTH = MIMEDSH / WIDTH,
                                              TOTAL_LANES = INCLANES + DECLANES)

# Parameters hierarchical model
# Np = A * exp(B)

# Parameters
parameters <- read.csv2(file = 'C:\\Users\\gonza\\Desktop\\Pasantía QUT\\Lit review about crash modification factors and functions\\Document 20260528\\Parameters_hierarchical_models.csv',
                        sep = ";") %>% mutate(parameter = as.numeric(parameter))

hierarchical_parameters <- parameters %>% filter(model == 'hierarchical') %>% filter(analysis == 'washington')

A <- as.matrix(hierarchical_parameters$parameter[1:7])
Xa <- as.matrix(washington.data %>% 
  select(const, WIDTH_PER_LANE, CURVES, TANGENT, INTECHAG, MXGRDIFF, SHOULDER_WIDTH) %>%
  mutate(TANGENT  = as.numeric(TANGENT),
         MXGRDIFF = as.numeric(MXGRDIFF)))

# A
A_base <- exp(Xa %*% A)

# B
B <- as.matrix(hierarchical_parameters$parameter[8:9])
B_base <- exp(washington.data$SLOPE_FLAT * B[2])

# Fitted
# Paper form (Eq 37-39): Np = A(z1) * x^{B(z2)} with x = AADT (exposure), base is AADT not log(AADT)
Np <- A_base * washington.data$AADT**(B[1] * B_base)
hist(Np, "fd")

washington.data$hierarchical.vals <- Np

# Literature
literature_parameters <- parameters %>% filter(model == 'literature') %>% filter(analysis == 'washington')

A <- as.matrix(literature_parameters$parameter)
Xa <- as.matrix(washington.data %>%
                  select(const, LOWPRE, GRADEBR, FRICTION, EXPOSE, INTPM, CURVES, HISNOW) %>% 
                  mutate(FRICTION = as.numeric(FRICTION),
                         EXPOSE = as.numeric(EXPOSE),
                         INTPM = as.numeric(INTPM)))

Np <- Xa %*% A
washington.data$N_pred_wash <- exp(Np)

# pred vals hierarchical
data.frame(AADTmaj = c(washington.data$AADTmaj,
                       washington.data$AADTmaj),
           preds   = c(washington.data$hierarchical.vals,
                       washington.data$N_pred_wash),
           Model   = c(rep('Hierarchical', nrow(washington.data)),
                       rep('Literature', nrow(washington.data)))) %>%
  ggplot(aes(x = AADTmaj, y = preds, col = Model)) +
  geom_point(size = 1.5) +
  geom_smooth(method = lm) +
  ylim(c(0, 200)) +
  theme_bw(14) +
  ylab('Fitted values') +
  xlab('log(AADT)') +
  scale_color_manual(values = c("Hierarchical" = "brown3",
                                "Literature" = "deepskyblue3")) +
  theme(axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 15),
        text = element_text(family = "Times New Roman"))

# Queensland Heavy Vehicles
qld.hv <- openxlsx::read.xlsx(xlsxFile = 'C:\\Users\\gonza\\Desktop\\Pasantía QUT\\Lit review about crash modification factors and functions\\Update 20260403\\GA_CMF_AADT\\Stage5A_1848_All_Initial_Columns.xlsx',
                              sheet = 'Stage5A_1848')

qld.hv <- qld.hv %>%
  mutate(AADTmaj = log(AADT),
         Total_width = Lwidth * Nlanes,
         RS_HS = RS * HSP,
         const = 1)

hierarchical_parameters <- parameters %>% filter(analysis == 'QLD HV') %>% filter(model == 'hierarchical')

A  <- as.matrix(hierarchical_parameters$parameter[1:4])
Xa <- as.matrix(qld.hv %>% select(const, Total_width, RS_HS, LNMCV))

A_base <- exp(Xa %*% A)

B  <- as.matrix(hierarchical_parameters$parameter[5:6])
Xb <- as.matrix(qld.hv %>% select(const, RS))
B_base <- exp(B[2] * qld.hv$RS)

Np <- A_base * qld.hv$AADT ** (B[1] * B_base)  
qld.hv$fitted_hierarchical <- Np
  
# Benchmark
qld.hv <- qld.hv %>%
  mutate(AADTmaj     = log(AADT),
         Total_width = Lwidth * Nlanes,
         RS_HS       = RS * HSP,
         RS_MSP      = RS * MSP,
         RD_HS       = RD * HSP,
         RD_MSP      = RD * MSP,
         US_HS       = US * HSP,
         US_MSP      = US * MSP,
         UD_HS       = UD * HSP,
         UD_MSP      = UD * MSP,
         ADTLANE     = AADT / Nlanes)

# Variables
base_vars   <- c('SP','Curve','Nlanes','Total_width','Median','LNMCV',
                 'RS_HS','RS_MSP','US_MSP',
                 'RD_HS','RD_MSP','US','RS','LNMCV')

formula_nb <- as.formula(
  paste('Headon ~ offset(AADTmaj) +', paste(base_vars, collapse = " + "))
)

# All variables
model_nb_full <- MASS::glm.nb(formula = formula_nb,
                              data = qld.hv)
summary(model_nb_full)

# Selection with BIC
step_backward_bic <- MASS::stepAIC(model_nb_full, 
                                   direction = 'backward', 
                                   trace = 1,
                                   k = log(nrow(qld.hv)))

# BIC specification
model_nb_stepwise_bic <- MASS::glm.nb(formula(step_backward_bic),
                                      data = qld.hv)
summary(model_nb_stepwise_bic)

BIC(model_nb_stepwise_bic)
logLik(model_nb_stepwise_bic)

qld.hv$fitted_fe <- fitted(model_nb_stepwise_bic)

data.frame(AADTmaj = c(qld.hv$AADTmaj,
                       qld.hv$AADTmaj),
           preds   = c(qld.hv$fitted_hierarchical,
                       qld.hv$fitted_fe),
           Model   = c(rep('Hierarchical', nrow(qld.hv)),
                       rep('Analyst', nrow(qld.hv)))) %>%
  ggplot(aes(x = AADTmaj, y = preds, col = Model)) +
  geom_point(size = 1.0) +
  geom_smooth(method = lm) +
  theme_bw(14) +
  ylab('Fitted values') +
  xlab('log(AADT)') +
  scale_color_manual(values = c("Hierarchical" = "brown3",
                                "Analyst" = "deepskyblue3")) +
  theme(axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 15),
        text = element_text(family = "Times New Roman"))

# Maine dataset
setwd('C:\\Users\\gonza\\Desktop\\Pasantía QUT\\Lit review about crash modification factors and functions\\Update 20260223\\GA_CMF_AADT')
list.files()

maine.data <- openxlsx::read.xlsx('rural_int.xlsx', 
                                  sheet = 'rural_int') %>% mutate(const = 1)

maine.data <- maine.data %>% mutate(const = 1, AADTmaj = log(monthly_AADT))

hierarchical_parameters <- parameters %>% filter(model == 'hierarchical') %>% filter(analysis == 'Maine')

A  <- as.matrix(hierarchical_parameters$parameter[1:5])
Xa <- as.matrix(maine.data %>% select(
  const, speed, right_shoulder_width, DP01, DX32
))

A_base <- exp(Xa %*% A)

B  <- as.matrix(hierarchical_parameters$parameter[6:7])
B_base <- exp(maine.data$dummy_winter * B[2])

Np <- A_base * maine.data$monthly_AADT**(B[1] * B_base)
maine.data$pred_jerarquico <- Np

literature_parameters <- parameters %>% filter(model == 'literature') %>% filter(analysis == 'Maine')

A <- as.matrix(literature_parameters$parameter)
Xa <- as.matrix(maine.data %>% select(
  const, AADTmaj, segment_length, speed, right_shoulder_width, curve, DP01, DX32
))

Np_maine <- Xa %*% A

maine.data$pred_literature <- exp(Np_maine)

data.frame(AADTmaj = c(maine.data$AADTmaj,
                       maine.data$AADTmaj),
           preds   = c(maine.data$pred_jerarquico,
                       maine.data$pred_literature),
           Model   = c(rep('Hierarchical', nrow(maine.data)),
                       rep('Literature', nrow(maine.data)))) %>%
  slice_sample(n = 20000) %>%
  ggplot(aes(x = AADTmaj, 
             y = preds,
             col = Model)) +
  geom_point(size = 0.5) +
  geom_smooth(method = lm) +
  theme_bw(14) +
  ylab('Fitted values') +
  xlab('log(AADT)') +
  ylim(c(0, 15)) +
  scale_color_manual(values = c("Hierarchical" = "brown3",
                                "Literature" = "deepskyblue3")) +
  theme(axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 15),
        text = element_text(family = "Times New Roman"))
  



