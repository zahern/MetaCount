library(rstudioapi)
# Set the working directory to the script's location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

setwd('current files')
library(lme4)

nlogit_data_simple <- read.csv("nlogit_data_simple.csv", 
  stringsAsFactors = F)

colnames(nlogit_data_simple)

mixed_data <- nlogit_data_simple

mixed_data$AmountDiff <- mixed_data$Amount - mixed_data$AmountA


summary(mixed_data$AmountA)
sd(mixed_data$AmountA)
summary(mixed_data$Amount)
sd(mixed_data$Amount)
summary(mixed_data$AmountDiff)
sd(mixed_data$AmountDiff)



summary(mixed_data$Duration)



mixed_data[mixed_data$Amount==0,]
mixed_data[mixed_data$ChoiceNm==4,]
#OthrNtPd ->Number of all unpaid parking in the zone except the current observation (if unpaid) for the hour of parking start time

mixed_data$OthrNtPd = ifelse(mixed_data$ChoiceNm==4, mixed_data$MetrNtPd-1, mixed_data$MetrNtPd)
mixed_data$NtPd = ifelse(mixed_data$ChoiceNm==4, 1, 0)

mixed_data$Paid = ifelse(mixed_data$ChoiceNm==2, T, F)
# Assuming mixed_data is your data frame
dependent_var <- "AmountDiff"


unique(mixed_data$tmMdnght)
#0, we remove


#We reduced the variables to the ones described in the table 2 .
independent_vars <- c("StayDrtn","Weekend", "StrtTmHr", "TmTClrwy", "VehiclCr", "VehiclUt", "VehiclVn",
                      "VehclHvy", "FndMtrNP", "OthrNtPd", "TrnvrAdj", "LctnElzS", "LctnAlcS")


library(MASS)

full_model <- lm(AmountDiff ~ StayDrtn+Weekend+StrtTmHr+TmTClrwy+
                   VehiclCr+VehiclUt+VehiclVn+VehclHvy+FndMtrNP+
                   OthrNtPd+TrnvrAdj+LctnElzS+LctnAlcS,
                 data = mixed_data)
#we only test PrsnlAPD	The average parking payment of the same vehicle in a month before the observation (May 2019 - min)
#reason: not determinant of unpaid which will be used later, 


# Perform stepwise regression using BIC
stepwise_model <- stepAIC(full_model, direction = "both", k = log(nrow(mixed_data)))

# Summary of the final model
summary(stepwise_model)

require(broom)
broom::glance(stepwise_model)


model <- lm(AmountDiff ~ StayDrtn  + Weekend + StrtTmHr + 
              TmTClrwy +  TrnvrAdj + LctnElzS + LctnAlcS, data = mixed_data)

library(car)

vif(model)
#StayDrtn  Weekend StrtTmHr TmTClrwy TrnvrAdj LctnElzS LctnAlcS 
#1.090938 1.020858 1.281073 1.794273 1.211123 1.942245 1.272790
#less than 5

# Select the variables for the correlation matrix
selected_vars <- mixed_data[, c("StayDrtn", "Weekend",
                                "StrtTmHr", "TmTClrwy", 
                                "TrnvrAdj", "LctnElzS", "LctnAlcS")]

# Calculate the correlation matrix
cor_matrix <- cor(selected_vars)
cor_matrix
#StayDrtn      Weekend    StrtTmHr     TmTClrwy    TrnvrAdj    LctnElzS    LctnAlcS
#StayDrtn  1.00000000  0.085641746 -0.09642962 -0.171054456 -0.05617111 -0.26162901  0.05543271
#Weekend   0.08564175  1.000000000  0.01657438 -0.002505481  0.09235742 -0.03285935  0.01179171
#StrtTmHr -0.09642962  0.016574381  1.00000000  0.410346972  0.12189629  0.15395331 -0.12448964
#TmTClrwy -0.17105446 -0.002505481  0.41034697  1.000000000 -0.09621887  0.50765439 -0.17631027
#TrnvrAdj -0.05617111  0.092357416  0.12189629 -0.096218874  1.00000000  0.24261984 -0.14383782
#LctnElzS -0.26162901 -0.032859352  0.15395331  0.507654393  0.24261984  1.00000000 -0.44641119
#LctnAlcS  0.05543271  0.011791714 -0.12448964 -0.176310274 -0.14383782 -0.44641119  1.00000000

write.csv(mixed_data, file="mixed_data_for_nlogit_v2.csv",
          row.names = F)


#we save the data as csv to be read by Nlogit 

#Call:
#  lm(formula = AmountDiff ~ StayDrtn + Weekend + StrtTmHr + TmTClrwy + 
#       TrnvrAdj + LctnElzS + LctnAlcS, data = mixed_data)

#Residuals:
#  Min       1Q   Median       3Q      Max 
#-17.8280  -1.4890   0.1024   1.3912  16.8029 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)  3.0097592  0.2377098  12.661  < 2e-16 ***
#  StayDrtn    -0.0271105  0.0010970 -24.714  < 2e-16 ***
#  Weekend      0.5002948  0.1241050   4.031 5.71e-05 ***
#  StrtTmHr    -0.1491183  0.0162053  -9.202  < 2e-16 ***
#  TmTClrwy    -0.0004423  0.0001324  -3.340 0.000851 ***
#  TrnvrAdj     0.6466354  0.0843339   7.668 2.49e-14 ***
#  LctnElzS    -1.4443813  0.1584059  -9.118  < 2e-16 ***
#  LctnAlcS    -0.9155088  0.1421674  -6.440 1.43e-10 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 2.747 on 2500 degrees of freedom
#Multiple R-squared:  0.2525,	Adjusted R-squared:  0.2504 
#F-statistic: 120.6 on 7 and 2500 DF,  p-value: < 2.2e-16

############################################################################################################################



##############################################logit with NtPd as dependent variable###############################
full_model <- glm(NtPd ~ StayDrtn+Weekend+StrtTmHr+AmountA+
TmTClrwy+
                   VehiclCr+VehiclUt+VehiclVn+VehclHvy+FndMtrNP+
                   OthrNtPd+TrnvrAdj+LctnElzS+LctnAlcS,
                 data = mixed_data, family = "binomial")



# Perform stepwise regression using BIC
stepwise_model <- stepAIC(full_model, direction = "both", k = log(nrow(mixed_data)))
summary(stepwise_model)

require(broom)
broom::glance(stepwise_model)


model <-  glm(NtPd ~ StayDrtn + Weekend + StrtTmHr + AmountA + 
                TrnvrAdj + LctnElzS + LctnAlcS, data = mixed_data, family = "binomial")


#Call:
#  glm(formula = NtPd ~ StayDrtn + Weekend + StrtTmHr + AmountA + 
#        TrnvrAdj + LctnElzS + LctnAlcS, family = "binomial", data = mixed_data)
#
#Coefficients:
#  Estimate Std. Error z value Pr(>|z|)    
#(Intercept) -0.25645    0.21416  -1.198   0.2311    
#StayDrtn     0.02062    0.00333   6.192 5.95e-10 ***
#  Weekend     -0.92969    0.18693  -4.974 6.57e-07 ***
#  StrtTmHr     0.05416    0.01222   4.434 9.26e-06 ***
#  AmountA     -0.54586    0.05465  -9.988  < 2e-16 ***
#  TrnvrAdj    -0.54757    0.07280  -7.522 5.41e-14 ***
#  LctnElzS     0.39453    0.12707   3.105   0.0019 ** 
#  LctnAlcS     1.15087    0.12963   8.878  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
#(Dispersion parameter for binomial family taken to be 1)
#
#Null deviance: 2863.8  on 2507  degrees of freedom
#Residual deviance: 2446.0  on 2500  degrees of freedom
#AIC: 2462
#
#Number of Fisher Scoring iterations: 5
####################################################################################################################



##############################################continuous with only on the ones who paid with mobil app dummy###############################

mixed_data_paid = mixed_data[mixed_data$NtPd==0,]
colnames(mixed_data_paid)
model <- lm(AmountDiff ~ StayDrtn  + Weekend + StrtTmHr + 
              TmTClrwy +  TrnvrAdj + LctnElzS + LctnAlcS +MobilePd, data = mixed_data_paid)

summary(model)
vif(model)
#StayDrtn  Weekend StrtTmHr TmTClrwy TrnvrAdj LctnElzS LctnAlcS MobilePd 
#1.141749 1.016548 1.265310 1.915924 1.199016 2.156406 1.199792 1.013693 

write.csv(mixed_data_paid , file="C:/Users/flori/OneDrive/Desktop/QUT_2024/Parking_Payment/Florian_work/mixed_data_paid_for_nlogit_mobilepaid.csv",
          row.names = F)

mixed_data_paid$MobilePd_StayDrtn = mixed_data_paid$MobilePd * mixed_data_paid$StayDrtn
model <- lm(AmountDiff ~ StayDrtn  + Weekend + StrtTmHr + 
              TmTClrwy +  TrnvrAdj + LctnElzS + LctnAlcS +MobilePd + MobilePd_StayDrtn, data = mixed_data_paid)
summary(model)
#Call:
#  lm(formula = AmountDiff ~ StayDrtn + Weekend + StrtTmHr + TmTClrwy + 
#       TrnvrAdj + LctnElzS + LctnAlcS + MobilePd + MobilePd_StayDrtn, 
#     data = mixed_data_paid)

#Residuals:
#  Min       1Q   Median       3Q      Max 
#-14.1372  -1.1477  -0.1293   0.9996  15.8100 

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)    
#(Intercept)        4.4056343  0.2467216  17.857  < 2e-16 ***
#  StayDrtn          -0.0270957  0.0012471 -21.727  < 2e-16 ***
#  Weekend            0.6500514  0.1168718   5.562 3.05e-08 ***
#  StrtTmHr          -0.1835896  0.0170110 -10.792  < 2e-16 ***
#  TmTClrwy          -0.0005016  0.0001273  -3.939 8.48e-05 ***
#  TrnvrAdj           0.4139168  0.0784020   5.279 1.45e-07 ***
#  LctnElzS          -1.1676234  0.1538064  -7.592 4.98e-14 ***
#  LctnAlcS           0.0905499  0.1355551   0.668 0.504221    
#MobilePd          -0.7697259  0.2151726  -3.577 0.000356 ***
#  MobilePd_StayDrtn  0.0064397  0.0022532   2.858 0.004311 ** 
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 2.178 on 1851 degrees of freedom
#Multiple R-squared:  0.3067,	Adjusted R-squared:  0.3033 
#F-statistic: 90.97 on 9 and 1851 DF,  p-value: < 2.2e-16
####################################################################################################################


#############################################graphs see mobile paid impact on payment distribution###########################
mixed_data_paid = mixed_data[mixed_data$NtPd==0,]
mixed_data_paid_mobile = mixed_data_paid[mixed_data_paid$MobilePd==1,]
mixed_data_paid_notmobile = mixed_data_paid[mixed_data_paid$MobilePd==0,]

hist(mixed_data_paid_mobile$AmountDiff)
hist(mixed_data_paid_notmobile$AmountDiff)

table(mixed_data_paid$LctnElzS,mixed_data_paid$MobilePd)
table(mixed_data_paid$LctnAlcS,mixed_data_paid$MobilePd)
table(mixed_data_paid$LctnTrb,mixed_data_paid$MobilePd)

colnames(mixed_data_paid)

library(ggplot2)
# Calculate means
mean_mobile <- mean(mixed_data_paid_mobile$AmountDiff, na.rm = TRUE)
mean_notmobile <- mean(mixed_data_paid_notmobile$AmountDiff, na.rm = TRUE)

# Calculate maximum density positions for better annotation placement
density_mobile <- density(mixed_data_paid_mobile$AmountDiff)
density_notmobile <- density(mixed_data_paid_notmobile$AmountDiff)
max_y_mobile <- max(density_mobile$y)
max_y_notmobile <- max(density_notmobile$y)

# Plot
ggplot() +
  geom_density(data = mixed_data_paid_mobile, aes(x = AmountDiff, color = "Mobile Payment", fill = "Mobile Payment"), alpha = 0.3) +
  geom_density(data = mixed_data_paid_notmobile, aes(x = AmountDiff, color = "Parking Meter Payment", fill = "Parking Meter Payment"), alpha = 0.3) +
  labs(x = "Difference: Amount Paid – Expected Amount", y = "Density") +
  scale_color_manual(name = "Payment Method", values = c("Mobile Payment" = "blue", "Parking Meter Payment" = "red")) +
  scale_fill_manual(name = "Payment Method", values = c("Mobile Payment" = "blue", "Parking Meter Payment" = "red")) +
  scale_x_continuous(breaks = seq(floor(min(mixed_data_paid_mobile$AmountDiff, mixed_data_paid_notmobile$AmountDiff)), 
                                  ceiling(max(mixed_data_paid_mobile$AmountDiff, mixed_data_paid_notmobile$AmountDiff)), by = 1),
                     limits = c(min(mixed_data_paid_mobile$AmountDiff, mixed_data_paid_notmobile$AmountDiff), 
                                max(mixed_data_paid_mobile$AmountDiff, mixed_data_paid_notmobile$AmountDiff))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 0.8) + # Add vertical line at 0
  # Annotate mean for mobile at the top of the mobile density
  annotate("text", x = mean_mobile, y = max_y_mobile, label = paste(round(mean_mobile, 2),"     "), 
           color = "blue", vjust = -0.5, size = 3.5) +
  # Annotate mean for non-mobile at the top of the non-mobile density
  annotate("text", x = mean_notmobile, y = max_y_notmobile, label = paste("       ",round(mean_notmobile, 2)), 
           color = "red", vjust = -0.5, size = 3.5) +
  theme_minimal(base_size = 14) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95"),
    legend.position = "top",
    legend.title = element_blank()
  )


#####################################################################################################



