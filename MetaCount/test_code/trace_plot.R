extrafont::loadfonts(device="win")
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(ggtext)
library(ggthemes)
library(ggpomological)
library(bbplot)
library(extrafont)
library(extrafontdb)
library(cowplot)
library(ggtext)
library(glue)


library(envalysis)
library(firatheme)
warnings()
#find the folder of the best solution
df <- read.csv('set1logtrc.csv')
df_folder <- df[order(df$result),]
df_folder <- df_folder[['folder']][1]
df2 <- read.csv('set2logtrc.csv')
df2_folder <- df2[order(df2$result),]
df3 <- read.csv('set3logtrc.csv')
df3_folder <- df3_folder[order(df$result),]



df <- read.csv(paste('Z:/phd/paper_1/',toString(df_folder),'/result_frame1.csv',sep=""))
df_long <- gather(df, Solution, fitness, incumbent:best, factor_key = TRUE)
levels(df_long$Solution)[levels(df_long$Solution)=="incumbent"] <- "Incumbent"
levels(df_long$Solution)[levels(df_long$Solution)=="best"] <- "Current Best"

# Plot
ggplot(df_long, aes(x=iteration, y=fitness, group=Solution, 
                    color=Solution))+geom_line()+
  labs(x = 'Iteration', y = 'Fitness Value (BIC)')+ theme_cowplot() 
ggsave('trace1.eps')

df <- read.csv('Z:/phd/trc/227/result_frame1.csv')
df_long <- gather(df, Solution, fitness, incumbent:best, factor_key = TRUE)
levels(df_long$Solution)[levels(df_long$Solution)=="incumbent"] <- "Incumbent"
levels(df_long$Solution)[levels(df_long$Solution)=="best"] <- "Current Best"

# Plot
ggplot(df_long, aes(x=iteration, y=fitness, group=Solution, 
                    color=Solution))+geom_line()+
  labs(x = 'Iteration', y = 'Fitness Value (BIC)')+ theme_cowplot() 
ggsave('trace2.eps')

df <- read.csv('Z:/phd/trc/237/result_frame1.csv')
df_long <- gather(df, Solution, fitness, incumbent:best, factor_key = TRUE)
levels(df_long$Solution)[levels(df_long$Solution)=="incumbent"] <- "Incumbent"
levels(df_long$Solution)[levels(df_long$Solution)=="best"] <- "Current Best"

# Plot
ggplot(df_long, aes(x=iteration, y=fitness, group=Solution, 
                    color=Solution))+geom_line()+
  labs(x = 'Iteration', y = 'Fitness Value (BIC)')+ theme_cowplot() 

ggsave('trace3.eps')

