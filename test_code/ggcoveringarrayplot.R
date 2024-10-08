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
library(ggbeeswarm)
library(ggforce)
head(df)
df <- read.csv('set1logtrc.csv')
df_folder <- df[order(df$result),]
df2 <- read.csv('set2logtrc.csv')
df2_folder <- df2[order(df2$result),]
df3 <- read.csv('set3logtrc.csv')
df3_folder <- df3_folder[order(df$result),]
#extract out the means of the seeds
df_seed = aggregate(df$result, by = list(df$temp_scale, df$steps, df$algorithm, df$population, df$crossover, df$hmcr, df$par), FUN = mean)
df1 = df %>% group_by(temp_scale, steps, algorithm, population, crossover, hmcr, par) %>% summarise(result = mean(result))

df_2 = df2 %>% group_by(temp_scale, steps, algorithm, population, crossover, hmcr, par) %>% summarise(result = mean(result))
df_3 = df3 %>% group_by(temp_scale, steps, algorithm, population, crossover, hmcr, par) %>% summarise(result = mean(result))  

ggplot(df1, aes(factor(algorithm), result))+ geom_density(aes(algorithm), trim = F)+scale_color_manual()

ggplot(df1, aes(x=1, y=result, group=algorithm, color=algorithm))+ labs(
   x="Distinct Set of Hyperparameters",
   y="Fitness of the Experiment",
   color="Algorithm",
)+ geom_violin()+geom_sina() +theme_cowplot()+theme(axis.ticks.x = element_blank(),
                                        axis.text.x = element_blank(), axis.title.x.top  = element_blank())+scale_color_fivethirtyeight()  
ggsave('Indiana_redone.eps')

ggplot(df_2, aes(x=1, y=result, group=algorithm, color=algorithm))+ labs(
   x="Distinct Set of Hyperparameters",
   y="Fitness of the Experiment",
   color="Algorithm",
)+ geom_violin()+geom_sina() +theme_cowplot()+theme(axis.ticks.x = element_blank(),
                                                    axis.text.x = element_blank(), axis.title.x.top  = element_blank())+scale_color_fivethirtyeight()  
ggsave('QLD_redone.eps')
ggplot(df_3, aes(x=1, y=result, group=algorithm, color=algorithm))+ labs(
   x="Distinct Set of Hyperparameters",
   y="Fitness of the Experiment",
   color="Algorithm",
)+ geom_violin()+geom_sina() +theme_cowplot()+theme(axis.ticks.x = element_blank(),
                                                    axis.text.x = element_blank(), axis.title.x.top  = element_blank())+scale_color_fivethirtyeight()  
ggsave('Syntehtic_redone.eps')

