library(tidyverse)

spam <- read.csv("../Data/spambase_csv.csv")
head(spam)

### 1. EDA:

# 1.1 Estado inicial
library(skimr)

glimpse(spam)
skim(spam)

# Cambio de variable target
spam$class <- ifelse(spam$class == 1, "Si", "No")
spam$class <- as.factor(spam$class)

spam <- spam %>%
  mutate(spam = spam$class)

spam$class <- NULL

head(spam)  

# 1.2 Comportamineto de los datos
any(!complete.cases(spam))

# Resumen
summary(spam)
str(spam)

# Existencia de NA
map_dbl(spam, .f = function(x){sum(is.na(x))})

# comportamiento de NA´s
library(mice)
library(VIM)

# Cuenta el número de NA con mice
md.pattern(spam)

# Da la proporción de los NA respecto al total con VIM
aggr(spam,
     numbers = T,
     sortVars = T)

# Distribución de la variable target
prop.table(table(spam$spam)) %>%
  round(2)

# 1.3 Importancia de las variables

# Correlaciones
library(corrplot)
library(PerformanceAnalytics)

spam_cor <- cor(spam[, 1:57], method = "pearson") %>%
  round(2)

corrplot(spam_cor, method = "circle",
         type = "upper",
         shade.col = NA, tl.col = "black",
         tl.srt = 45)

chart.Correlation(spam[, 1:57], 
                  histogram = FALSE, 
                  pch = 19)


# Variables más influyentes
library(randomForest)
library(ggpubr)

modelo_rf_imp <- randomForest(spam ~ ., data = spam,
                              mtry = 5, importance = TRUE,
                              ntree = )

importancia <- as.data.frame(modelo_rf_imp$importance)
head(importancia)
importancia <- rownames_to_column(importancia, var = "variable")


graf1 <- ggplot(data = importancia, aes(x = reorder(variable, MeanDecreaseAccuracy),
                                        y = MeanDecreaseAccuracy,
                                        fill = MeanDecreaseAccuracy)) +
  geom_col() +
  labs(x = "variable", title = "Reducción del Accuracy") + 
  coord_flip() +
  theme_update() + theme(legend.position = "bottom")

graf2 <-  ggplot(data = importancia, aes(x = reorder(variable, MeanDecreaseGini),
                                         y = MeanDecreaseGini,
                                         fill = MeanDecreaseGini)) +
  geom_col() +
  labs(x = "variable", title = "Reducción de la pureza - Gini") + 
  coord_flip() +
  theme_update() + theme(legend.position = "bottom")

ggarrange(graf1, graf2)


# Conclusiones:

# 1. Las variables estan casi nada correlacionadas entre ellas.
# 2. No se tienen valores NA´s
# 3. Las variables que más inflyen para decidir si el correo es spam o no son:
# capital_run_length_longest, capital_run_length_longest, char_freq_.21,
# word_freq_remove, capital_run_length_total, capital_run_length_average,
#"char_freq_.24, word_freq_free, word_freq_your, word_freq_george.


### 2. PRE-PROCESADOS DE DATOS:

# Se reescalan y se normalizan las variables
library(scales)
library(caret)

spam_def <- lapply(spam[, 1:57], rescale)
spam_def <- as.data.frame(spam_def)

spam_def <- lapply(spam_def, scale)
spam_def <- as.data.frame(spam_def)  

spam_def <- spam_def %>%
  mutate(spam = spam$spam)

glimpse(spam_def)

# Se realiza la partición de dataset en train y test
t_id <- createDataPartition(spam_def$spam, p = 0.7, list = F)
train_spam <- spam_def[t_id, ]
test_spam <- spam_def[-t_id, ]


### 3. MODELADO:
# Se usaran 2 algoritmos de predicción y se elegira el mejor resultado
library(doParallel)

# 3.1 Random Forest

# Control
particiones <- 10
repeticioens <- 5
n_hiper <- expand.grid(mtry = c(2 , 10, 20, 50),
                       min.node.size = c(2, 10, 30, 50),
                       splitrule = "gini")

control_train <- trainControl(method = "repeatedcv",
                              number = particiones,
                              repeats = repeticioens,
                              returnResamp = "final")

# Inicio del paralelizado para disminuir los tiempos
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

# Modelo
rf_modelo <- train(spam ~ ., data = train_spam,
                   method= "ranger", tuneGrid = n_hiper,
                   metric = "Accuracy", trControl = control_train,
                   num.trees = 500)

# Fin del paralelizado
stopCluster(cl)

rf_modelo$resample %>% head()

# Plot del comportamiento de los hiperparámetros
ggplot(rf_modelo, highlight = T) +
  labs(title = "Evaluación de Accuracy con Random Forest") +
  guides(color = guide_legend(title = "mtry"),
         shape = guide_legend(title = "mtry")) +
  theme_update()

# Predicción 
predic_rf <- predict(rf_modelo, test_spam)
predic_rf %>%
  head()

# Error del test
  mc_rf <- confusionMatrix(predic_rf, test_spam$spam)
  mc_rf

# Accuracy final rf
precis_rf <- mc_rf$overall[["Accuracy"]]

# Conclusión
# El mejor resultado del algoritmo se obtiene con un tamaño minimo de nodo de 10 y con un mtry de 2

# 3.2 Redes Neuronales

# Control
particiones<- 10
repeticioens <- 5
n_hiper_rn <- expand.grid(size = c(5, 8, 10, 15),
                          decay = c(0.01, 0.08, 0.1, 0.5))

control_train_rn <- trainControl(method = "repeatedcv",
                              number = particiones,
                              repeats = repeticioens,
                              returnResamp = "final")

# Paralelizado
cl <- makePSOCKcluster(7)
registerDoParallel(cl)

# Modelo
nnet_model <- train(spam ~ ., data = train_spam,
                    method = "nnet", tuneGrid = n_hiper_rn,
                    metric = "Accuracy", trControl = control_train_rn,
                    rang = c(-0.5, 0.5), MaxNWts = 2000,
                    trace = FALSE)

# Fin del paralelizado
stopCluster(cl)

nnet_model
nnet_model$resample %>% head()

# Plot del comportamiento de los hiperparámetros
ggplot(nnet_model, highlight = T) +
  labs(title = "Evaluación de Accuracy con Redes Neuronales") +
  theme_update()

# Predicción 
predic_nnet <- predict(nnet_model, test_spam)
predic_nnet %>%
  head()

# Error del test
mc_nnet <- confusionMatrix(predic_nnet, test_spam$spam)
mc_nnet

# Accuracy final rf
precis_nnet <- mc_nnet$overall[["Accuracy"]]

# Conclusión
# El mejor resultado se obtiene con 5 capas escondidas y un decay de 0.5

### 4. RESULTADOS:
comparacion <- tibble(Modelo = c("Random Forest", "Redes Neuronales"),
                      Accuracy = c(precis_rf, precis_nnet))

comparacion %>%
  ggplot(aes(x = reorder(comparacion$Modelo, comparacion$Accuracy),
             y = comparacion$Accuracy)) +
  geom_segment(aes(x = reorder(comparacion$Modelo, comparacion$Accuracy),
                   y = 0, 
                   xend = comparacion$Modelo, 
                   yend = comparacion$Accuracy),
               color = "darkorange") +
  geom_point(size = 10, 
             color = "cadetblue") +
  geom_text(label = round(comparacion$Accuracy, 3),
            color = "white", size = 2.4) +
  scale_y_continuous(limits = c(0,1)) +
  labs(title = "Comparación de Accuracy",
       x = "Modelo", 
       y = "Accuracy Total") +
  coord_flip() +
  theme_update()

# CONCLUSIÓN
# El algoritmo con mejor resultado final fue Random Forest y es el elegido.
# Con esto se tiene que el 94.2% de las veces que llegue un email y sea spam, sera detectado.
