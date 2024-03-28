install.packages("ranger")

# Loading required packges

pacman::p_load(tidymodels, tidyverse)
tidymodels_prefer()

# Importing the dataset

df <- read_csv(file.choose())
df <- df %>%  # Featuring engeneiring - tranforming the target variable from numeric to factor
  mutate(death = factor(death))

# Data visualization

df %>% 
  ggplot() + 
  geom_bar(aes(x = death, fill = sex), position = "dodge") +
  scale_fill_manual(values = c("black", "gray"))+
  theme_bw()

df %>% 
  ggplot() +
  geom_histogram(aes(x = age, fill = sex),binwidth = 5, position = "dodge") +
  scale_fill_manual(values = c("black", "gray")) +
  theme_bw()

df %>% 
  ggplot() +
  geom_boxplot(aes(x = death, y = ejection_fraction, fill = sex)) +
  labs(x = 'Death',
         y = 'Ejection fraction (%)') +
  scale_fill_manual(values = c("black", "gray"))+
  theme_bw()

df %>% 
  ggplot() +
  geom_violin(aes(x = death, y = creatinine_phosphokinase, fill = sex)) +
  labs(x = 'Death',
       y = 'Creatinine phosphokinase') +
  scale_fill_manual(values = c("black", "gray"))+
  theme_bw()

# Splitting the data

df_split <- initial_split(df, prop = .8)
df_train <- training(df_split)
df_test <-  testing(df_split)

# Creating a cross-validation folds

df_folds <-  vfold_cv(df_train, v = 10)

# Building a recipe

hf_recipe <- recipe(death ~., data = df_train) %>% 
  step_dummy(sex) %>% 
  step_normalize(age, serum_creatinine:time)

wf <- workflow() %>% 
  add_recipe(hf_recipe)

# Specify the model

tune_spec <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lasso_grid <- tune_grid(add_model(wf, tune_spec),
                        resamples = df_folds,
                        grid = grid_regular(penalty(), levels = 30))

highest_roc_auc_lasso <- lasso_grid %>% select_best("roc_auc")

# Fitting the final model

final_lasso <- finalize_workflow(
  add_model(wf, tune_spec),
  highest_roc_auc_lasso)

# Model Evaluation

last_fit(final_lasso, df_split) %>% 
  collect_metrics()

last_fit(final_lasso, df_split) %>% 
  collect_predictions() %>% 
  conf_mat(death, .pred_class) %>% 
  autoplot(type = "heatmap")

# Assessing variable importances

var_imp <- final_lasso %>% 
  fit(df_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vi(lambda = highest_roc_auc_lasso$penalty)

var_imp %>%
  filter(Importance > 0) %>% 
  ggplot() +
  geom_col(aes(y = Variable, x = Importance)) +
  theme_bw() +
  labs(title =  "Figure - Variable Importance")
