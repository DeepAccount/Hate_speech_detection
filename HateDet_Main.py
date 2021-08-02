import HateDet_Preprocessing
import HateDet_MLModels
import HateDet_PCA_plots
import HateDet_MLModels_Embedding
import HateDet_Category_MLModels
import HateDet_Category_MLModels_Embedding
import HateDet_Target
import HateDet_LSTM
import HateDet_LSTM_Embedding
import HateDet_Category_LSTM
import HateDet_Category_LSTM_Embedding

"""
Goal: Training and Evaluation
"""

# Hate vs Non Hate

df = HateDet_Preprocessing.create_clean_file('All Set_hate_non_hate.xlsx', 'clean1.xlsx', 'output')
#Visualize data
HateDet_PCA_plots.draw_pca_plot(df)
# Model creation and evaluation
HateDet_MLModels.run_ml_models_hate_detection(df, 'output')
HateDet_MLModels_Embedding.run_embedding_hate_detection(df)
HateDet_LSTM.run_lstm_hate_detection(df, 'output')
HateDet_LSTM_Embedding.run_lstm_hate_detection(df, 'output')

#Category

df_category = HateDet_Preprocessing.create_clean_file('All Set.xlsx', 'clean2.xlsx', 'output')
#Model creation and evaluation
HateDet_Category_MLModels.run_ml_models_hate_detection(df_category, 'output')
HateDet_Category_MLModels_Embedding.run_embedding_hate_detection(df_category)
HateDet_Category_LSTM.run_lstm_hate_detection(df_category, 'output')
HateDet_Category_LSTM_Embedding.run_lstm_hate_detection(df_category, 'output')


#Finds Hate targets for all the Hate records and save in a Excel
df_target = HateDet_Preprocessing.create_clean_file_target_identification('All Set.xlsx', 'clean3.xlsx', 'output')
HateDet_Target.generate_targets_fullfile(df_target)


