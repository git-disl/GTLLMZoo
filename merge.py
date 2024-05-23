import pandas as pd
import numpy as np

# Load the CSV files
open_llm_lb = pd.read_csv('open_llm_lb.csv')
llm_safety_lb = pd.read_csv('llm_safety_lb.csv')
llm_perf_lb = pd.read_csv('llm_perf_lb.csv')
chatbot_arena_lb = pd.read_csv("chatbot_arena.csv")

print(open_llm_lb.shape[0], llm_safety_lb.shape[0], llm_perf_lb.shape[0], chatbot_arena_lb.shape[0])

# Process llm_perf_lb to extract the model name part from 'Model_link'. legacy, not needed anymore
# llm_perf_lb['Model_name_for_merge'] = llm_perf_lb['Model_link'].apply(lambda x: x.split('https://huggingface.co/')[-1])

# Rename columns for merging. Legacy, not needed anymore
# open_llm_lb.rename(columns={'model_name_for_query': 'Model_name_for_merge'}, inplace=True) #model_name
# llm_safety_lb.rename(columns={'model_name_for_query': 'Model_name_for_merge'}, inplace=True)

# Merge the tables on the processed common keys
merged_df = pd.merge(open_llm_lb, llm_safety_lb, on='Model_name_for_merge', suffixes=('', '_safety'), how='outer')
merged_df = pd.merge(merged_df, llm_perf_lb, on='Model_name_for_merge', suffixes=('', '_perf'), how='outer')
merged_df = pd.merge(merged_df, chatbot_arena_lb, on='Model_name_for_merge', suffixes=('', '_arena'), how='outer')

# print(merged_df.shape[0])
# Handle the columns as specified
merged_df.rename(columns={
    'Average ⬆️': 'Average_open_llm_score ⬆️',
    'Average ⬆️_safety': 'Average_llm_safety_score ⬆️',
    'Rank* (UB)': 'chatbot_arena_rank ⬆️',
    'Votes': 'chatbot_arena_votes',
}, inplace=True)

# Drop unwanted columns
columns_to_drop = ['Open LLM Score (%)', 'Model_link_safety', 'Model_link_perf', 'T', 'T_safety', '95% CI', 'Organization']
merged_df.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)

# Select columns from open_llm_lb to keep
columns_to_keep = ['Type', 'Architecture', 'Precision', 'Hub License', '#Params (B)', 'Hub ❤️', 'Model sha', 'Available on the hub', 'Model_link']
postfix_list = ['_safety', '_perf']
print(merged_df.columns.tolist())
for col in columns_to_keep:
    for postfix in postfix_list:
        if col+postfix in merged_df.columns:
            merged_df.drop(col+postfix, axis=1, inplace=True)

merged_df['Model_link'] = np.where(merged_df['Model_link_arena'].notna(), merged_df['Model_link_arena'], merged_df['Model_link'])
merged_df.drop('Model_link_arena', axis=1, inplace=True)
merged_df['Hub License'] = np.where(merged_df['Hub License_arena'].notna(), merged_df['Hub License_arena'], merged_df['Hub License'])
merged_df.drop('Hub License_arena', axis=1, inplace=True)

# Rename 'Model_name_for_merge' to 'Model_name'
merged_df.rename(columns={'Model_name_for_merge': 'Model_name'}, inplace=True)

# Fill empty 'Model_link' with 'https://huggingface.co/' followed by 'Model_name'
merged_df['Model_link'] = merged_df.apply(
    lambda row: row['Model_link'] if pd.notna(row['Model_link']) else f"https://huggingface.co/{row['Model_name']}",
    axis=1
)

# # print(merged_df.columns.tolist())
# print(merged_df.loc[merged_df['Model_name'] == 'GPT-4o-2024-05-13']['Model_link'])
# exit()

# print(type(merged_df.columns.tolist()))
# print(merged_df.columns.tolist())
# exit()

# change 'Available on the hub'
merged_df['Available on the hub'] = merged_df['Available on the hub'].apply(lambda x: not x if pd.notna(x) else True)
merged_df.rename(columns={'Available on the hub': 'Not Available on HF'}, inplace=True)

merged_df['Merged'] = merged_df['Merged'].apply(lambda x: x if pd.notna(x) else False)
merged_df['MoE'] = merged_df['MoE'].apply(lambda x: x if pd.notna(x) else False)
merged_df['Flagged'] = merged_df['Flagged'].apply(lambda x: x if pd.notna(x) else False)

merged_df['Type'] = merged_df['Type'].apply(lambda x: x if pd.notna(x) else "not available")
merged_df['Architecture'] = merged_df['Architecture'].apply(lambda x: x if pd.notna(x) else "not available")
merged_df['Precision'] = merged_df['Precision'].apply(lambda x: x if pd.notna(x) else "not available")
merged_df['Hub License'] = merged_df['Hub License'].apply(lambda x: x if pd.notna(x) else "not available")
merged_df['Weight type'] = merged_df['Weight type'].apply(lambda x: x if pd.notna(x) else "not available")

merged_df['No Efficiency Data'] = merged_df['Experiment 🧪'].apply(lambda x: True if pd.isna(x) else False)
merged_df['No Safety Data'] = merged_df['Average_llm_safety_score ⬆️'].apply(lambda x: True if pd.isna(x) else False)
merged_df['No Arena Data'] = merged_df['chatbot_arena_rank ⬆️'].apply(lambda x: True if pd.isna(x) else False)

merged_df.to_csv(f"merged.csv", index=False)
print(f"merged.csv exported.")