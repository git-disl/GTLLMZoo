from leaderboard import get_df
def llm_checkbox_fn(columns):
    df = get_df("llm")
    selected_df = df[columns]
    return selected_df

def dataset_checkbox_fn(columns):
    df = get_df("dataset")
    selected_df = df[columns]
    return selected_df

def create_llm_checkbox_callback(
    # input
    columns_checkboxes,
    # outputs
    leaderboard_table,
):
    columns_checkboxes.change(
        fn=llm_checkbox_fn,
        inputs=[columns_checkboxes],
        outputs=[leaderboard_table],
    )

def create_dataset_checkbox_callback(
    # input
    columns_checkboxes,
    # outputs
    leaderboard_table,
):
    columns_checkboxes.change(
        fn=dataset_checkbox_fn,
        inputs=[columns_checkboxes],
        outputs=[leaderboard_table],
    )