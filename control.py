from leaderboard import get_df
def select(
    search,     
    basic_info_checkboxes,
    benchmark_checkboxes,
    efficiency_checkboxes,
    safety_checkboxes,
    arena_checkboxes,
    df):
    selected_df = df[
        df["Model_name"].str.contains(search, case=False)
    ]
    columns = basic_info_checkboxes+benchmark_checkboxes+efficiency_checkboxes+safety_checkboxes+arena_checkboxes
    selected_df = selected_df[columns]
    selected_df = selected_df.drop_duplicates()
    return selected_df

def filter(
    # filters
    type_checkboxes,
    precision_checkboxes,
    weight_checkboxes,
    architecture_checkboxes,
    license_checkboxes,
    hide_checkboxes,
):
    df = get_df("merged_llm")
    filtered_df = df[
        df['Type'].isin(type_checkboxes)
        & df['Precision'].isin(precision_checkboxes)
        & df['Weight type'].isin(weight_checkboxes)
        & df['Architecture'].isin(architecture_checkboxes)
        & df['Hub License'].isin(license_checkboxes)
    ]
    for col in hide_checkboxes:
        filtered_df = filtered_df[filtered_df[col]==False]
    return filtered_df

def fn(
    # column selectors
    search,     
    basic_info_checkboxes,
    benchmark_checkboxes,
    efficiency_checkboxes,
    safety_checkboxes,
    arena_checkboxes,

    # filters
    type_checkboxes,
    precision_checkboxes,
    weight_checkboxes,
    architecture_checkboxes,
    license_checkboxes,
    hide_checkboxes,
):
    filtered_df = filter(
        type_checkboxes,
        precision_checkboxes,
        weight_checkboxes,
        architecture_checkboxes,
        license_checkboxes,
        hide_checkboxes,
    )
    # print("\n"*5)
    # print(df.columns.tolist())
    # print("-"*100)
    # print("\n"*5)
    # exit()
    selected_df = select(
        search,     
        basic_info_checkboxes,
        benchmark_checkboxes,
        efficiency_checkboxes,
        safety_checkboxes,
        arena_checkboxes,
        filtered_df
    )
    return selected_df

def dataset_fn(columns):
    df = get_df("dataset")
    selected_df = df[columns]
    return selected_df

def create_callback(
    search,     
    basic_info_checkboxes,
    benchmark_checkboxes,
    efficiency_checkboxes,
    safety_checkboxes,
    arena_checkboxes,

    filter_button,
    type_checkboxes,
    precision_checkboxes,
    weight_checkboxes,
    architecture_checkboxes,
    license_checkboxes,
    hide_checkboxes,

    leaderboard_table,
):
    filter_button.click(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    search.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    basic_info_checkboxes.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    benchmark_checkboxes.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    efficiency_checkboxes.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    safety_checkboxes.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
    arena_checkboxes.change(
        fn=fn,
        inputs=[
            search,     
            basic_info_checkboxes,
            benchmark_checkboxes,
            efficiency_checkboxes,
            safety_checkboxes,
            arena_checkboxes,

            type_checkboxes,
            precision_checkboxes,
            weight_checkboxes,
            architecture_checkboxes,
            license_checkboxes,
            hide_checkboxes,
        ],
        outputs=[
            leaderboard_table,
        ],
    )
def create_dataset_checkbox_callback(
    # input
    columns_checkboxes,
    # outputs
    leaderboard_table,
):
    columns_checkboxes.change(
        fn=dataset_fn,
        inputs=[columns_checkboxes],
        outputs=[leaderboard_table],
    )