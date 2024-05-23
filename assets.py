custom_css = """
.logo {
    width: 300px;
    height: auto;
    margin: 0 auto;
    max-width: 100%
    object-fit: contain;
}
.text {
    font-size: 16px !important;
}
.tabs button {
    font-size: 20px;
}
.subtabs button {
    font-size: 20px;
}
.descriptive-text span {
    font-size: 16px !important;
}
#control-panel span {
    font-size: 20px !important;
}
#column-selector span {
    font-size: 20px !important;
}
#search-bar span {
    font-size: 16px !important;
}
#threshold-slider span {
    font-size: 16px !important;
}
#memory-slider span {
    font-size: 16px !important;
}
#columns-checkboxes span {
    font-size: 16px !important;
}
#backend-checkboxes span {
    font-size: 16px !important;
}
#dtype-checkboxes span {
    font-size: 16px !important;
}
#optimization-checkboxes span {
    font-size: 16px !important;
}
#quantization-checkboxes span {
    font-size: 16px !important;
}
#kernel-checkboxes span {
    font-size: 16px !important;
}
/* Limit the width of the first AutoEvalColumn so that names don't expand too much */
#llm-leaderboard td:first-child,
#llm-leaderboard th:first-child {
    max-width: 350px;
    overflow: auto;
    white-space: nowrap;
}
/* Limit the width of the first AutoEvalColumn so that names don't expand too much */
#llm-leaderboard td:nth-child(2),
#llm-leaderboard th:nth-child(2) {
    max-width: 350px;
    overflow: auto;
    white-space: nowrap;
    font-size: 13px !important;
}
#llm-columns-checkboxes span {
    font-size: 14px !important;
}

/* Full width space */
.gradio-container {
  max-width: 95%!important;
}
"""