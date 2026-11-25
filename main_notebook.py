import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Investigating Customer Retention in Sports and Fitness Facilities

    ## 1. Project Overview and Context
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("DadosV3.csv", sep=";")

    df_raw.head()
    return


if __name__ == "__main__":
    app.run()
