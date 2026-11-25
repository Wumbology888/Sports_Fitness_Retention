import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from scipy import stats
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sns.set_theme(style="whitegrid")
    RANDOM_STATE = 42
    return (
        LogisticRegression,
        RANDOM_STATE,
        StandardScaler,
        StratifiedKFold,
        cross_val_score,
        make_pipeline,
        mo,
        mutual_info_classif,
        np,
        pd,
        plt,
        roc_auc_score,
        sklearn,
        sns,
        stats,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Project Overview and Context
    *Authorship: Lead — Abdirahman Mohamed | Contributors — Gabriel Aina, Justin Casey, Richie Adu*

    **Roles:** Abdirahman Mohamed — pipeline, governance, ethics (sections 1, 2, 9); Gabriel Aina — EDA (3); Richie Adu — cleaning (4); Justin Casey — features (5).

    ## Problem and context
    Attrition is the central commercial problem for sports facilities: only 30–60% of fitness-centre members remain after a year (Clavel San Emeterio et al., 2019), and most new members of a Brazilian centre left within three months (Sperandei, Vieira and Reis, 2016). Because most losses happen early, the decision that matters is **which new members to support during their first year**. Using five years of real records from a Lisbon sports facility (Pinheiro and Cavique, 2021), this project asks whether at-risk members can be identified from data the facility already holds. Managers can then target retention effort, and members can be supported before they disengage.

    ## Research questions
    1. Which member and plan characteristics are associated with leaving within 12 months of joining?
    2. Can first-year dropout be predicted better than a naive baseline, without using information that leaks the outcome?
    3. Would such predictions fall disproportionately on particular age or gender groups?

    ## Machine learning framing
    **Supervised binary classification:** the target `dropout_12m` is 1 if a member left within 365 days of joining (section 4). Labels are observed for every member; reinforcement learning does not fit, as there is no environment in which to trial interventions. The business goal fixes the task (Fayyad, Piatetsky-Shapiro and Smyth, 1996). Clavel San Emeterio et al. (2019) framed dropout the same way and report around 70% effectiveness, which gives TB2 a benchmark. **Baseline:** predict dropout for everyone (about 62% of the cohort). Because the classes are imbalanced, TB2 will report recall and precision–recall, not accuracy alone (Saito and Rehmsmeier, 2015).

    **Datasets considered:** this dataset (selected: real records with a documented label); `gym_membership.csv` (rejected after audit: synthetic, with no feature–target relationship); a telecom churn set (rejected: outside the domain).

    ## TB2 plan
    | Week | Work (section) |
    |---|---|
    | 14–15 | Stratified split; majority baseline; logistic regression versus random forest (Breiman, 2001); tuning on training data only, with class weights (6) |
    | 17 | Recall, precision, PR-AUC against the baseline; confusion matrix costed as missed leaver vs wasted offer (7) |
    | 18 | Survival analysis, which uses the censored members excluded in section 4 (8) |
    | 19 | Fairness audit by age band and gender; security threats (9) |
    | 20–21 | Conclusions, limitations, Copilot reflection (10) |

    A sensitivity analysis will compare sign-up-only features with sign-up plus usage rates (section 5). **Literature gaps for TB2:** a dropout-model evaluation study in sports services, a survival-analysis churn reference, and a peer-reviewed source on children's data in analytics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Use of generative AI
    Copilot was used weekly for orientation and checked against the cited literature. Claude (Anthropic) helped audit the data and structure and draft the narrative, which the group reviewed. All numbers come from the notebook's own code; every source was checked by the group.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Dataset Description
    *Authorship: Lead — Abdirahman Mohamed*

    The Sport Services Dataset (Pinheiro and Cavique, 2020; described in Pinheiro and Cavique, 2021) holds real ERP records from a Lisbon sports facility: every member who enrolled between June 2014 and October 2019, with visits logged by RFID access control. Identifiers were removed and IDs regenerated before publication, with the facility's permission. Both the dataset (Mendeley Data, version 1, published 26 September 2020; accessed 25 November 2026) and its data article are licensed under CC BY 4.0, which permits reuse, including adaptation, with attribution. A membership ends when the member asks to leave or reaches two months of unpaid fees. Members still enrolled at extraction (31 October 2019) carry that date as their finish date, with Dropout = 0.

    The cell below loads the file (semicolon-separated, relative path) and runs the size and completeness checks (Paullada et al., 2021).
    """)
    return


@app.cell
def _(mo, pd):
    df_raw = pd.read_csv(mo.notebook_location() / "DadosV3.csv", sep=";")
    audit_shape = pd.Series(
        {
            "rows (members)": len(df_raw),
            "columns": df_raw.shape[1],
            "exact duplicate rows": int(df_raw.duplicated().sum()),
            "duplicate member IDs": int(df_raw["Id"].duplicated().sum()),
            "missing cells": int(df_raw.isna().sum().sum()),
            "dropout rate (raw label)": round(df_raw["Dropout"].mean(), 3),
        },
        name="value",
    )
    audit_shape
    return audit_shape, df_raw


@app.cell
def _(audit_shape, mo):
    mo.md(f"""
    **Interpretation.** {audit_shape['rows (members)']:,} members is ample for a held-out test set, with no duplicates or blank cells. The raw label is imbalanced ({audit_shape['dropout rate (raw label)']:.0%} dropouts). The next cell groups the columns by **what they measure and when**, which matters most for leakage.
    """)
    return


@app.cell
def _(df_raw, pd):
    column_roles = {
        "Identifier": ["Id"],
        "Demographic": ["Age", "Gender"],
        "Enrolment dates": ["EnrollmentStart", "EnrollmentFinish", "LastPeriodStart", "LastPeriodFinish", "DateLastVisit"],
        "Plan / sign-up": ["UseByTime", "AllowedWeeklyVisitsBySLA", "NumberOfReferences", "HasReferences"],
        "Activity type (flags)": [c for c in df_raw.columns if c.endswith("Activities") and c != "NumberOfActivities"] + ["NumberOfActivities"],
        "Usage rate (whole membership)": ["AverageFrequency", "AttendedClassesAverage"],
        "Running total (whole membership)": ["EnrollmentDuration", "LifetimeValue", "NumberOfFrequencies", "AttendedClasses", "NumberOfRenewals"],
        "Last two months / at exit": ["DaysWithoutFrequency", "AllowedNumberOfVisitsBySLA", "RealNumberOfVisits", "Ratio"],
        "Binned copy of another column": [c for c in df_raw.columns if "Class1" in c or "Class2" in c or c == "AgeClass"],
        "Target": ["Dropout"],
    }
    column_role = {c: role for role, cols in column_roles.items() for c in cols}
    role_table = (
        pd.Series(column_role, name="role")
        .reset_index()
        .rename(columns={"index": "column"})
        .groupby("role")["column"]
        .agg(n="count", columns=lambda s: ", ".join(s))
        .reset_index()
    )
    assert len(column_role) == df_raw.shape[1], "every column must have a role"
    role_table
    return (column_role,)


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** Only the demographic, plan and activity columns describe a member at joining. Running totals grow with membership length, and the "last two months" columns measure the exit period itself, so both carry outcome information (tested in section 3). Twelve columns are binned copies. Next: range and consistency checks.
    """)
    return


@app.cell
def _(df_raw, pd):
    _d = df_raw.copy()
    for _c in ["EnrollmentStart", "EnrollmentFinish", "DateLastVisit"]:
        _d[_c] = pd.to_datetime(_d[_c])
    _activity_cols = [c for c in _d.columns if c.endswith("Activities") and c != "NumberOfActivities"]
    _extract_date = pd.Timestamp("2019-10-31")
    range_report = pd.DataFrame(
        [
            ("Age between 0 and 100", _d["Age"].between(0, 100).all(), f"range {_d['Age'].min()}–{_d['Age'].max()}"),
            ("Age above 0", (_d["Age"] > 0).all(), f"{(_d['Age'] == 0).sum()} members recorded as age 0"),
            ("Member is an adult (18+)", (_d["Age"] >= 18).all(), f"{(_d['Age'] < 18).sum():,} under 18, {(_d['Age'] < 10).sum():,} under 10"),
            ("Finish date on/after start date", (_d["EnrollmentFinish"] >= _d["EnrollmentStart"].dt.normalize()).all(), ""),
            ("Last visit on/before finish date", (_d["DateLastVisit"] <= _d["EnrollmentFinish"] + pd.Timedelta(days=1)).all(), ""),
            ("HasReferences = (NumberOfReferences > 0)", (_d["HasReferences"] == (_d["NumberOfReferences"] > 0).astype(int)).all(), "redundant pair"),
            ("NumberOfActivities = sum of activity flags", (_d["NumberOfActivities"] == _d[_activity_cols].sum(axis=1)).mean() == 1, f"{(_d['NumberOfActivities'] == _d[_activity_cols].sum(axis=1)).mean():.1%} match"),
            ("No constant columns", _d.nunique().min() > 1, ", ".join(_d.columns[_d.nunique() == 1])),
            ("Every active member (Dropout = 0) finishes on extract date", (_d.loc[_d["Dropout"] == 0, "EnrollmentFinish"] == _extract_date).all(), f"{(_d.loc[_d['Dropout'] == 0, 'EnrollmentFinish'] == _extract_date).sum():,} of {(_d['Dropout'] == 0).sum():,}"),
        ],
        columns=["check", "passed", "detail"],
    )
    range_report
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** Three findings shape the project. **Children:** a large minority are under 18, and the 19 age-0 records are implausible. **The finish date encodes the label:** every active member ends on the extraction date. **Redundancy:** `HasReferences` duplicates `NumberOfReferences`, two activity flags are always zero, and `NumberOfActivities` is exactly the sum of the flags. Next: does the label mean the same thing for every enrolment year?
    """)
    return


@app.cell
def _(df_raw, pd):
    _start = pd.to_datetime(df_raw["EnrollmentStart"])
    censoring_table = (
        df_raw.assign(start_year=_start.dt.year)
        .groupby("start_year")["Dropout"]
        .agg(members="size", dropout_rate="mean")
        .round(3)
    )
    censoring_table
    return (censoring_table,)


@app.cell
def _(censoring_table, mo):
    mo.md(f"""
    **Interpretation.** Recorded dropout falls from {censoring_table['dropout_rate'].iloc[0]:.0%} ({censoring_table.index[0]} joiners) to {censoring_table['dropout_rate'].iloc[-1]:.0%} ({censoring_table.index[-1]} joiners). Recent members have simply had less time to leave: this is **right-censoring**, which would make enrolment date look predictive. Section 4 fixes it with a 12-month window.

    | Criterion | Verdict |
    |---|---|
    | Size / completeness | 14,942 members; no blanks or duplicates; 19 implausible ages |
    | Relevance / provenance | Real dropout records; peer-reviewed data article |
    | Licence | CC BY 4.0 (article); dataset page to confirm |
    | Bias risk | Imbalance, censoring, children, 60/40 female/male |

    **Decision: accept**, handling leakage, censoring and children in sections 4–5. Paullada et al. (2021) show that reusing data without understanding how its labels were produced misleads; the finish-date finding is such a case.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Exploratory Data Analysis
    *Authorship: Lead — Gabriel Aina*

    Following Tukey (1977), the data are explored before hypotheses are fixed. The raw data are used, so that the findings justify the cleaning in section 4. First, summary statistics.
    """)
    return


@app.cell
def _(df_raw):
    eda_numeric = ["Age", "EnrollmentDuration", "AverageFrequency", "AttendedClassesAverage", "AllowedWeeklyVisitsBySLA", "LifetimeValue", "NumberOfFrequencies"]
    summary_stats = df_raw[eda_numeric].describe().T.round(2)
    summary_stats
    return (summary_stats,)


@app.cell
def _(mo, summary_stats):
    mo.md(f"""
    **Interpretation.** Median age is {summary_stats.loc['Age', '50%']:.0f}, but the minimum of {summary_stats.loc['Age', 'min']:.0f} shows children are mixed in. Usage is right-skewed (median {summary_stats.loc['AverageFrequency', '50%']:.2f} weekly visits, maximum {summary_stats.loc['AverageFrequency', 'max']:.2f}), which suits tree models or a log transform in TB2. Most plans allow 7 visits a week.
    """)
    return


@app.cell
def _(df_raw, plt, sns):
    _fig, _ax = plt.subplots(2, 3, figsize=(16, 8))
    sns.histplot(df_raw["Age"], bins=40, ax=_ax[0, 0])
    _ax[0, 0].axvline(18, color="red", linestyle="--", label="18")
    _ax[0, 0].legend()
    _ax[0, 0].set_title("Age")
    sns.histplot(df_raw["EnrollmentDuration"], bins=40, ax=_ax[0, 1])
    _ax[0, 1].set_title("Enrolment duration (months)")
    sns.histplot(df_raw["AverageFrequency"], bins=40, ax=_ax[0, 2])
    _ax[0, 2].set_title("Average weekly visits")
    sns.histplot(df_raw["AttendedClassesAverage"], bins=40, ax=_ax[1, 0])
    _ax[1, 0].set_title("Average weekly classes")
    df_raw["AllowedWeeklyVisitsBySLA"].round().value_counts().sort_index().plot.bar(ax=_ax[1, 1])
    _ax[1, 1].set_title("Allowed weekly visits (plan)")
    _act = df_raw[[c for c in df_raw.columns if c.endswith("Activities") and c != "NumberOfActivities"]].mean().sort_values()
    _act.plot.barh(ax=_ax[1, 2])
    _ax[1, 2].set_title("Share of members per activity type")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** Age has a separate child peak below 12. Enrolment duration clusters in the first year, matching Sperandei, Vieira and Reis (2016). Most members take no classes, fitness and water dominate, and rare activity types have too few members to learn from. Next: disguised missing values.
    """)
    return


@app.cell
def _(df_raw, pd):
    _finish = pd.to_datetime(df_raw["EnrollmentFinish"])
    disguised_missing = pd.DataFrame(
        {
            "check": ["Age recorded as 0", "Zero average weekly visits", "No classes ever attended", "Finish date = extract date (still active)"],
            "members": [
                int((df_raw["Age"] == 0).sum()),
                int((df_raw["AverageFrequency"] == 0).sum()),
                int((df_raw["AttendedClasses"] == 0).sum()),
                int((_finish == pd.Timestamp("2019-10-31")).sum()),
            ],
        }
    )
    disguised_missing["share"] = (disguised_missing["members"] / len(df_raw)).round(3)
    disguised_missing
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Interpretation.** Age 0 is a missing-value placeholder, removed with the children in section 4. Zero classes is real behaviour. The extract-date finish means "outcome unknown", not "stayed". No imputation is needed. Next: raw dropout by group, with chi-square tests.
    """)
    return


@app.cell
def _(df_raw, pd, stats):
    _d = df_raw.assign(
        age_band=pd.cut(df_raw["Age"], [-1, 17, 25, 35, 50, 120], labels=["<18", "18–25", "26–35", "36–50", "50+"]),
        plan=df_raw["AllowedWeeklyVisitsBySLA"].round().map(lambda v: "unrestricted (7)" if v == 7 else "restricted (<7)"),
        takes_classes=(df_raw["AttendedClasses"] > 0).map({True: "yes", False: "no"}),
        has_ties=(df_raw["NumberOfReferences"] > 0).map({True: "yes", False: "no"}),
    )
    _rows = []
    for _g in ["age_band", "Gender", "plan", "takes_classes", "has_ties"]:
        _p = stats.chi2_contingency(pd.crosstab(_d[_g], _d["Dropout"]))[1]
        for _lvl, _sub in _d.groupby(_g, observed=True):
            _rows.append({"group": _g, "level": str(_lvl), "members": len(_sub), "dropout_rate": round(_sub["Dropout"].mean(), 3), "chi2_p": round(_p, 4)})
    group_rates = pd.DataFrame(_rows)
    group_rates
    return (group_rates,)


@app.cell
def _(group_rates, mo):
    _r = group_rates.set_index(["group", "level"])["dropout_rate"]
    mo.md(f"""
    **Interpretation.** Under-18s ({_r[('age_band', '<18')]:.0%}) behave differently from adults. Class attendees ({_r[('takes_classes', 'yes')]:.0%} vs {_r[('takes_classes', 'no')]:.0%}) and members with social ties ({_r[('has_ties', 'yes')]:.0%} vs {_r[('has_ties', 'no')]:.0%}) leave less, consistent with social support aiding adherence. These rates are censored, so they are re-checked on the clean cohort in section 5. Age band is strongly associated with dropout, a fairness risk for RQ3 (section 9). Next: a **leakage screen**, scoring each column alone by ROC-AUC (0.5 = no information).
    """)
    return


@app.cell
def _(column_role, df_raw, pd, plt, roc_auc_score, sns):
    _num = df_raw.select_dtypes("number").drop(columns=["Dropout"])
    _num = _num.loc[:, (_num.nunique() > 1) & ~_num.columns.str.contains("Class")]
    leakage_screen = pd.DataFrame(
        {
            "feature": _num.columns,
            "single_feature_auc": [max(roc_auc_score(df_raw["Dropout"], _num[c]), 1 - roc_auc_score(df_raw["Dropout"], _num[c])) for c in _num.columns],
            "role": [column_role[c] for c in _num.columns],
        }
    ).sort_values("single_feature_auc", ascending=False)
    _fig, _ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=leakage_screen, y="feature", x="single_feature_auc", hue="role", dodge=False, ax=_ax)
    _ax.axvline(0.5, color="black", linewidth=1)
    _ax.set_xlim(0.45, 0.85)
    _ax.set_title("Leakage screen: single-feature AUC against raw Dropout")
    plt.tight_layout()
    _fig
    return (leakage_screen,)


@app.cell
def _(leakage_screen, mo):
    _top = leakage_screen.head(5)
    mo.md(f"""
    **Interpretation.** The top five, {', '.join(f'`{f}`' for f in _top['feature'])}, all come from the exit-period or running-total roles. `{_top.iloc[0]['feature']}` alone reaches an AUC of {_top.iloc[0]['single_feature_auc']:.2f} because it measures the gap before exit. This is leakage as Kaufman et al. (2012) define it: information not legitimately available at prediction time. Sign-up and rate columns show modest, believable signal (about 0.50–0.61). Next: correlations among the candidates.
    """)
    return


@app.cell
def _(df_raw, plt, sns):
    _cols = ["Age", "AllowedWeeklyVisitsBySLA", "UseByTime", "NumberOfReferences", "WaterActivities", "FitnessActivities", "CombatActivities", "NumberOfActivities", "AverageFrequency", "AttendedClassesAverage", "Dropout"]
    corr_matrix = df_raw[_cols].corr().round(2)
    _fig, _ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0, ax=_ax)
    _ax.set_title("Correlation between candidate features and Dropout")
    plt.tight_layout()
    _fig
    return (corr_matrix,)


@app.cell
def _(corr_matrix, mo):
    mo.md(f"""
    **Interpretation.** No candidate correlates strongly with the label (maximum |r| = {corr_matrix['Dropout'].drop('Dropout').abs().max():.2f}), so prediction must combine features. Water and fitness activities are opposed ({corr_matrix.loc['WaterActivities', 'FitnessActivities']:.2f}), reflecting distinct pool and gym members. Pearson correlation misses non-linear effects (section 5 uses mutual information).

    **Key findings:** (1) the outcome leaks into exit-period and cumulative columns; (2) the raw label is right-censored; (3) children are a distinct population. Each drives a step in section 4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 4. Data Cleaning and Preparation
    *Authorship: Lead — Justin Casey

    Each transformation has its own cell, preceded by the problem, decision and rejected alternative.

    **4.1 Missing values: no imputation.** The mechanism is known (section 3): age 0 goes with the children, and the extract-date finish is handled as censoring. Rubin (1976) and van Buuren (2018) both require understanding the missingness mechanism before choosing a remedy; median-imputing age 0 would invent ages for likely children. Zero duplicates were confirmed in section 2.

    **4.2 Drop binned copies, constant columns and `HasReferences`.** These are pure redundancy; keeping them would inflate apparent importance.
    """)
    return


@app.cell
def _(column_role, df_raw, pd):
    _drop = [c for c, r in column_role.items() if r == "Binned copy of another column"]
    _drop += list(df_raw.columns[df_raw.nunique() == 1]) + ["HasReferences"]
    df_step1 = df_raw.drop(columns=_drop)
    step1_summary = pd.Series({"columns before": df_raw.shape[1], "columns after": df_step1.shape[1], "dropped": ", ".join(_drop)})
    step1_summary
    return df_step1, step1_summary


@app.cell
def _(mo, step1_summary):
    mo.md(f"""
    **Result:** {step1_summary['columns before']} → {step1_summary['columns after']} columns, with no information lost. **4.3 Parse dates** and derive `days_enrolled` and `start_month`, needed for the target and for seasonality.
    """)
    return


@app.cell
def _(df_step1, pd):
    df_step2 = df_step1.copy()
    for _c in ["EnrollmentStart", "EnrollmentFinish", "LastPeriodStart", "LastPeriodFinish", "DateLastVisit"]:
        df_step2[_c] = pd.to_datetime(df_step2[_c])
    df_step2["days_enrolled"] = (df_step2["EnrollmentFinish"] - df_step2["EnrollmentStart"]).dt.days
    df_step2["start_month"] = df_step2["EnrollmentStart"].dt.month
    df_step2[["EnrollmentStart", "EnrollmentFinish", "days_enrolled", "start_month"]].describe(include="all").loc[["min", "max", "mean"]]
    return (df_step2,)


@app.cell
def _(mo):
    mo.md(r"""
    **Result:** all dates parse, and no durations are negative.

    **4.4 Adults only.** Parents decide whether children stay, and children's data need extra protection (Recital 38). *Rejected:* using age as a feature, which would mix two business problems. **4.5 Joined before November 2018,** so every member has at least 12 months of observation. *Rejected:* survival analysis now; it is deferred to TB2 week 18.
    """)
    return


@app.cell
def _(df_step2, pd):
    _adult = df_step2["Age"] >= 18
    _observable = df_step2["EnrollmentStart"] < pd.Timestamp("2018-11-01")
    df_cohort = df_step2[_adult & _observable].copy()
    cohort_flow = pd.Series(
        {
            "all members": len(df_step2),
            "excluded: under 18": int((~_adult).sum()),
            "excluded: joined from Nov 2018 (adults)": int((_adult & ~_observable).sum()),
            "analysis cohort": len(df_cohort),
        },
        name="members",
    )
    cohort_flow
    return cohort_flow, df_cohort


@app.cell
def _(cohort_flow, mo):
    mo.md(f"""
    **Result:** {cohort_flow['analysis cohort']:,} adults remain ({cohort_flow['analysis cohort'] / cohort_flow['all members']:.0%}). The exclusions are a limitation (section 9). **4.6 Target:** `dropout_12m` = 1 if the member left within 365 days. It is uncensored for everyone in the cohort and matches the one-year horizon of Clavel San Emeterio et al. (2019).
    """)
    return


@app.cell
def _(df_cohort, pd):
    df_target = df_cohort.copy()
    df_target["dropout_12m"] = ((df_target["Dropout"] == 1) & (df_target["days_enrolled"] <= 365)).astype(int)
    target_check = pd.DataFrame(
        {
            "dropout_12m rate": df_target.groupby(df_target["EnrollmentStart"].dt.year)["dropout_12m"].mean().round(3),
            "raw Dropout rate": df_target.groupby(df_target["EnrollmentStart"].dt.year)["Dropout"].mean().round(3),
        }
    )
    target_check.loc["all"] = [round(df_target["dropout_12m"].mean(), 3), round(df_target["Dropout"].mean(), 3)]
    target_check
    return df_target, target_check


@app.cell
def _(mo, target_check):
    mo.md(f"""
    **Result:** {target_check.loc['all', 'dropout_12m rate']:.1%} left within a year (raw label: {target_check.loc['all', 'raw Dropout rate']:.1%}). The raw rate drifts with enrolment year while the 12-month rate is broadly stable from 2015 to 2018, confirming the censoring fix. **4.7 Drop IDs, dates, the raw label and leaking columns.** Age and gender are kept for fairness auditing only.
    """)
    return


@app.cell
def _(column_role, df_raw, df_target, pd):
    leaky_columns = [c for c, r in column_role.items() if r in ("Running total (whole membership)", "Last two months / at exit") and c in df_target.columns]
    _drop = ["Id", "Dropout", "EnrollmentStart", "EnrollmentFinish", "LastPeriodStart", "LastPeriodFinish", "DateLastVisit", "days_enrolled"] + leaky_columns
    df_clean = df_target.drop(columns=_drop).reset_index(drop=True)
    clean_summary = pd.DataFrame(
        {
            "raw data": [df_raw.shape[0], df_raw.shape[1], "Dropout (censored)"],
            "clean data": [df_clean.shape[0], df_clean.shape[1], "dropout_12m (fixed window)"],
        },
        index=["rows", "columns", "target"],
    )
    clean_summary
    return clean_summary, df_clean, leaky_columns


@app.cell
def _(clean_summary, leaky_columns, mo):
    mo.md(f"""
    **Result:** {clean_summary.loc['rows', 'clean data']:,} rows × {clean_summary.loc['columns', 'clean data']} columns. Leaking columns removed: {', '.join(f'`{c}`' for c in leaky_columns)}.

    | Step | Transformation | Justification |
    |---|---|---|
    | 4.1 | No imputation | Known mechanism (Rubin, 1976; van Buuren, 2018) |
    | 4.2 | Drop redundant columns | Duplicated information |
    | 4.3 | Parse dates | Needed for target and season |
    | 4.4 | Adults only | Different decision-maker; Recital 38 |
    | 4.5 | Joined before Nov 2018 | Removes censoring |
    | 4.6 | 12-month target | Fixed, comparable horizon |
    | 4.7 | Drop leaking columns | Kaufman et al. (2012) |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 5. Feature Engineering and Selection
    *Authorship: Lead — Justin Casey*

    Feature construction usually matters more than algorithm choice (Domingos, 2012). Every feature is checked for leakage. **Tier A** features are known at sign-up. **Tier B** features are usage *rates* over the membership: they don't grow with membership length, but a leaver's rate covers a shorter period, so they carry partial outcome exposure (tested in TB2's sensitivity analysis).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature 1: `unrestricted_plan`
    **What:** 1 if the plan allows unlimited weekly visits (7).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why predictive:** a fixed-schedule plan signals deliberate commitment; open-ended intentions are often overestimated (DellaVigna and Malmendier, 2006).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Leakage check:** describes what was bought, not behaviour. Tier A.
    """)
    return


@app.cell
def _(df_clean):
    feat_unrestricted = (df_clean["AllowedWeeklyVisitsBySLA"].round() == 7).astype(int).rename("unrestricted_plan")
    feat_unrestricted.value_counts()
    return (feat_unrestricted,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature 2: `start_season`
    **What:** enrolment season — Jan–Feb, summer, Sep–Oct or other (one-hot encoded).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why predictive:** New Year and post-summer joiners may act on short-lived motivation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Leakage check:** known on joining; the month is used, not the year, so it cannot proxy for censoring. Tier A.
    """)
    return


@app.cell
def _(df_clean, pd):
    _season_map = {1: "jan_feb", 2: "jan_feb", 6: "summer", 7: "summer", 8: "summer", 9: "sep_oct", 10: "sep_oct"}
    feat_season = pd.get_dummies(df_clean["start_month"].map(_season_map).fillna("other"), prefix="season", dtype=int)
    feat_season.sum()
    return (feat_season,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature 3: `class_share`
    **What:** weekly classes ÷ weekly visits (0 if no visits, capped at 1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why predictive:** class attendees leave less (section 3); a share isolates class-centred members regardless of how often they visit.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Leakage check:** built from two rates only; no totals or exit-period data. Tier B.
    """)
    return


@app.cell
def _(df_clean, np, pd):
    _ratio = np.where(df_clean["AverageFrequency"] > 0, df_clean["AttendedClassesAverage"] / df_clean["AverageFrequency"], 0.0)
    feat_class_share = pd.Series(np.clip(_ratio, 0, 1), index=df_clean.index, name="class_share")
    feat_class_share.describe().round(3)
    return (feat_class_share,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature 4: `plan_utilisation`
    **What:** weekly visits ÷ weekly visits allowed (capped at 1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why predictive:** under-using a paid entitlement is the intention–behaviour gap described by DellaVigna and Malmendier (2006).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Leakage check:** whole-membership rate, unlike the excluded exit-period `Ratio`. Tier B.
    """)
    return


@app.cell
def _(df_clean):
    feat_utilisation = (df_clean["AverageFrequency"] / df_clean["AllowedWeeklyVisitsBySLA"]).clip(0, 1).rename("plan_utilisation")
    feat_utilisation.describe().round(3)
    return (feat_utilisation,)


@app.cell
def _(mo):
    mo.md(r"""
    The cell below assembles the candidates. Gender (as `is_male`) and age go into a separate **audit frame** that never enters selection; the activity flags are already binary.
    """)
    return


@app.cell
def _(
    df_clean,
    feat_class_share,
    feat_season,
    feat_unrestricted,
    feat_utilisation,
    pd,
):
    tier_a = ["UseByTime", "NumberOfReferences", "AthleticsActivities", "WaterActivities", "FitnessActivities", "TeamActivities", "RacketActivities", "CombatActivities", "SpecialActivities", "OtherActivities", "NumberOfActivities", "unrestricted_plan"] + list(feat_season.columns)
    tier_b = ["AverageFrequency", "AttendedClassesAverage", "class_share", "plan_utilisation"]
    X_all = pd.concat([df_clean, feat_unrestricted, feat_season, feat_class_share, feat_utilisation], axis=1)[tier_a + tier_b].astype(float)
    y_all = df_clean["dropout_12m"]
    audit_frame = pd.DataFrame({"Age": df_clean["Age"], "is_male": (df_clean["Gender"] == "Male").astype(int)})
    leakage_table = pd.DataFrame(
        {"feature": tier_a + tier_b, "tier": ["A: sign-up"] * len(tier_a) + ["B: usage rate"] * len(tier_b), "uses totals, dates or exit-period data?": "no"}
    )
    leakage_table
    return X_all, tier_a, tier_b, y_all


@app.cell
def _(mo):
    mo.md(r"""
    **Leakage prevention** is built in: roles were assigned before analysis (section 2), leaking roles were removed (section 4), and the data are split **before** selection (Kaufman et al., 2012). Next: an 80/20 stratified split.
    """)
    return


@app.cell
def _(RANDOM_STATE, X_all, train_test_split, y_all):
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)
    {"train rows": len(X_train), "test rows": len(X_test), "train dropout rate": round(y_train.mean(), 3), "test dropout rate": round(y_test.mean(), 3)}
    return X_train, y_train


@app.cell
def _(mo):
    mo.md(r"""
    **Filter: mutual information** (Guyon and Elisseeff, 2003) captures non-linear dependence. Scores are averaged over 10 seeds, and a **noise floor** is set at the 95th percentile of scores against 20 shuffled labels. Training data only.
    """)
    return


@app.cell
def _(RANDOM_STATE, X_train, mutual_info_classif, np, pd, y_train):
    _discrete = [X_train[c].nunique() <= 10 for c in X_train.columns]
    mi_scores = pd.DataFrame(
        {s: mutual_info_classif(X_train, y_train, discrete_features=_discrete, random_state=s) for s in range(10)},
        index=X_train.columns,
    ).mean(axis=1)
    _rng = np.random.default_rng(RANDOM_STATE)
    _perm = np.concatenate([mutual_info_classif(X_train, _rng.permutation(y_train.values), discrete_features=_discrete, random_state=i) for i in range(20)])
    mi_noise_floor = float(np.percentile(_perm, 95))
    mi_table = mi_scores.rename("mutual_information").to_frame().assign(above_noise_floor=lambda t: t["mutual_information"] > mi_noise_floor).sort_values("mutual_information", ascending=False).round(4)
    mi_table
    return mi_noise_floor, mi_table


@app.cell
def _(mi_noise_floor, mi_table, mo, tier_b):
    _above = mi_table.index[mi_table["above_noise_floor"]].tolist()
    _above_b = [f for f in _above if f in tier_b]
    mo.md(f"""
    **Interpretation.** {len(_above)} of {len(mi_table)} features clear the floor ({mi_noise_floor:.4f}); the strongest are `{mi_table.index[0]}` and `{mi_table.index[1]}`. {len(_above_b)} of these are Tier B, but sign-up features such as `NumberOfActivities` and `UseByTime` also clear it, so honest signal exists without usage data. A univariate filter misses redundancy and interactions, so an embedded method follows.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Embedded: L1 logistic regression** (Tibshirani, 1996) zeroes the coefficients of uninformative features while considering all features together. `C` is chosen by 5-fold cross-validated ROC-AUC on standardised training data. It is used for selection only, not as the TB2 model.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    RANDOM_STATE,
    StandardScaler,
    StratifiedKFold,
    X_train,
    cross_val_score,
    make_pipeline,
    pd,
    sklearn,
    y_train,
):
    def _l1(c):
        # sklearn >= 1.8 expresses L1 via l1_ratio; older versions via penalty="l1"
        _major, _minor = (int(p) for p in sklearn.__version__.split(".")[:2])
        if (_major, _minor) >= (1, 8):
            return LogisticRegression(solver="saga", l1_ratio=1.0, C=c, max_iter=5000)
        return LogisticRegression(solver="saga", penalty="l1", C=c, max_iter=5000)

    _cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    l1_cv = pd.Series({c: cross_val_score(make_pipeline(StandardScaler(), _l1(c)), X_train, y_train, cv=_cv, scoring="roc_auc").mean() for c in [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]}, name="cv_roc_auc").round(4)
    l1_best_c = float(l1_cv.idxmax())
    _fit = make_pipeline(StandardScaler(), _l1(l1_best_c)).fit(X_train, y_train)
    l1_coefs = pd.Series(_fit[-1].coef_[0], index=X_train.columns, name="l1_coefficient").round(4).sort_values(key=abs, ascending=False)
    l1_cv.to_frame().T
    return l1_best_c, l1_coefs, l1_cv


@app.cell
def _(l1_best_c, l1_coefs, l1_cv, mo):
    _zero = l1_coefs.index[l1_coefs == 0].tolist()
    mo.md(f"""
    **Interpretation.** AUC peaks at C = {l1_best_c} ({l1_cv.max():.3f}), where {len(_zero)} coefficients are zero. The largest are `{l1_coefs.index[0]}` ({l1_coefs.iloc[0]:+.3f}) and `{l1_coefs.index[1]}` ({l1_coefs.iloc[1]:+.3f}); negative means lower risk. Next, the week 9 demonstration: the MI ranking computed on the **full** data.
    """)
    return


@app.cell
def _(X_all, mi_table, mutual_info_classif, pd, stats, y_all):
    _discrete = [X_all[c].nunique() <= 10 for c in X_all.columns]
    _full = pd.DataFrame({s: mutual_info_classif(X_all, y_all, discrete_features=_discrete, random_state=s) for s in range(10)}, index=X_all.columns).mean(axis=1)
    _train = mi_table["mutual_information"].reindex(_full.index)
    leak_demo = pd.Series(
        {
            "Spearman rank agreement (train vs full)": round(stats.spearmanr(_train, _full)[0], 3),
            "top-5 overlap": len(set(_train.nlargest(5).index) & set(_full.nlargest(5).index)),
        }
    )
    leak_demo
    return (leak_demo,)


@app.cell
def _(leak_demo, mo):
    mo.md(f"""
    **Interpretation.** The rankings agree closely here (Spearman {leak_demo.iloc[0]:.2f}; {leak_demo.iloc[1]}/5 shared top features). But any difference is borrowed from test labels, so selection stays on training data. Next: redundant pairs.
    """)
    return


@app.cell
def _(X_train, np):
    _c = X_train.corr()
    redundant_pairs = _c.where(np.triu(np.ones(_c.shape), k=1).astype(bool)).stack().rename("r").loc[lambda s: s.abs() > 0.7].round(3)
    redundant_pairs.to_frame()
    return (redundant_pairs,)


@app.cell
def _(mo, redundant_pairs):
    mo.md(f"""
    **Interpretation.** {len(redundant_pairs)} pair(s) exceed |r| = 0.7: {'; '.join(f'`{a}`–`{b}` ({r:+.2f})' for (a, b), r in redundant_pairs.items()) if len(redundant_pairs) else 'none'}. The weaker member of each pair is dropped. `unrestricted_plan` mirrors `class_share` because restricted plans are class plans.
    """)
    return


@app.cell
def _(X_train, l1_coefs, mi_table, pd, redundant_pairs, tier_a, tier_b):
    # Rule 1: from each redundant pair (strongest first), drop the member with lower mutual information
    _redundant_drop = set()
    for (_a, _b), _r in redundant_pairs.abs().sort_values(ascending=False).items():
        if _a not in _redundant_drop and _b not in _redundant_drop:
            _redundant_drop.add(_a if mi_table.loc[_a, "mutual_information"] < mi_table.loc[_b, "mutual_information"] else _b)
    # Rule 2: binary features held by fewer than 1% of training members are too rare to learn from
    _rare = {c for c in X_train.columns if X_train[c].nunique() == 2 and X_train[c].mean() < 0.01}
    L1_MIN = 0.05  # minimum absolute standardised coefficient counted as a meaningful L1 weight
    _rows = []
    for _f in tier_a + tier_b:
        _mi_ok = bool(mi_table.loc[_f, "above_noise_floor"])
        _l1_ok = abs(l1_coefs[_f]) >= L1_MIN
        if _f in _redundant_drop:
            _decision, _reason = "exclude", "redundant: near copy (|r| > 0.7) of a feature with more signal"
        elif _f in _rare:
            _decision, _reason = "exclude", "rare: held by under 1% of members"
        elif _mi_ok and _l1_ok:
            _decision, _reason = "keep", "both methods find signal"
        elif _mi_ok:
            _decision, _reason = "keep", "above MI noise floor"
        elif _l1_ok:
            _decision, _reason = "keep", "meaningful L1 weight alongside other features"
        else:
            _decision, _reason = "exclude", "near-zero: below MI noise floor and negligible L1 weight"
        _rows.append({"feature": _f, "tier": "A" if _f in tier_a else "B", "mutual_information": mi_table.loc[_f, "mutual_information"], "l1_coefficient": l1_coefs[_f], "decision": _decision, "reason": _reason})
    selection_decision = pd.DataFrame(_rows).sort_values(["decision", "mutual_information"], ascending=[False, False]).reset_index(drop=True)
    final_features = selection_decision.loc[selection_decision["decision"] == "keep", "feature"].tolist()
    selection_decision
    return L1_MIN, final_features


@app.cell
def _(L1_MIN, final_features, mo, tier_a):
    _kept_a = [f for f in final_features if f in tier_a]
    mo.md(f"""
    ## Final feature set
    **{len(final_features)} kept** ({len(_kept_a)} Tier A, {len(final_features) - len(_kept_a)} Tier B): {', '.join(f'`{f}`' for f in final_features)}. **Rule:** drop the weaker member of each redundant pair; drop features held by under 1% of members; keep a feature that clears the MI floor **or** has a standardised L1 coefficient of at least {L1_MIN}. The derived `class_share` and `plan_utilisation` replace the raw rates they were built from.

    **Ethics of selection:** excluding age and gender does not remove proxy bias (activity type may track both), so TB2 audits predictions by group. Inflated performance would be a reliability failure, so leakage control is also an ethical safeguard.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 9. Ethics, Privacy, Security and Limitations
    *Authorship: Lead — Abdirahman Mohamed*

    | Columns | UK GDPR category | Handling |
    |---|---|---|
    | `Id` | Pseudonymous identifier | Dropped |
    | `Age`, `Gender`, dates | Quasi-identifiers | Dates dropped; age and gender audit only |
    | Plan, activities, usage, spend | Behavioural / financial | Modelled (non-leaking only) |
    | Special category (Art. 9) | None present | — |
    | Under-18 records | Children's data (Recital 38) | Excluded |

    The next cell tests the "anonymised" claim: how many members are unique on age, gender and enrolment date?
    """)
    return


@app.cell
def _(df_raw, pd):
    _d = df_raw.assign(
        start_day=pd.to_datetime(df_raw["EnrollmentStart"]).dt.date,
        start_month_year=pd.to_datetime(df_raw["EnrollmentStart"]).dt.to_period("M"),
        age_band=pd.cut(df_raw["Age"], [-1, 17, 25, 35, 50, 120]),
    )

    def _unique_share(cols):
        _sizes = _d.groupby(cols, observed=True).size()
        return round((_sizes == 1).sum() / len(_d), 3)

    reid_table = pd.Series(
        {
            "age + gender + exact enrolment day": _unique_share(["Age", "Gender", "start_day"]),
            "age band + gender + enrolment month (mitigated)": _unique_share(["age_band", "Gender", "start_month_year"]),
        },
        name="share of members uniquely identified",
    )
    reid_table
    return (reid_table,)


@app.cell
def _(mo, reid_table):
    mo.md(f"""
    **Interpretation.** {reid_table.iloc[0]:.0%} of members are unique on exact age, gender and enrolment day, so anyone knowing when a friend joined could find their record. Generalising to age band and month cuts this to {reid_table.iloc[1]:.1%} (k-anonymity; Sweeney, 2002); the modelling data hold no dates at all.

    ## Risk register
    | # | Risk | UK GDPR | Mitigation | Source |
    |---|---|---|---|---|
    | 1 | Re-identification ({reid_table.iloc[0]:.0%} unique) | Art. 5(1)(c); Recital 26 | Drop dates and IDs; band outputs | Sweeney (2002) |
    | 2 | Discriminatory targeting (dropout varies by age) | Art. 5(1)(a) fairness | Age and gender audit-only; TB2 fairness audit | Barocas and Selbst (2016) |
    | 3 | Unexplained automated decisions | Art. 22 | Interpretable baseline; human review of offers | Mittelstadt et al. (2016) |
    | 4 | Inflated performance (leakage, censoring) | Art. 5(1)(d) accuracy | Leakage roles; 12-month window; split before selection | Kaufman et al. (2012) |
    | 5 | Children's data (2,686 records) | Recital 38; Art. 8 | Excluded from analysis | GDPR text (academic source: TB2 gap) |

    **Security:** data stay in the private repository, and the cohort holds no dates or IDs (Art. 5(1)(f); Art. 32); TB2 week 19 covers deployment threats. **Source:** publication was permitted by the facility, and CC BY 4.0 allows reuse with attribution, but the re-identification result means the group treats the data as personal data throughout.

    **Week 10 checkpoint:** re-identification, leakage and children are addressed; risk 5 still needs a peer-reviewed source. **Limitations:** a single facility, 2014–2018 joiners only, partial outcome exposure in Tier B, and rare activity types.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # References
    *UWE Harvard (Cite Them Right). Section 10 will hold the final list in TB2.*

    Barocas, S. and Selbst, A.D. (2016) Big data's disparate impact. *California Law Review*, 104(3), pp. 671–732.

    Breiman, L. (2001) Random forests. *Machine Learning*, 45(1), pp. 5–32.

    Clavel San Emeterio, I., García-Unanue, J., Iglesias-Soler, E., Felipe, J.L. and Gallardo, L. (2019) Prediction of abandonment in Spanish fitness centres. *European Journal of Sport Science*, 19(2), pp. 217–224. doi: 10.1080/17461391.2018.1510036.

    DellaVigna, S. and Malmendier, U. (2006) Paying not to go to the gym. *American Economic Review*, 96(3), pp. 694–719. doi: 10.1257/aer.96.3.694.

    Domingos, P. (2012) A few useful things to know about machine learning. *Communications of the ACM*, 55(10), pp. 78–87.

    Fayyad, U., Piatetsky-Shapiro, G. and Smyth, P. (1996) From data mining to knowledge discovery in databases. *AI Magazine*, 17(3), pp. 37–54.

    Guyon, I. and Elisseeff, A. (2003) An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, pp. 1157–1182.

    Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), pp. 1–21. doi: 10.1145/2382577.2382579.

    Mittelstadt, B.D., Allo, P., Taddeo, M., Wachter, S. and Floridi, L. (2016) The ethics of algorithms: mapping the debate. *Big Data & Society*, 3(2), pp. 1–21.

    Paullada, A., Raji, I.D., Bender, E.M., Denton, E. and Hanna, A. (2021) Data and its (dis)contents: a survey of dataset development and use in machine learning research. *Patterns*, 2(11), 100336. doi: 10.1016/j.patter.2021.100336.

    Pinheiro, P. and Cavique, L. (2021) Regular sports services: dataset of demographic, frequency and service level agreement. *Data in Brief*, 36, 107054. doi: 10.1016/j.dib.2021.107054.

    Rubin, D.B. (1976) Inference and missing data. *Biometrika*, 63(3), pp. 581–592.

    Saito, T. and Rehmsmeier, M. (2015) The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.

    Sperandei, S., Vieira, M.C. and Reis, A.C. (2016) Adherence to physical activity in an unsupervised setting: explanatory variables for high attrition rates among fitness center members. *Journal of Science and Medicine in Sport*, 19(11), pp. 916–920.

    Sweeney, L. (2002) k-anonymity: a model for protecting privacy. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 10(5), pp. 557–570.

    Tibshirani, R. (1996) Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), pp. 267–288.

    Tukey, J.W. (1977) *Exploratory data analysis*. Reading, MA: Addison-Wesley.

    van Buuren, S. (2018) *Flexible imputation of missing data*. 2nd edn. Boca Raton, FL: CRC Press.
    """)
    return


if __name__ == "__main__":
    app.run()
