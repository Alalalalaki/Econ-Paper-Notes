"""
This file replicates the code in march-prep-wages.
The input is raw-data. The output is cleaned-data.

# means original comment
# @ means additional comment
"""

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm
from patsy import dmatrices

from tqdm import tqdm

input_path = "../../ref/origin/March-CPS/cleaned-data/"


"""
# Basic Wage Series
"""

"""
tab-march-ineq.do
"""


def tabulate_march_basic(year):
    """
    #@
    This is the part of code in tab-march-ineq.do but also used in predict-marchwg-regs-exp.do
    We combine the comment in two code to make it better documented
    #@
    """

    df = pd.read_stata(input_path+"mar"+str(year)[-2:]+".dta")

    # Only keep earnings sample ages 16-64
    df = df.query("16 <= agely <=64")

    # Drop if more than 48 years of experience to allow full range of experience for each education/age category (16, 3 year experience categories, last 45-48)
    df = df.loc[df.exp <= 48]

    # Drop self-employed;
    df = df.query("selfemp == 0")

    # @ ?
    assert df.loc[df.winc_ws.isna(), "hinc_ws"].isna().value_counts().all()
    assert df.loc[df.hinc_ws.isna(), "winc_ws"].isna().value_counts().all()
    df = df.query("winc_ws == winc_ws")  # @ this drop not in tab-march-ineq, but same result

    # Drop allocators? Yes
    df = df.query("allocated == 0")  # @ what is allocators?

    # Create consistent education categories
    if year <= 1991:
        # gen byte educomp= max(0,_grdhi-(grdcom==2))
        # tab educomp
        assert df.eval("0 <= educomp <= 18").all()
        df.grdcom = df.grdcom.cat.codes + 1  # @ to be consistent with stat
        df = df.eval("""
                ed8 = educomp<=8
                ed9  = educomp==9
                ed10 = educomp==10
                ed11 = educomp==11
                edhsg = educomp==12 & grdcom==1
                edsmc = (educomp>=13 & educomp<=15) | (educomp==12 & grdcom==2)
                edclg = educomp==16 | educomp==17
                edgtc = educomp>17
        """)
    else:
        try:
            assert df.eval("0 <= grdatn.cat.codes <= 15").all()
            df.grdatn = df.grdatn.cat.codes + 31  # @ to be consistent with stat
        except:
            assert df.eval("31 <= grdatn <= 46").all()
        df = df.eval("""
                ed8  = grdatn<=34
                ed9  = grdatn==35
                ed10 = grdatn==36
                ed11 = grdatn==37
                edhsg = grdatn== 38 | grdatn==39
                edsmc = grdatn>=40 & grdatn<=42
                edclg = grdatn==43
                edgtc = grdatn>=44
        """)

    df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)  # @ turn bool into 0,1
    assert df.eval("ed8+ed9+ed10+ed11+edhsg+edsmc+edclg+edgtc==1").all()

    # Generate education categories
    df = df.eval("""
            edhsd= ed8 + ed9 + ed10 + ed11
            edcat = edhsd + 2*edhsg + 3*edsmc + 4*edclg + 5*edgtc
    """)
    df.edcat.value_counts()
    assert df.eval("1 <= edcat <= 5").all()

    return df


def tabulate_march_inequality(year):
    """
    #
    For years 1964-2009 (year is March year, not earnings year), tabulate:

    These inequality metrics:

    - 90/50, 50/10, 90/10, Vln
    - 60/50, 70/50, 80/50, 95/50, 97/50
    - 50/3, 50/5, 50/20, 50/30, 50/40

    For these samples

    - Males
    - Females
    - Both

    For these wage measures

    - All hourly

    For these conditioning variables

    - raw wage inequality
    - residual wage inequality

    Also note:

    - Always dropping allocators where possible

    D. Autor, 2/24/2004
    D. Autor, 6/15/2004 - Updated for consistency of controls for quantime simulation methods
    M. Anderson, 12/13/2005 - Updated for new quantiles and years
    D. Autor, 9/5/2006. Updated for 2005 March
    M. Wasserman, 10/14/2009 Updated for 2007/8 March
    #
    """

    df = tabulate_march_basic(year)
    df = df.eval("""
            lnwinc = log(winc_ws) + log(gdp)
            lnhinc = log(hinc_ws) + log(gdp)
        """)

    # Full-time and hourly samples
    df = df.eval("ftfy = fulltime*fullyear")
    df.ftfy.describe().to_frame().T
    df = df.eval("""
            ftsamp = (lnwinc == lnwinc) * ftfy * abs(bcwkwgkm-1)
            hrsamp = (lnhinc == lnhinc) * abs(bchrwgkm-1)
        """)
    # @ ftsamp: weekly real wage not none + ftfy + above weekly real wage limit
    # @ hrsamp: hourly real wage not none + above hourly real wage limit

    df.loc[df.ftsamp == 0, "lnwinc"] = np.nan
    df.loc[df.hrsamp == 0, "lnhinc"] = np.nan
    df.query("ftsamp == 1")["lnwinc"].describe().to_frame().T
    df.query("hrsamp == 1")["lnhinc"].describe().to_frame().T
    df = df.query("ftsamp == 1 | hrsamp == 1")

    # Generate experience categories
    df = df.assign(expcat=(df.exp/3).astype(int) + 1)
    df.loc[df.expcat == 17, "expcat"] = 16
    assert df.eval("1<= expcat <= 16").all()

    df.groupby("expcat")["exp"].agg(["mean", "min", "max"])

    # interaction terms - 80 of these
    # @ move to residual wage part

    # Drop reference group's interaction term: HSG with 0-2 years of experience
    # @ simiarly skip now

    df = df.filter(["year", "wgt", "wgt_hrs", "female", "lnwinc", "lnhinc", "hrsamp", "ftsamp", "edcat", "expcat"])

    ###############################################################
    # Summarize raw inequality
    ###############################################################

    pctiles = pd.Series([3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97])
    pctiles_ = pctiles / 100
    tot_pct = pd.DataFrame(index=pctiles)
    tot_stat = pd.DataFrame(index=["mn", "vln"])

    dt = df.query("ftsamp==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_ft_mf"] = [wq.mean, wq.var]

    dt = df.query("ftsamp==1 & female==0")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_m"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_ft_m"] = [wq.mean, wq.var]

    dt = df.query("ftsamp==1 & female==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_f"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_ft_f"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    tot_pct["tot_hr_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_hr_mf"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1 & female==0")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    tot_pct["tot_hr_m"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_hr_m"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1 & female==1")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    tot_pct["tot_hr_f"] = wq.quantile(probs=pctiles_, return_pandas=False)
    tot_stat["tot_hr_f"] = [wq.mean, wq.var]

    df_stat = pd.concat([tot_stat, tot_pct], axis=0, sort=False)

    ###############################################################
    # Summarize residual inequality - Weekly & Hourly
    ###############################################################

    res_pct = pd.DataFrame(index=pctiles)
    res_stat = pd.DataFrame(index=["mn", "vln"])

    dt = df.query("ftsamp==1")
    y, X = dmatrices('lnwinc ~ female + C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt)
    res_stat["res_ft_mf"] = [wq.mean, wq.var]  # @ mean is not necessary but to be consistent
    res_pct["res_ft_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("ftsamp==1 & female==0")
    y, X = dmatrices('lnwinc ~ C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt)
    res_stat["res_ft_m"] = [wq.mean, wq.var]
    res_pct["res_ft_m"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("ftsamp==1 & female==1")
    y, X = dmatrices('lnwinc ~ C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt)
    res_stat["res_ft_f"] = [wq.mean, wq.var]
    res_pct["res_ft_f"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("hrsamp==1")
    y, X = dmatrices('lnhinc ~ female + C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt_hrs).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt_hrs)
    res_stat["res_hr_mf"] = [wq.mean, wq.var]
    res_pct["res_hr_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("hrsamp==1 & female==0")
    y, X = dmatrices('lnhinc ~ C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt_hrs).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt_hrs)
    res_stat["res_hr_m"] = [wq.mean, wq.var]
    res_pct["res_hr_m"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("hrsamp==1 & female==1")
    y, X = dmatrices('lnhinc ~ C(edcat) : C(expcat) - 1', dt, return_type="dataframe")
    X = sm.add_constant(X.drop("C(edcat)[2]:C(expcat)[1]", axis=1))
    res = sm.WLS(y, X, weights=dt.wgt_hrs).fit()
    resid = res.resid
    wq = DescrStatsW(data=resid, weights=dt.wgt_hrs)
    res_stat["res_hr_f"] = [wq.mean, wq.var]
    res_pct["res_hr_f"] = wq.quantile(probs=pctiles_, return_pandas=False)

    df_stat_ = pd.concat([res_stat, res_pct], axis=0)
    df_stat = pd.concat([df_stat, df_stat_], axis=1)

    # march-ineq-data-`1'
    df_stat = df_stat.T.rename_axis('sample').reset_index().assign(year=year)  # @ tidy data

    ###############################################################
    # Percentiles of weekly earnings
    ###############################################################

    # @ simply generate more percentiles under full-time samples
    # @ note here year is march census year thus minus one to be earnings year

    pctiles = pd.Series(range(3, 98))
    pctiles_ = pctiles / 100
    tot_pct = pd.DataFrame(index=pctiles)

    dt = df.query("ftsamp==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("ftsamp==1 & female==0")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_m"] = wq.quantile(probs=pctiles_, return_pandas=False)

    dt = df.query("ftsamp==1 & female==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    tot_pct["tot_ft_f"] = wq.quantile(probs=pctiles_, return_pandas=False)

    # march-pctile-`yr'
    tot_pct = tot_pct.T.rename_axis('sample').reset_index().assign(year=year-1)  # @ tidy data

    # @ the code then combine 1963-2008 generated files
    # @ we remove this as not sure necessary
    # @ actually this part can be combined with #Summarize raw inequality#

    return df_stat, tot_pct


"""
tab-march-ineq-loop.do
"""


def tabulate_march_inequality_loop():

    # Run calculations
    # Assemble data
    ineq_stat = pd.DataFrame()
    # ineq_pct = pd.DataFrame()
    for y in tqdm(range(1964, 2010)):
        df_stat, tot_pct = tabulate_march_inequality(year=y)
        ineq_stat = pd.concat([ineq_stat, df_stat], axis=0, ignore_index=True)
        # ineq_pct = pd.concat([ineq_pct, tot_pct], axis=0, ignore_index=True)

    # Generate 90/50 and 50/10 and 90/10, etc etc
    # @ skip this as the percentile gap can be easily obtained by column calculation

    # @ label the columns
    # @ "tot_ft_mf" : "Total ineq, full-time weekly, males and females"
    # @ "tot_ft_m" : "Total ineq, full-time weekly, males"
    # @ "tot_ft_f" : "Total ineq, full-time weekly, females"
    # @ "tot_hr_mf" : "Total ineq, all hourly, males and females"
    # @ "tot_hr_m" : "Total ineq, all hourly, males"
    # @ "tot_hr_f" : "Total ineq, all hourly, females"
    # @ "res_ft_mf" : "Resid ineq, full-time weekly, males and females"
    # @ "res_ft_m" : Resid ineq, full-time weekly, males"
    # @ "res_ft_f" : "Resid ineq, full-time weekly, females"
    # @ "res_hr_mf" : "Resid ineq, all hourly, males and females"
    # @ "res_hr_m" : "Resid ineq, all hourly, males"
    # @ "res_hr_f" : "Resid ineq, all hourly, females"

    # march-ineq-data-1963-2008
    ineq_stat.year = ineq_stat.year - 1

    # @ keep tot_ft_mf percentiles and year, reshape into long data (year x quantile), generate gender dummy "mf"
    # @ save as data_mf
    # @ do the same for tot_ft_m and tot_ft_f, combine these 3 data, save as march-ftfy-quantile-plot-data
    # @ skip this as can be easily down from our tidy data

    return ineq_stat


"""
# College High School Gap Series
# By potential experience, using regression
"""

"""
predict-marchwg-regs-exp.do
"""


def predict_archwg_regs_exp(year):
    """
    #
    For years 1964-2009 (year is March year, not earnings year), tabulate:

    - raw wage inequality
    - residual wage inequality
    - regression estimated education diffs (AKK style)
    - regression predicted wages (Katz-Murphy style)

    - This program predicts wages for 5 education levels by 5 experience categories by 2 genders:

      - Ed: hsd/hsg/smc/clg/gtc
      - Exp 5/15/25/35/45

    - Note: This program is adapted from one that used 8 education categories. We use some of the legacy
            code and drop the extraneous categories.

    Autor, 6/12/2004
    Autor, 10/9/2006: Use March 2006 data
    Wasserman, 10/2009: Use March 2007/8/9
    #
    """
    df = tabulate_march_basic(year)

    df = df.eval("""
            lnwinc = log(winc_ws)
            lnhinc = log(hinc_ws)
        """)

    # Create experience categories
    # @ the code split exp by [0,9] [10,19], ... however for years >= 1992 there can be 9.x 19.x in exp
    # @ here I split by [0,10), [10,20), ... thus the result might be differnt from the original one
    df = df.rename(columns={"exp": "exp1"})
    assert df.eval("0<= exp1 <=48").all()
    bins = range(0, 51, 10)
    # @ rather than construct dummy columns, here set category column
    df = df.assign(expcat5=pd.cut(df.exp1, bins, right=False, labels=range(5, 46, 10)))

    # experience education interactions;
    def prod_exp_edu_interactions(dx):
        dx = dx.eval("""
                exp2 = (exp1**2)/100
                exp3 = (exp1**3)/1000
                exp4 = (exp1**4)/10000

                e1edhsd = exp1*edhsd
                e1edsmc = exp1*edsmc
                e1edclg = exp1*(edclg+edgtc)

                e2edhsd = exp2*edhsd
                e2edsmc = exp2*edsmc
                e2edclg = exp2*(edclg+edgtc)

                e3edhsd = exp3*edhsd
                e3edsmc = exp3*edsmc
                e3edclg = exp3*(edclg+edgtc)

                e4edhsd = exp4*edhsd
                e4edsmc = exp4*edsmc
                e4edclg = exp4*(edclg+edgtc)
        """)
        return dx
    df = prod_exp_edu_interactions(df)

    # Gender interactions with experience, race, region, part-time;
    df = df.eval("""
            pt = abs(fulltime-1)

            fexp1 = female*exp1
            fexp2 = female*exp2
            fexp3 = female*exp3
            fexp4 = female*exp4

            fblack = female*black
            fother = female*other

            fpt = female*pt
    """)

    # Full-time samples
    df = df.eval("ftfy = fulltime*fullyear")

    # @ I skip some summ code here

    # we don't want to drop top-coded obs
    # @ no idea what this is
    df.tcwkwg = 0
    df.tchrwg = 0

    ###############################################################
    # Predict wages -- Weekly, Using Only Obs With $67/Week Or More
    ###############################################################

    # Male regression
    dt = df.query("ftfy==1 & female==0 & tcwkwg==0 & bcwkwgkm==0")
    X_ = ["edhsd", "edsmc", "edclg", "edgtc",
          "exp1", "exp2", "exp3", "exp4",
          "e1edhsd", "e1edsmc", "e1edclg",
          "e2edhsd", "e2edsmc", "e2edclg",
          "e3edhsd", "e3edsmc", "e3edclg",
          "e4edhsd", "e4edsmc", "e4edclg",
          "black", "other"]
    y, X = dt.lnwinc, dt[X_]
    X = sm.add_constant(X)
    res = sm.WLS(y, X, weights=dt.wgt).fit()

    # Predict values
    # Now conduct 25 predictions based upon the the 5 education and 5 experience categories
    edcat = range(5)
    edcol = ["edhsd", "edhsg", "edsmc", "edclg", "edgtc"]
    expcat = range(5, 46, 10)
    expcol = [f"exp{i}" for i in expcat]
    s = pd.DataFrame(0, index=pd.MultiIndex.from_product([edcol, expcol]), columns=X_)
    for ed in edcat:
        n = edcol[ed]
        if n != "edhsg":
            s.loc[n, n] = 1
    s = s.swaplevel()
    for i, exp in enumerate(expcat):
        n = expcol[i]
        s.loc[n, "exp1"] = exp
    X_cf = sm.add_constant(prod_exp_edu_interactions(s))
    marchwk_ed_exp_m = res.predict(X_cf)  # @ tidy dataframe for marchwk-`ed'-exp`exp'-m

    # Female regression
    dt = df.query("ftfy==1 & female==1 & tcwkwg==0 & bcwkwgkm==0")
    y, X = dt.lnwinc, dt[X_]
    X = sm.add_constant(X)
    res = sm.WLS(y, X, weights=dt.wgt).fit()

    # Predict values
    # Now conduct 25 predictions based upon the the 5 education and 5 experience categories
    marchwk_ed_exp_f = res.predict(X_cf)  # @ tidy dataframe for marchwk-`ed'-exp`exp'-f

    # Compile all male/female predicted weekly wages
    pwkwageskm = s.assign(plnwkw_m=marchwk_ed_exp_m, plnwkw_f=marchwk_ed_exp_f)

    # @ label of edcat 1 "Hsd" 2 "Hsg" 3 "Smc" 4 "Clg" 5 "Gtc"
    pwkwageskm = pwkwageskm.eval("edcat = edhsd + 2*edhsg + 3*edsmc + 4*edclg + 5*edgtc")
    assert pwkwageskm.eval("1 <= edcat <= 5").all()

    ###############################################################
    # Predict hourly wages --  Using Only Obs With $67/Week Or More
    ###############################################################

    return pwkwageskm


"""
assemb-march-lswts-exp.do
"""


"""
assemb-marchwg-regs-exp.do
"""


"""
calc-marchwg-byexp.do
"""


def main():
    # tabulate_march_inequality(1997)
    print(tabulate_march_inequality_loop())
    pass


if __name__ == "__main__":
    main()
