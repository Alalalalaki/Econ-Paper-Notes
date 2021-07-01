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

import prep_supply

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
    assert df.loc[df.winc_ws.isna(), "hinc_ws"].isna().all(), print(year)
    assert df.loc[df.hinc_ws.isna(), "winc_ws"].isna().all(), print(year)
    df = df.query("winc_ws == winc_ws")  # @ this drop not in tab-march-ineq, but same result

    # Drop allocators? Yes
    df = df.query("allocated == 0")  # @ what is allocators?

    # Create consistent education categories
    if year <= 1991:
        assert df._grdhi.isna().sum() == 0  # @ see both tab-march-ineq.do & prepmarchcell.do
        assert df.eval("0 <= educomp <= 18").all()
        df.grdcom = df.grdcom.cat.codes + 1  # @ to be consistent with stat
        df.grdcom = df.grdcom.replace(3, 1)  # @ turn 3 to 1 in some years
        assert df.eval("1 <= grdcom <= 2").all()
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

    ######################################################################
    # Summarize raw inequality
    ######################################################################

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

    ######################################################################
    # Summarize residual inequality - Weekly & Hourly
    ######################################################################

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

    ######################################################################
    # Percentiles of weekly earnings
    ######################################################################

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

    ######################################################################
    # Predict wages -- Weekly, Using Only Obs With $67/Week Or More
    ######################################################################

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
    s = pd.DataFrame(0, index=pd.MultiIndex.from_product([edcol, expcol]), columns=X_+["edhsg"])
    for ed in edcat:
        n = edcol[ed]
        # if n != "edhsg":
        s.loc[n, n] = 1
    s = s.swaplevel()
    for i, exp in enumerate(expcat):
        n = expcol[i]
        s.loc[n, "exp1"] = exp
    X_cf = s.drop("edhsg", axis=1)
    X_cf = sm.add_constant(prod_exp_edu_interactions(X_cf))
    marchwk_ed_exp_m = res.predict(X_cf)  # @ marchwk-`ed'-exp`exp'-m

    # Female regression
    dt = df.query("ftfy==1 & female==1 & tcwkwg==0 & bcwkwgkm==0")
    y, X = dt.lnwinc, dt[X_]
    X = sm.add_constant(X)
    res = sm.WLS(y, X, weights=dt.wgt).fit()

    # Predict values
    # Now conduct 25 predictions based upon the the 5 education and 5 experience categories
    marchwk_ed_exp_f = res.predict(X_cf)  # @ marchwk-`ed'-exp`exp'-f

    # Compile all male/female predicted weekly wages
    pwkwageskm_m = s.assign(plnwkw=marchwk_ed_exp_m, female=0)
    pwkwageskm_f = s.assign(plnwkw=marchwk_ed_exp_f, female=1)
    pwkwageskm = pd.concat([pwkwageskm_m, pwkwageskm_f], axis=0)

    ######################################################################
    # Predict hourly wages --  Using Only Obs With $67/Week Or More
    ######################################################################

    # Male regression
    dt = df.query("female==0 & tchrwg==0 & bchrwgkm==0")
    X_ = X_ + ["pt"]
    y, X = dt.lnhinc, dt[X_]
    X = sm.add_constant(X)
    wgt = dt.wgt * dt._wkslyr.astype(float)
    res = sm.WLS(y, X, weights=wgt).fit()

    # Predict values
    X_cf = X_cf.assign(pt=0)
    marchhr_ed_exp_m = res.predict(X_cf)  # @ marchhr-`ed'-exp`exp'-m

    # Female regression
    dt = df.query("female==1 & tcwkwg==0 & bcwkwgkm==0")
    y, X = dt.lnhinc, dt[X_]
    X = sm.add_constant(X)
    wgt = dt.wgt * dt._wkslyr.astype(float)
    res = sm.WLS(y, X, weights=wgt).fit()

    # Predict values
    marchhr_ed_exp_f = res.predict(X_cf)  # @ marchhr-`ed'-exp`exp'-f

    # @ data : Predicted wk & hr wages, March `1'
    plnhrw_m = s.assign(plnhrw=marchhr_ed_exp_m, female=0)
    plnhrw_f = s.assign(plnhrw=marchhr_ed_exp_f, female=1)
    plnhrw = pd.concat([plnhrw_m, plnhrw_f], axis=0)
    predwg = pwkwageskm.merge(plnhrw)

    # @ label of edcat 1 "Hsd" 2 "Hsg" 3 "Smc" 4 "Clg" 5 "Gtc"
    predwg = predwg.eval("edcat = edhsd + 2*edhsg + 3*edsmc + 4*edclg + 5*edgtc")
    assert predwg.eval("1 <= edcat <= 5").all()

    predwg = predwg.filter(["edcat", "exp1", "female", "plnwkw", "plnhrw"])

    # @ gdp : "PCE deflator: 2008 basis"
    gdp = df.gdp.unique()
    assert len(gdp) == 1
    predwg = predwg.assign(gdp=gdp[0])
    # @ year : Earnings year
    predwg_mar = predwg.assign(year=year-1)  # @ predwg-mar`1'

    return predwg_mar


def predict_archwg_regs_exp_loop():

    predwg = pd.DataFrame()
    for y in tqdm(range(1964, 2010)):
        predwg_ = predict_archwg_regs_exp(year=y)
        predwg = pd.concat([predwg, predwg_], axis=0)

    return predwg


"""
assemb-march-lswts-exp.do
"""


def assemb_march_lswts_exp():
    """
    #
    Assemble labor supply weights for March data

    Updated 10/9/2006 D. Autor for March 2006 data
    Updated 10/2009 M. Wasserman for March 2008/9 data
    #
    """

    ######################################################################
    # Consolidate marchcells count data into compatible format
    # Need counts for college and high school groups

    # @ recall that as the labor supply weights q_lsweight use all sample
    # @ in original code, here is quite different, actually smaller
    ######################################################################

    # @ marchcells6308
    df = prep_supply.assembcellsmarch()
    df = df.reset_index()[["q_lsweight", "exp", "edcat5", "year", "female"]]
    df = df.rename(columns={"edcat5": "edcat"})

    # Aggregate to experience categories: 5, 15, 25, 35, 45
    df = df.assign(
        expcat=pd.cut(df.exp, bins=range(0, 51, 10),
                      right=False, labels=range(1, 6)))

    lswt = df.groupby(["expcat", "edcat", "female", "year"])["q_lsweight"].sum().rename("lswt")

    # @ label
    # @ expcat "1: 0-9(5), 2:10-19(15), 3:20-29(25), 4:30-39(35) 5:40-48(45)"
    # @ expcat 1 "5" 2 "15" 3 "25" 4 "35" 5 "45"
    # @ lswt "Summ of hours x weeks x weight"

    march_lswts_exp = lswt.reset_index()  # @ data "March labor supply by grouped experience levels"

    return march_lswts_exp


"""
assemb-marchwg-regs-exp.do
"""


def assemb_marchwg_regs_exp():
    """
     #
     Assemble march predicted wage data

     Autor, 6/20/03, 6/12/04, 9/4/2006
     Wasserman 10/2009
     #
     """
    predwg = predict_archwg_regs_exp_loop()

    # Rename and label vars
    predwg = predwg.rename(columns={"exp1": "expcat"})
    predwg = predwg.assign(expcat=predwg.expcat.astype("category").cat.codes+1)  # 5=1 15=2 25=3 35=4 45=5

    # Add in March employment shares for 1963 - 2008
    march_lswts_exp = assemb_march_lswts_exp()
    df = predwg.merge(march_lswts_exp)

    # Calculate fixed weights
    t1 = df.groupby(["year"])["lswt"].sum().rename("t1")
    df = df.merge(t1, left_on="year", right_index=True)
    df = df.eval("normlswt=lswt/t1")
    avlswt = df.groupby(["edcat", "female", "expcat"])["normlswt"].mean().rename("avlswt")
    df = df.merge(avlswt.reset_index(), on=["edcat", "female", "expcat"])

    # @ labels
    # @ lswt : "Labor supply in cell"
    # @ normlswt : "Labor supply share in cell/year"
    # @ avlswt : "Average labor supply share in cell over 1963 - 2008"
    # @ plnhrw : "Pred ln hr wg"
    # @ plnwkw : "Pred ln wk wg"

    df = df.eval("""
            rplnhrw = plnhrw + log(gdp)
            rplnwkw = plnwkw + log(gdp)
    """)

    # @ labels
    # @ rplnhrw "Real pred ln hr wg"
    # @ rplnwkw "Real pred ln wk wg"

    # @ expcat : 1 "5 years" 2 "15 years" 3 "25 years" 4 "35 years" 5 "45 years"
    # @ female : "1=F, 0=M"
    # @ edcat5 : "Education cats (5)"

    pred_marwg_6308 = df.filter(["year", "female", "edcat", "expcat", "rplnwkw", "rplnhrw",
                                "plnwkw", "plnhrw", "gdp", "avlswt", "normlswt", "lswt"])
    # @ data "March pred wgs 1963-2008: By Year/gender/ed/experience"
    return pred_marwg_6308


"""
calc-marchwg-byexp.do
"""


def calc_marchwg_byexp():
    """
    Calculate college/high school wage differentials by gender and experience group
    We use the regression estimated wage differentials by age/experience/year/gender
    We weight them together using fixed share weights averaged over all years 1963-2005
    So, this is akin to our preferred series for overall education differentials but
    now done by experience only

    D. Autor, 6/11/2004, 9/4/2006, 10/9/2006 (updated to March 2006 data)

    Updated for 2008/9 data - M. Wasserman 10/2009
    """

    df = assemb_marchwg_regs_exp()

    clghsgwg_march_regseries_exp = pd.DataFrame()

    for i in tqdm(["m", "f", "mf"]):
        ######################################################################
        # Calculate weighted wage series
        ######################################################################

        # Gender exclusions
        if i == "m":
            dt = df.query("female==0")
        elif i == "f":
            dt = df.query("female==1")
        else:
            dt = df.copy()

        # Overall high school
        v1 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==2)")))
        v2 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("avlswt*(edcat==2)")))
        hsgwg = (v1/v2).rename("hsgwg")
        dt = dt.merge(hsgwg, left_on="year", right_index=True)

        # Overall college-plus
        v1 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==4 | edcat==5)")))
        v2 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("avlswt*(edcat==4 | edcat==5)")))
        clpwg = (v1/v2).rename("clpwg")
        dt = dt.merge(clpwg, left_on="year", right_index=True)

        # Overall just college
        v1 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==4)")))
        v2 = dt.groupby(["year"]).apply(lambda x: sum(x.eval("avlswt*(edcat==4)")))
        clgwg = (v1/v2).rename("clgwg")
        dt = dt.merge(clgwg, left_on="year", right_index=True)

        # By experience high school
        v1 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==2)")))
        v2 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("avlswt*(edcat==2)")))
        exphsgwg = (v1/v2).rename("exphsgwg")
        dt = dt.merge(exphsgwg.reset_index(), on=["year", "expcat"])

        # By experience college-plusc
        v1 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==4 | edcat==5)")))
        v2 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("avlswt*(edcat==4 | edcat==5)")))
        expclpwg = (v1/v2).rename("expclpwg")
        dt = dt.merge(expclpwg.reset_index(), on=["year", "expcat"])

        # By experience just college
        v1 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("rplnwkw*avlswt*(edcat==4)")))
        v2 = dt.groupby(["year", "expcat"]).apply(lambda x: sum(x.eval("avlswt*(edcat==4)")))
        expclgwg = (v1/v2).rename("expclgwg")
        dt = dt.merge(expclgwg.reset_index(), on=["year", "expcat"])

        dt = dt.eval("""
                clphsg_all = clpwg - hsgwg
                clghsg_all = clgwg - hsgwg

                clphsg_exp = expclpwg - exphsgwg
                clghsg_exp = expclgwg - exphsgwg
        """)

        # @ label
        # @ clphsg_all : "COLLEGE-PLUS/HSG all"
        # @ clghsg_all : "COLLEGE-GRAD/HSG all"
        # @ clphsg_exp : "COLLEGE-PLUS/HSG by experience"
        # @ clghsg_exp : "COLLEGE-GRAD/HSG by experience"

        # Organize/save
        col = dt.columns[dt.columns.str.startswith(("year", "expcat", "clphsg_", "clghsg_"))]
        dt = dt.drop_duplicates(["year", "expcat"]).filter(col)

        if i != "mf":
            dt = dt.rename(columns=lambda x: x+"_"+i if x[:2] == "cl" else x)

        if i == "m":
            clghsgwg_march_regseries_exp = dt
        else:
            clghsgwg_march_regseries_exp = clghsgwg_march_regseries_exp.merge(dt)
        # data "March CLG/HSG wage series overall and by experience and gender using Handbook approach, average March wts 1963-2008"

    return clghsgwg_march_regseries_exp


def main():
    # tabulate_march_inequality(1997)
    # print(tabulate_march_inequality_loop())
    # print(predict_archwg_regs_exp_loop())
    # assemb_marchwg_regs_exp()
    calc_marchwg_byexp()


if __name__ == "__main__":
    main()
