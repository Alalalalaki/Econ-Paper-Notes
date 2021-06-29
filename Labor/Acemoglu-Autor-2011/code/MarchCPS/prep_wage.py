"""
This file replicates the code in march-prep-wages.
The input is raw-data. The output is cleaned-data.

# means original comment
# @ means additional comment
"""

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW

input_path = "../../ref/origin/March-CPS/cleaned-data/"


"""
# Basic Wage Series
"""

"""
tab-march-ineq.do
"""


def tabulate_march_inequality(year, input_path=input_path):
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

    df = pd.read_stata(input_path+"mar"+str(year)[-2:]+".dta")

    # Only keep earnings sample ages 16-64
    df = df.query("selfemp == 0")
    assert df.loc[df.winc_ws.isna(), "hinc_ws"].isna().value_counts().all()
    assert df.loc[df.hinc_ws.isna(), "winc_ws"].isna().value_counts().all()
    df = df.eval("""
            lnwinc = log(winc_ws) + log(gdp)
            lnhinc = log(hinc_ws) + log(gdp)
        """)
    df = df.query("16 <= agely <=64")

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

    # Drop allocators
    df = df.query("allocated == 0")  # @ ?

    # Create consistent education categories
    if year <= 1991:
        # gen byte educomp= max(0,_grdhi-(grdcom==2))
        # tab educomp
        assert df.eval("0 <= educomp <= 18").all()
        df = df.eval("""
                ed8 = educomp<=8
                ed9  = educomp==9
                ed10 = educomp==10
                ed11 = educomp==11
                edhsg = educomp==12 & grdcom.cat.codes+1==1
                edsmc = (educomp>=13 & educomp<=15) | (educomp==12 & grdcom.cat.codes+1==2)
                edclg = educomp==16 | educomp==17
                edgtc = educomp>17
        """)
    else:
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

    # Drop if more than 48 years of experience to allow full range of experience for each education/age category (16, 3 year experience categories, last 45-48)
    df = df.loc[df.exp <= 48]

    # Generate experience categories
    df["expcat"] = (df.exp/3).astype(int) + 1
    df.loc[df.expcat == 17, "expcat"] = 16
    assert df.eval("1<= expcat <= 16").all()

    df.groupby("expcat")["exp"].agg(["mean", "min", "max"])

    # Generate education categories
    df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)
    df = df.eval("""
            edhsd= ed8 + ed9 + ed10 + ed11
            edcat5 = edhsd + 2*edhsg + 3*edsmc + 4*edclg + 5*edgtc
    """)
    df.edcat5.value_counts()
    assert df.eval("1 <= edcat5 <= 5").all()

    # interaction terms - 80 of these
    # @ generate interaction terms for edcat and expcat. not sure necessary. skip now

    # Drop reference group's interaction term: HSG with 0-2 years of experience
    # @ simiarly skip now
    df = df.filter(["year", "wgt", "wgt_hrs", "female", "lnwinc", "lnhinc", "hrsamp", "ftsamp"])  # @ ed1exp1-ed5exp16

    ###############################################################
    # Summarize raw inequality
    ###############################################################

    pctiles = pd.Series([3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97])
    pctiles_ = pctiles / 100
    df_pct = pd.DataFrame(index=pctiles)
    df_stat = pd.DataFrame(index=["mn", "vln"])

    dt = df.query("ftsamp==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    df_pct["tot_ft_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_ft_mf"] = [wq.mean, wq.var]

    dt = df.query("ftsamp==1 & female==0")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    df_pct["tot_ft_m"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_ft_m"] = [wq.mean, wq.var]

    dt = df.query("ftsamp==1 & female==1")
    wq = DescrStatsW(data=dt.lnwinc, weights=dt.wgt)
    df_pct["tot_ft_f"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_ft_f"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    df_pct["tot_hr_mf"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_hr_mf"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1 & female==0")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    df_pct["tot_hr_m"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_hr_m"] = [wq.mean, wq.var]

    dt = df.query("hrsamp==1 & female==1")
    wq = DescrStatsW(data=dt.lnhinc, weights=dt.wgt_hrs)
    df_pct["tot_hr_f"] = wq.quantile(probs=pctiles_, return_pandas=False)
    df_stat["tot_hr_f"] = [wq.mean, wq.var]

    df_stat = pd.concat([df_stat,df_pct], axis=0, sort=False)

    ###############################################################
    # Summarize residual inequality - Weekly
    ###############################################################

    

    return df


"""
tab-march-ineq-loop.do
"""


"""
# College High School Gap Series
# By potential experience, using regression
"""

"""
predict-marchwg-regs-exp.do
"""


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
    tabulate_march_inequality(1976)


if __name__ == "__main__":
    main()
