"""
This file replicates the code in march-prep-supply.
The input is raw-data. The output is cleaned-data.

# means original comment
#@ means additional comment
"""

import numpy as np
import pandas as pd

from tqdm import tqdm

from prep_wage import tabulate_march_basic


input_path = "../../ref/origin/March-CPS/cleaned-data/"


"""
prepmarchcell.do
"""


def prepmarchcell(year):
    """
    @
    original data cleaning code same as prep_wage thus we directly use tabulate_march_basic
    Note
        - although here no drop exp > 48, given age limit this meets
        - original code selfemp later, we drop at first in tabulate_march_basic
          (can add back by set if in tabulate_march_basic if necessary)
        - both winc and hinc original code drop !ftfy while in prep_wage only prep_wage
    As a result, our results have less sample than the original, but otherwise the results are exactly the same
    @
    """

    df = tabulate_march_basic(year)
    df = df.rename(columns={"edcat": "edcat5"})

    # @ edcat8 label : 1 "0-8" 2 "9" 3 "10" 4 "11" 5 "Hsg" 6 "Smc" 7 "Clg" 8 "Gtc"
    df = df.eval("edcat8 = ed8 + 2*ed9 + 3*ed10 + 4*ed11 + 5*edhsg + 6*edsmc + 7*edclg + 8*edgtc")
    assert df.eval("1<= edcat8 <=8").all()
    # @ original code generate dummy columns for edcat5, here skip

    df = df.assign(exp=df.exp.round())

    df = df.eval("ftfy = fulltime*fullyear")

    # Replace earnings and wages as missing if not FTFY or if top/bottom coded.    Note that we don't really have to set these to missing as long as we assign them 0 weights for the p_weights/counts, but we'll just set to missing anyway (this won't affect the quantity collapses because earnings are not collapsed using quantity weights);
    # df.loc[df.eval("ftfy==0 | bcwkwgkm==1"), "winc_ws"] = np.nan
    # df.loc[df.eval("ftfy==0 | bchrwgkm==1"), "hinc_ws"] = np.nan
    # @ skip this as missing would make calculation vulnerable

    # Generate some earnings variables
    df = df.eval("rwinc = winc_ws*gdp")  # @ "Real earnings (2008$)"
    df = df.eval("lnwinc= log(winc_ws)")  # @ "Log of earnings"
    df = df.eval("rlnwinc = lnwinc + log(gdp)")  # @ "Log of real earnings (2008$)"
    df = df.eval("rhinc = hinc_ws*gdp")  # @ "Real hourly wage (2008$)"
    df = df.eval("lnhinc=log(hinc_ws)")  # @ "Log of hourly wage"
    df = df.eval("rlnhinc = lnhinc + log(gdp)")  # @ "Log of real hourly wages (2008$)"

    # Generate some count variables
    if df._wkslyr.dtype.name == "category":
        df = df.assign(_wkslyr=df._wkslyr.astype(int))
    if df.hrslyr.dtype.name == "category":
        df = df.assign(hrslyr=df.hrslyr.astype(int))
    df = df.eval("q_obs = 1")  # "Number of observations"
    df = df.eval("q_weight = wgt")  # "Earnings weight"
    df = df.eval("q_lsweight = wgt*_wkslyr")  # Earnings weight times weeks last year"
    df = df.eval("q_lshrsweight = wgt*_wkslyr*hrslyr;")  # "Earnings weight times weeks last year times hours last year"

    # Generate some price variables

    # Generate some edu by experience interactions (for predicting allocation status, if we ever choose to do that - currently we are not in March);
    # @ skip as can use exp:C(edcat5) expsq:C(edcat5)
    df = df.assign(expsq=df.exp**2)

    # What do we want to do with allocators?
    # We will try two things for March - keep them, and throw them out

    # Note for code below that we have already taken care of top/bottom coding and FTFY status by replacing winc_ws with missing if top/bottom coded or if not FTFY

    # Drop the allocators
    # @ Note rather than replace winc_ws and hinc_ws to missing at above,
    # @ I assign weights to 0 here for ftfy==0 | bcwkwgkm==1 | bchrwgkm==1
    # @ "Number of FTFY obs with non-missing earnings" // (drop allcators & selfemp)
    df = df.eval("p_obs = 1")
    df.loc[df.eval("ftfy==0 | bcwkwgkm==1 | bchrwgkm==1 | allocated==1 | selfemp ==1"), "p_obs"] = 0
    # @ "Earnings weight for FTFY obs with non-missing earnings"
    df = df.eval("p_weight = wgt")
    df.loc[df.eval("ftfy==0 | bcwkwgkm==1 | bchrwgkm==1 | allocated==1 | selfemp ==1"), "p_weight"] = 0
    # @ "Earnings weight times weeks last year for FTFY obs with non-missing earnings"
    df = df.eval("p_lsweight = wgt*_wkslyr")
    df.loc[df.eval("ftfy==0 | bcwkwgkm==1 | bchrwgkm==1 | allocated==1 | selfemp ==1 "), "p_lsweight"] = 0

    # @ save precollapsemarch`1'.dta

    # Collapse the dataset down to year by edu by experience by gender cells;
    cols = ["q_obs", "q_weight", "q_lsweight", "q_lshrsweight", "p_obs", "p_weight", "p_lsweight"]
    marchcells1 = df.groupby(["year", "edcat5", "exp", "female"])[cols].sum()  # marchcells1`1'.dta

    cols = ["rwinc", "rlnwinc", "rhinc", "rlnhinc"] + ["p_weight"]
    marchcells = df.groupby(["year", "edcat5", "exp", "female"])[cols].apply(
        lambda x: pd.Series(
            np.ma.average(x[cols[:-1]], axis=0, weights=x.p_weight)
        )).set_axis(cols[:-1], axis=1)  # marchcells1`1'.dta

    marchcells = pd.concat([marchcells, marchcells1], axis=1)

    return marchcells


"""
assembcellsmarch.do
"""


def assembcellsmarch():
    """
    #
    Assemble March cell data

    Original: Anderson, 10/28/03

    Updated 9/4/2006 to use March 2005 data by D. Autor
    Updated 10/14/2009 to use March 2007/8 data by M. Wasserman
    #
    """

    marchcells6308 = pd.DataFrame()
    for y in tqdm(range(1964, 2010)):
        marchcells = prepmarchcell(year=y)
        marchcells6308 = pd.concat([marchcells6308, marchcells], axis=0)

    # @  edcat5 : "Education cats (Five: HSD, HSG, SMC, CLG, GTC)"
    # @ edcat5 : 1 "Hsd" 2 "Hsg" 3 "Smc" 4 "Clg" 5 "Gtc"
    # @ rlnwinc :  "Mean of log real weekly FT earnings, 2008$ (collapse weight is p_weight)"
    # @ rlnhinc :  "Mean of log real FT wage, 2008$ (collapse weight is p_weight)"

    # @ /*exp :  "Potential years of experience (max(age-educomp-6,0))"*/
    # @ exp :  "Potential years of experience ( max(min(age-educomp-7,age-17),0) )"
    # @ q_obs :  "Total number of observations"
    # @ q_weight :  "Sum of (CPS weights*weeks last year) for all observations"
    # @ q_lsweight :  "Sum of (CPS weights*weeks last year*hours last year) for all observations"
    # @ p_obs :  "Number of FT obs (FT flag) with non-missing earnings, not self employed (may be reweighted for allocation issues)"
    # @ p_weight :  "Sum of (CPS weights*weeks last year) for FT obs (FT flag) with non-missing earnings, not self employed (may be reweighted for allocation issues)"
    # @ p_lsweight :  "Sum of (CPS weights*weeks last year*hours last wk) for FT obs (FT flag) with non-missing earnings, not self employed (may be reweighted for allocation issues)"
    # @ _merge :  "Merge will be equal to 2 when there is a cell for which there were only obs WITHOUT earnings/wage data"

    return marchcells6308


"""
effunit-supplies-exp-byexp.do
"""


def effunit_supplies_exp_byexp():
    """
    #
    /* Updated 10/10/2006 by D. Autor for March 2006 data */
    /* Updated 10/2009 by M. Wasserman for March 2007/08 data*/
    #
    #@
    Due to the different sample used in prepmarchcell comparing with the original code, the results are
    different. In particular
      - for Overall Eff units supply share (@?), the gap is quite small (as using wage and q_* data)
      - for total hours, the gap is somehow large (as using p_* data where original code include all sample)
    #@
    """
    df = assembcellsmarch()

    df = df.reset_index()
    df = df.rename(columns={"edcat5": "edcat", "exp": "exp1"})

    # @ "10 yr experience groups (0-9,...,30-39,40-48)"
    # expcat 1 "0-9" 2 "10-19" 3 "20-29" 4 "30-39" 5 "40-48"
    df = df.assign(
        expcat=pd.cut(df.exp1, bins=range(0, 51, 10),
                      right=False, labels=range(1, 6)))

    # Calculate average relative wage by cell over time period
    # Efficiency units translation: Base is HSD Male with 10 years of potential experience in each year
    refwage = df.groupby("year").apply(lambda x: x.query(
        "female==0 & edcat==2 & exp1==10").rwinc.max()).rename("refwage")
    df = df.merge(refwage, left_on="year", right_index=True)
    df = df.eval("relwage = rwinc/refwage")
    # @ for each ed x sex x exp cohort mean relative wage across years
    celleu = df.groupby(["edcat", "female", "exp1"])["relwage"].mean().rename("celleu")
    df = df.merge(celleu.reset_index(), on=["edcat", "female", "exp1"])

    # Overall (not by experience group)
    # @ yearly sum of (time-invariant cohort relative wage * time-variant quantity weight)
    tot_euwt = df.groupby(["year"]).apply(lambda x: sum(x.celleu * x.q_lshrsweight)).rename("tot_euwt")
    df = df.merge(tot_euwt, left_on="year", right_index=True)
    tot_euwt_m = df.groupby(["year"]).apply(
        lambda x: sum(x.celleu * x.q_lshrsweight * (x.female == 0))).rename("tot_euwt_m")
    df = df.merge(tot_euwt_m, left_on="year", right_index=True)
    tot_euwt_f = df.groupby(["year"]).apply(
        lambda x: sum(x.celleu * x.q_lshrsweight * x.female)).rename("tot_euwt_f")
    df = df.merge(tot_euwt_f, left_on="year", right_index=True)
    # @ yearly share of (time-invariant cohort relative wage * time-variant quantity weight)
    df = df.eval("""
                    sh_euwt=celleu*q_lshrsweight/tot_euwt
                    sh_euwt_m=celleu*q_lshrsweight/tot_euwt_m*(female == 0)
                    sh_euwt_f=celleu*q_lshrsweight/tot_euwt_f*female
    """)
    # @ yearly sum of (yearly share of weighted cohort relative wage * high edu dummy)
    df = df.eval("ed_dummy_h = (edcat==4 | edcat==5) + 0.5*(edcat==3)")
    eu_shclg = df.groupby("year").apply(lambda x: sum(x.sh_euwt*x.ed_dummy_h)).rename("eu_shclg")
    df = df.merge(eu_shclg, left_on="year", right_index=True)
    eu_shclg_m = df.groupby("year").apply(lambda x: sum(
        x.sh_euwt_m*x.ed_dummy_h * (x.female == 0))).rename("eu_shclg_m")
    df = df.merge(eu_shclg_m, left_on="year", right_index=True)
    eu_shclg_f = df.groupby("year").apply(
        lambda x: sum(x.sh_euwt_f*x.ed_dummy_h * x.female)).rename("eu_shclg_f")
    df = df.merge(eu_shclg_f, left_on="year", right_index=True)
    # @
    df = df.eval("""
                    eu_lnclg=log(eu_shclg/(1-eu_shclg))
                    eu_lnclg_m=log(eu_shclg_m/(1-eu_shclg_m))
                    eu_lnclg_f=log(eu_shclg_f/(1-eu_shclg_f))
    """)
    df = df.drop(df.columns[df.columns.str.startswith(("tot_", "sh_"))], axis=1)

    # @ label
    # @ eu_shclg : "Eff units CLP supply share: All"
    # @ eu_shclg_m : "Eff units CLP supply share: M"
    # @ eu_shclg_f : "Eff units CLP supply share: F"
    # @ eu_lnclg : "Eff units Ln CLG/NON-CLG supply: All"
    # @ eu_lnclg_m : "Eff units Ln CLG/NON-CLG supply: M"
    # @ eu_lnclg_f : "Eff units Ln CLG/NON-CLG supply: F"
    # @ year : "MORG year" # @ earning year

    # Also tabulate total hours by CLG and HSG equivalents
    # Chad Jones asked for this on 9/29/2004 - might as well keep the info
    hr_clg = df.groupby("year").apply(lambda x: sum(x.q_lshrsweight*x.ed_dummy_h)).rename("hr_clg")
    df = df.merge(hr_clg, left_on="year", right_index=True)
    hr_clg_m = df.groupby("year").apply(lambda x: sum(
        x.q_lshrsweight*x.ed_dummy_h * (x.female == 0))).rename("hr_clg_m")
    df = df.merge(hr_clg_m, left_on="year", right_index=True)
    hr_clg_f = df.groupby("year").apply(lambda x: sum(x.q_lshrsweight*x.ed_dummy_h * x.female)).rename("hr_clg_f")
    df = df.merge(hr_clg_f, left_on="year", right_index=True)

    df = df.eval("ed_dummy_l = (edcat==1 | edcat==2) + 0.5*(edcat==3)")
    hr_hsg = df.groupby("year").apply(lambda x: sum(x.q_lshrsweight*x.ed_dummy_l)).rename("hr_hsg")
    df = df.merge(hr_hsg, left_on="year", right_index=True)
    hr_hsg_m = df.groupby("year").apply(lambda x: sum(
        x.q_lshrsweight*x.ed_dummy_l * (x.female == 0))).rename("hr_hsg_m")
    df = df.merge(hr_hsg_m, left_on="year", right_index=True)
    hr_hsg_f = df.groupby("year").apply(lambda x: sum(x.q_lshrsweight*x.ed_dummy_l * x.female)).rename("hr_hsg_f")
    df = df.merge(hr_hsg_f, left_on="year", right_index=True)

    # @ label
    # @ hr_clg "Hours labor supply CLP: All"
    # @ hr_clg_m "Hours labor supply CLP: M"
    # @ hr_clg_f "Hours labor supply CLP: F"
    # @ hr_hsg "Hours labor supply HSG: All"
    # @ hr_hsg_m "Hours labor supply HSG: M"
    # @ hr_hsg_f "Hours labor supply HSG: F"

    # By experience group
    # @ exactly the same code as above except use ["year", "expcat"] and change "eu" to "euexp"
    tot_euwt = df.groupby(["year", "expcat"]).apply(lambda x: sum(x.celleu * x.q_lshrsweight)).rename("tot_euwt")
    df = df.merge(tot_euwt, left_on=["year", "expcat"], right_index=True)
    tot_euwt_m = df.groupby(["year", "expcat"]).apply(
        lambda x: sum(x.celleu * x.q_lshrsweight * (x.female == 0))).rename("tot_euwt_m")
    df = df.merge(tot_euwt_m, left_on=["year", "expcat"], right_index=True)
    tot_euwt_f = df.groupby(["year", "expcat"]).apply(
        lambda x: sum(x.celleu * x.q_lshrsweight * x.female)).rename("tot_euwt_f")
    df = df.merge(tot_euwt_f, left_on=["year", "expcat"], right_index=True)

    df = df.eval("""
                    sh_euwt=celleu*q_lshrsweight/tot_euwt
                    sh_euwt_m=celleu*q_lshrsweight/tot_euwt_m*(female == 0)
                    sh_euwt_f=celleu*q_lshrsweight/tot_euwt_f*female
    """)

    euexp_shclg = df.groupby(["year", "expcat"]).apply(lambda x: sum(x.sh_euwt*x.ed_dummy_h)).rename("euexp_shclg")
    df = df.merge(euexp_shclg, left_on=["year", "expcat"], right_index=True)
    euexp_shclg_m = df.groupby(["year", "expcat"]).apply(lambda x: sum(
        x.sh_euwt_m*x.ed_dummy_h * (x.female == 0))).rename("euexp_shclg_m")
    df = df.merge(euexp_shclg_m, left_on=["year", "expcat"], right_index=True)
    euexp_shclg_f = df.groupby(["year", "expcat"]).apply(
        lambda x: sum(x.sh_euwt_f*x.ed_dummy_h * x.female)).rename("euexp_shclg_f")
    df = df.merge(euexp_shclg_f, left_on=["year", "expcat"], right_index=True)
    # @
    df = df.eval("""
                    euexp_lnclg=log(euexp_shclg/(1-euexp_shclg))
                    euexp_lnclg_m=log(euexp_shclg_m/(1-euexp_shclg_m))
                    euexp_lnclg_f=log(euexp_shclg_f/(1-euexp_shclg_f))
    """)

    # @ label
    # @ euexp_shclg : "Eff by exper units CLP supply share: All"
    # @ euexp_shclg_m : "Eff by exper units CLP supply share: M"
    # @ euexp_shclg_f : "Eff by exper units CLP supply share: F"
    # @ euexp_lnclg : "Eff by exper units Ln CLG/NON-CLG supply: All"
    # @ euexp_lnclg_m : "Eff by exper units Ln CLG/NON-CLG supply: M"
    # @ euexp_lnclg_f : "Eff by exper units Ln CLG/NON-CLG supply: F"
    # @ year : "MORG year" # @ earning year

    # @ save/output
    cols = df.columns[df.columns.str.startswith(("year", "expcat", "eu_", "euexp_", "hr_"))]
    effunits_exp_by_exp_6308 = df[cols].drop_duplicates()  # @ effunits-exp-byexp-6308

    return effunits_exp_by_exp_6308


def main():
    effunit_supplies_exp_byexp()


if __name__ == "__main__":
    main()
