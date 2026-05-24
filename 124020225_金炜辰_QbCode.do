cd "C:\Users\ASUS\Desktop"
use "个股_clean.dta", clear
sort month lag_ret
by month: gen group = ceil(5 * _n / _N) if !missing(lag_ret)
collapse (mean) ret = Mretnd, by(month group)
reshape wide ret, i(month) j(group)
gen ret_LS = ret5 - ret1
merge 1:1 month using "因子_clean.dta", keep(match) nogen
save "mom_portfolios.dta", replace
foreach i in 1 2 3 4 5 {
    qui ttest ret`i' == 0
    di "Group `i'" _col(12) %9.4f r(mu_1) _col(27) %9.4f r(t)
}
qui ttest ret_LS == 0
di "G5-G1" _col(12) %9.4f r(mu_1) _col(27) %9.4f r(t)
collapse (mean) ret1-ret5
gen id = 1
reshape long ret, i(id) j(group)
drop id
twoway (scatter ret group, msize(medlarge)) (line ret group, lpattern(solid)), xtitle("Past Return Quintile (1=Losers, 5=Winners)") ytitle("Average Monthly Return") title("Momentum Effect") xlabel(1(1)5) ylabel(, format(%9.4f)) legend(off)
graph export "momentum_effect.png", replace
use "mom_portfolios.dta", clear
foreach i in 1 2 3 4 5 {
    qui reg ret`i' mktrf smb hml
    di "Group `i': alpha=" %9.4f _b[_cons] " t=" %9.4f _b[_cons]/_se[_cons] " beta_mkt=" %9.4f _b[mktrf] " t=" %9.4f _b[mktrf]/_se[mktrf] " beta_smb=" %9.4f _b[smb] " t=" %9.4f _b[smb]/_se[smb] " beta_hml=" %9.4f _b[hml] " t=" %9.4f _b[hml]/_se[hml]
}
qui reg ret_LS mktrf smb hml
di "G5-G1: alpha=" %9.4f _b[_cons] " t=" %9.4f _b[_cons]/_se[_cons] " beta_mkt=" %9.4f _b[mktrf] " t=" %9.4f _b[mktrf]/_se[mktrf] " beta_smb=" %9.4f _b[smb] " t=" %9.4f _b[smb]/_se[smb] " beta_hml=" %9.4f _b[hml] " t=" %9.4f _b[hml]/_se[hml]
