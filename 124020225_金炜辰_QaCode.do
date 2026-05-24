cd"C:\Users\ASUS\Desktop"
use "月个股.dta", clear
gen month = monthly(Trdmnt, "YM")
format month %tm
gen str3 prefix = substr(Stkcd, 1, 3)
keep if inlist(prefix, "600", "601", "603", "605", "000", "001", "002", "003", "004")
drop prefix
keep Stkcd month Msmvosd Mretnd
sort Stkcd month
by Stkcd: gen lag_mv = Msmvosd[_n-1] if month == month[_n-1] + 1
by Stkcd: gen lag_ret = Mretnd[_n-1] if month == month[_n-1] + 1
keep if !missing(lag_mv) & !missing(lag_ret)
save "个股_clean.dta", replace
use "因子.dta", clear
gen month = monthly(TradingMonth, "YM")
format month %tm
rename RiskPremium1 mktrf
rename SMB1 smb
rename HML1 hml
keep month mktrf smb hml
save "因子_clean.dta", replace
use "个股_clean.dta", clear
sort month lag_mv
by month: gen group = ceil(10 * _n / _N) if !missing(lag_mv)
collapse (mean) ret = Mretnd, by(month group)

reshape wide ret, i(month) j(group)
gen ret_LS = ret10 - ret1
merge 1:1 month using "因子_clean.dta", keep(match) nogen
save "size_portfolios.dta", replace
di _n "Size Effect: Average Monthly Returns and t-statistics"
di "Portfolio    Mean Return    t-statistic         N"
di "-------------------------------------------------"
foreach i in 1 2 3 4 5 6 7 8 9 10 {
    qui ttest ret`i' == 0
    di "Group `i'" _col(12) %9.4f r(mu_1) _col(27) %9.4f r(t) _col(42) r(N)
}
qui ttest ret_LS == 0
di "G10-G1" _col(12) %9.4f r(mu_1) _col(27) %9.4f r(t) _col(42) r(N)
use "size_portfolios.dta", clear
collapse (mean) ret1-ret10
gen id = 1
reshape long ret, i(id) j(group)
drop id
twoway (scatter ret group, msize(medlarge)) (line ret group, lpattern(solid)), xtitle("Size Decile (1=Smallest, 10=Largest)") ytitle("Average Monthly Return") title("Size Effect") xlabel(1(1)10) ylabel(, format(%9.4f)) legend(off)
use "size_portfolios.dta", clear
foreach i in 1 2 3 4 5 6 7 8 9 10 {
    qui reg ret`i' mktrf
    di "Group `i': alpha=" %9.4f _b[_cons] " t=" %9.4f _b[_cons]/_se[_cons] " beta=" %9.4f _b[mktrf] " t=" %9.4f _b[mktrf]/_se[mktrf]
}
qui reg ret_LS mktrf
di "G10-G1: alpha=" %9.4f _b[_cons] " t=" %9.4f _b[_cons]/_se[_cons] " beta=" %9.4f _b[mktrf] " t=" %9.4f _b[mktrf]/_se[mktrf]
