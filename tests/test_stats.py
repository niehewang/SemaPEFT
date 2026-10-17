from semapeft.stats import process_summary

def test_process_level_summary_uses_process_medians():
    s=process_summary([1,2,3,4,100],resamples=500,seed=1)
    assert s['median']==3.0
    assert s['p25']==2.0 and s['p75']==4.0
