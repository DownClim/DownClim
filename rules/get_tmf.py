rule get_tmf:
    input:
        "results/downscaled/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_future}_{period_eval}_{ds_method}.nc",
        "results/baselines/{area}_{base}_{aggregation}_{period_present}.nc",
        "results/tmf/rsds/{area}.nc",
        "results/areas/{area}.shp"
    output:
        "results/tmf/nc/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_future}_{period_present}_{period_eval}_{ds_method}.nc"
    log:
        "results/logs/tmf_{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_future}_{period_present}_{period_eval}_{ds_method}.log"
    benchmark:
        "results/benchmarks/tmf_{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_future}_{period_present}_{period_eval}_{ds_method}.benchmark.txt"
    threads: 1
    conda:
        "../envs/xarray.yml"
    script:
      "../scripts/get_tmf.py"
