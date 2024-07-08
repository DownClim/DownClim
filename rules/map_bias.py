rule map_bias:
    input:
        "results/downscaled/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_proj}_{period_eval}_{ds_method}.nc",
        "results/baselines/{area}_{base_eval}_{aggregation}_{period_eval}.nc",
        "results/areas/{area}.shp"
    output:
        temp("results/bias/{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_proj}_{period_eval}_{ds_method}_{base_eval}.nc")
    log:
        "results/logs/bias_{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_proj}_{period_eval}_{ds_method}_{base_eval}.log"
    benchmark:
        "results/benchmarks/bias_{area}_{project}_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_{aggregation}_{period_proj}_{period_eval}_{ds_method}_{base_eval}.benchmark.txt"
    threads: 1
    conda:
        "../envs/xarray.yml"
    params:
      base="{base}",
      base_eval="{base_eval}"
    script:
      "../scripts/map_bias.py"
