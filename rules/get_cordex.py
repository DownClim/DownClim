def list_areas(wildcards):
    return dom[dom.domain == wildcards.domain].area.values

def list_cordex_in(wildcards):
    return expand("results/baselines/{area}_{base}_{aggregation}_{period}.nc", 
                   area=dom[dom.domain == wildcards.domain].area, 
                   period=config["hist_years"],
                   aggregation=config["aggregation"],
                   allow_missing=True)
                   
rule get_cordex:
    input:
        list_cordex_in
    output:
        "results/projections/_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_done.txt"
    log:
        "results/logs/get_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}.log"
    benchmark:
        "results/benchmarks/get_cordex_CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}.benchmark.txt"
    threads: 10
    resources:
        mem_mb=10000
    conda:
        "../envs/xarray.yml"
    params:
        area=list_areas,
        domain="{domain}",
        institute="{institute}",
        model="{model}",
        experiment="{experiment}",
        ensemble="{ensemble}",
        rcm="{rcm}",
        downscaling="{downscaling}",
        base="{base}",
        variables=config["variables"],
        time_frequency=config["time_frequency"],
        esgf_credential=config["esgf_credential"],
        periods=all_periods,
        aggregation=config["aggregation"],
        tmp="results/projections/CORDEX_{domain}_{institute}_{model}_{experiment}_{ensemble}_{rcm}_{downscaling}_{base}_tmp/"
    script:
      "../scripts/get_cordex2.py"
