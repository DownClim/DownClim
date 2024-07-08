rule get_chelsa2:
    input:
        expand("results/areas/{area}.shp", area=config["area"])
    output:
        expand("results/baselines/{area}_chelsa2_{aggregation}_{period}.nc", 
                area=config["area"], 
                period=base_periods,
                aggregation=config["aggregation"])
    log:
        "results/logs/get_chelsa2.log"
    benchmark:
        "results/benchmarks/get_chelsa2.benchmark.txt"
    threads: 6
    resources:
        mem_mb=10000
    conda:
        "../envs/xarray.yml"
    params:
        area=config["area"],
        variables=config["variables"],
        time_frequency=config["time_frequency"],
        base_years=config["base_years"],
        periods=base_periods,
        aggregation=config["aggregation"],
        tmp="results/baselines/chelsa2_tmp"
    script:
      "../scripts/get_chelsa2.py"
