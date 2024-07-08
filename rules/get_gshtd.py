rule get_gshtd:
    input:
        expand("results/areas/{area}.shp", area=config["area"])
    output:
        expand("results/baselines/{area}_gshtd_{aggregation}_{period}.nc", 
                area=config["area"], 
                period=base_periods,
                aggregation=config["aggregation"])
    log:
        "results/logs/get_gshtd.log"
    benchmark:
        "results/benchmarks/get_gshtd.benchmark.txt"
    threads: 6
    resources:
        mem_mb=10000
    conda:
        "../envs/xarray.yml"
    params:
        area=config["area"],
        variables=config["variables"],
        time_frequency=config["time_frequency"],
        periods=base_periods,
        aggregation=config["aggregation"]
    script:
      "../scripts/get_gshtd.py"
