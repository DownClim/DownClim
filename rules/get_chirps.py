rule get_chirps:
    input:
        expand("results/areas/{area}.shp", area=config["area"])
    output:
        expand("results/baselines/{area}_chirps_{aggregation}_{period}.nc", 
                area=config["area"], 
                period=base_periods,
                aggregation=config["aggregation"])
    log:
        "results/logs/get_chirps.log"
    benchmark:
        "results/benchmarks/get_chirps.benchmark.txt"
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
      "../scripts/get_chirps.py"
