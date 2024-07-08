rule get_area:
    output:
        expand("results/areas/{area}.{ext}",
        ext=["shp", "cpg", "dbf", "shx"], allow_missing=True)
    log:
        "results/logs/get_area_{area}.log"
    benchmark:
        "results/benchmarks/get_area_{area}.benchmark.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda:
        "../envs/gadm.yml"
    params:
        area="{area}"
    script:
      "../scripts/get_area.py"
