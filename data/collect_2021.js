var fs = require('fs');
var results = json['results']
fs.writeFile(
    "results.json",
    JSON.stringify(
        {
            "solvers": results["solvers"],
            "fd_solvers": results["fd_solvers"],
            "free_solvers": results["free_solvers"],
            "par_solvers": results["par_solvers"],
            "open_solvers": results["open_solvers"],
            "problems": results["problems"],
            "kind": results["kind"],
            "instances": results["instances"],
            "benchmarks": results["benchmarks"],
            "results": results["results"],
            "times": results["times"],
            "objectives": results["objectives"],
            "scores": results["scores"],
            "step_obj": results["step_obj"],            
            "step_times": results["step_times"],
        },
        null, 2), () => {});
