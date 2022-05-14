var fs = require('fs');
fs.writeFile(
    "results.json",
    JSON.stringify(
        {
            "solvers": solvers,
            "fd_solvers": fd_solvers,
            "free_solvers": free_solvers,
            "par_solvers": par_solvers,
            "open_solvers": open_solvers,
            "problems": problems,
            "kind": kind,
            "instances": instances,
            "benchmarks": benchmarks,
            "results": results,
            "times": times,
            "objectives": objectives,
            "scores": scores,
            "step_obj": step_obj,
            "step_times": step_times
        },
        null, 2), () => {});
