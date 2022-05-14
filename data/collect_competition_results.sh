# collect mzn competition results and convert to .json file
# script adapted from https://github.com/informarte/minizinc-challenge-results

# from 2013 to 2016, there are no step_obj & step_times info
for year in `seq 2013 1 2016`
do
    echo "processing year $year"
    # get js result file and convert to json
    wget https://www.minizinc.org/challenge${year}/results.js
    cat collect_13_16.js >>results.js
    nodejs results.js
    rm results.js
    mv results.json results/${year}.json
done

for year in `seq 2017 1 2020`
do
    echo "processing year $year"
    # get js result file and convert to json
    wget https://www.minizinc.org/challenge${year}/results.js
    cat collect.js >>results.js
    nodejs results.js
    rm results.js
    mv results.json results/${year}.json
done

# 2021's format is different
for year in 2021
do
    echo "processing year $year"
    wget https://www.minizinc.org/challenge${year}/results.js
    head -n -5 results.js >temp.js
    mv temp.js results.js
    cat collect_2021.js >>results.js
    nodejs results.js
    rm results.js
    mv results.json results/${year}.json
done
