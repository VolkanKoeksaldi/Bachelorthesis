The source code and detailed execution instructions are available in the GitHub repository for the evaluation:
\href{https://github.com/VolkanKoeksaldi/Bachelorthesis/tree/1a762826745f2f88e68a0d35d8cd0724ee700415/prototype}{GitHub repository}

Start of prototype:
Open Windows Powershell under prototype folder.
Afterwards start docker build:
\texttt{docker build -t bachelor-prototype .}
Then create Output-Volume:
For example for size x1:
\texttt{docker volume create bachelor-x1-output}


Then start programm in docker by using for example (x1 is changeable, depending on the dataset scale chosen):
docker run --name bachelor-x1 -it `
  --mount "type=bind,source=$($PWD.Path)\src,target=/app/src" `
  --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" `
  --mount "type=volume,source=bachelor-x1-output,target=/app/output" `
  bachelor-prototype

Afterwards start the different scripts by using the commant \texttt{python src/} followed by the name of the script.
After the pipeline is finished with file 19_recovery.py use the following codes in order to store the results:
exit

and then depending on the chosen scaling to copy the output from the generated docker use (x1 is changeable):
docker cp bachelor-x1:/app/output/. .\docker_results\x1\

Afterwards the output is stored locally and the container can be deleted using:
docker rm bachelor-x1

Furthermore the volume from the docker can also be optionally deleted using:
docker volume rm bachelor-x1-output

Now for the execution order the script must be run in following order for correct results:
1. Configure the run in `experiment\_config.py`.
2. Run files 1-4 for MeSH and files 5-7 for IMDb. Beware that file 4 and 7 have the options baseline turned on under "MODE"
3. Run file 8 once for each dataset.
4. Run file 13 and then file 16 once for each dataset to create workload affinities.
5. Run file 9 and 10 in `baseline` mode.
6. Run file 11 in `baseline` mode for every available placement.
7. Run file 12 for every available placement.
8. Run file 14 for every available placement and file 15 once per dataset.
9. Run file 18 with `MODE = "prepare"`.
10. Recompute update overlaps with file 4 or 7 using `MODE = "updates"`.
11. Run file 9 and 10 with `MODE = "updates"`
12. Run file 11 with `MODE = "updates"` for Round Robin and all ILP placements.
13. Run file 18 with `MODE = "evaluate"`
14. Run file 17 for each dataset
15. Run file 19 for each dataset in baseline and updates mode.