(sexy_yeast_env) 15:42:07 💜 barc@cn472:~ > python '/home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/src/main.py' --help


usage: main.py [-h] [--rounds ROUNDS [ROUNDS ...]] [--forgiveness FORGIVENESS [FORGIVENESS ...]]
               [--error ERROR [ERROR ...]] [--results_dir RESULTS_DIR] [--sweep]
               [--parallel {thread,process}] [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--seed SEED] [--analysis_only] [--input_csv INPUT_CSV]

GTFT Parameter Sweep Simulator

options:
  -h, --help            show this help message and exit
  --rounds ROUNDS [ROUNDS ...]
                        Rounds (e.g. 50 100)
  --forgiveness FORGIVENESS [FORGIVENESS ...]
                        Forgiveness values (e.g. 0.1 0.3 0.5)
  --error ERROR [ERROR ...]
                        Error rates (e.g. 0.01 0.05 0.1)
  --results_dir RESULTS_DIR
                        Directory to save logs
  --sweep               Run parameter sweep across all combinations
  --parallel {thread,process}
                        Parallelization method
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Log level
  --seed SEED           Random seed for reproducibility
  --analysis_only       Run analysis on existing results only
  --input_csv INPUT_CSV
                        Input CSV file for analysis_only mode




#For a single simulation:

bsub -q short -R rusage[mem=12GB] -J Assignment_3_Evolution_of_Cooperation python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/src/main.py --rounds 10000 --forgiveness 0.3 --error 0.05 --results_dir  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/results
bsub -q short -R rusage[mem=12GB] -J Assignment_3_Evolution_of_Cooperation python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/src/main.py --rounds 10000 --forgiveness 0 --error 0.05 --results_dir  /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/results

# For a parameter sweep:
bsub -q short -R rusage[mem=12GB] -J Assignment_3_Evolution_of_Cooperation python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/src/main.py --sweep --rounds 10 50 100 1000 10000 --forgiveness  0 0.01 0.1 0.3 0.5 0.9 --error  0 0.01 0.05 0.1 0.2 0.5 0.9  --results_dir /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_3_Evolution_of_Cooperation/results --parallel process

