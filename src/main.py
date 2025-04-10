"""
# main.py
GTFT Simulation with Parameter Sweep and Heatmap Generation

This script simulates the Generous Tit-for-Tat strategy in an iterated
prisoner's dilemma scenario, with support for parameter sweeps, 
comprehensive logging, and visualization.

Author: Extended version based on original code
"""
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import itertools
import time
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import json
import traceback
import sys
import functools  # For properly implementing the timing decorator


#######################################################################
# Setup logging
#######################################################################
def init_log(results_path, log_level="INFO"):
    """Initialize logging with specified log level"""
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Initialize logging
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    # Add file handler
    os.makedirs(results_path, exist_ok=True)
    path = os.path.join(results_path, 'gtft_simulation.log')
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)

    logging.info("Logging initialized successfully.")
    logging.info(f"Results will be saved to: {results_path}")

    return logging.getLogger()


#######################################################################
# Time tracking decorator (fixed for pickle compatibility)
#######################################################################
# This is a top-level function that can be pickled, unlike a local function
def _timing_wrapper(func, *args, **kwargs):
    """Helper function for the timing decorator"""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    # If the result is already a tuple, we need to handle it properly
    if isinstance(result, tuple):
        return (*result, elapsed_time)  # Add elapsed time to the tuple
    else:
        return result, elapsed_time  # Wrap single result with elapsed time

# Top-level decorator function that returns a picklable function
def timing_decorator(func):
    """Decorator to measure and log execution time of functions"""
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
        return _timing_wrapper(func, *args, **kwargs)
    return wrapper


class GenerousTitForTatPlayer:
    """
    Implements a Generous Tit-for-Tat strategy for the Prisoner's Dilemma.
    
    With probability 'forgiveness_probability', the player will cooperate even
    if the opponent defected in the previous round. Additionally, the player
    can make errors with probability 'error_rate', which flips their intended move.
    """
    def __init__(self, forgiveness_probability=0.3, error_rate=0.05, id=None):
        """
        Initialize the GTFT player.
        
        Parameters:
        -----------
        forgiveness_probability : float
            Probability of cooperating despite opponent's defection (0.0-1.0)
        error_rate : float
            Probability of making an error (playing opposite of intended move) (0.0-1.0)
        id : str or int
            Optional identifier for the player
        """
        self.forgiveness_probability = forgiveness_probability
        self.error_rate = error_rate
        self.last_opponent_move = "C"  # Start assuming cooperation
        self.id = id
        self.moves_history = []
        self.opponent_moves_history = []

    def make_move(self):
        """
        Determine the next move based on GTFT strategy with possible errors.
        
        Returns:
        --------
        str : "C" for cooperation or "D" for defection
        """
        # Base decision on opponent's last move
        if self.last_opponent_move == "C":
            move = "C"  # Reciprocate cooperation
        else:
            # Opponent defected: decide whether to forgive
            move = "C" if random.random() < self.forgiveness_probability else "D"
        
        # Apply possible error (with probability error_rate)
        if random.random() < self.error_rate:
            move = "D" if move == "C" else "C"
        
        # Record this move
        self.moves_history.append(move)
        
        return move

    def observe_opponent_move(self, move):
        """
        Record the opponent's move for future decision making.
        
        Parameters:
        -----------
        move : str
            "C" for cooperation or "D" for defection
        """
        self.last_opponent_move = move
        self.opponent_moves_history.append(move)
    
    def get_cooperation_rate(self):
        """
        Calculate the player's cooperation rate over all rounds.
        
        Returns:
        --------
        float : Proportion of cooperation moves (0.0-1.0)
        """
        if not self.moves_history:
            return 0.0
        return sum(1 for move in self.moves_history if move == "C") / len(self.moves_history)


class GTFTGame:
    """
    Simulates an iterated Prisoner's Dilemma game between two GTFT players.
    
    Tracks game history, cooperation rates, and can output detailed logs.
    """
    def __init__(self, rounds, forgiveness_probability, error_rate, log_filename=None,
                 payoff_matrix=None, random_seed=None):
        """
        Initialize a new GTFT game simulation.
        
        Parameters:
        -----------
        rounds : int
            Number of rounds to play
        forgiveness_probability : float
            Probability of forgiving a defection
        error_rate : float
            Probability of making an error in move execution
        log_filename : str, optional
            Path to save detailed game logs
        payoff_matrix : dict, optional
            Custom payoff matrix for the game
        random_seed : int, optional
            Seed for random number generation
        """
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        self.rounds = rounds
        self.player1 = GenerousTitForTatPlayer(forgiveness_probability, error_rate, id=1)
        self.player2 = GenerousTitForTatPlayer(forgiveness_probability, error_rate, id=2)
        self.history = []
        self.log_filename = log_filename
        
        # Default Prisoner's Dilemma payoff matrix if none provided
        self.payoff_matrix = payoff_matrix or {
            ('C', 'C'): (3, 3),    # Both cooperate: both get 3
            ('C', 'D'): (0, 5),    # Player1 cooperates, Player2 defects: 0 and 5
            ('D', 'C'): (5, 0),    # Player1 defects, Player2 cooperates: 5 and 0
            ('D', 'D'): (1, 1)     # Both defect: both get 1
        }
        
        # Game statistics
        self.player1_score = 0
        self.player2_score = 0
        
        if log_filename:
            self.init_log()

    def init_log(self):
        """Initialize the game log file with header information."""
        try:
            os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
            with open(self.log_filename, "w") as f:
                f.write("GTFT Simulation Log\n")
                f.write(f"Date: {datetime.now()}\n")
                f.write(f"Rounds: {self.rounds}\n")
                f.write(f"Forgiveness Probability: {self.player1.forgiveness_probability}\n")
                f.write(f"Error Rate: {self.player1.error_rate}\n")
                f.write("Round\tPlayer1\tPlayer2\tP1_Score\tP2_Score\tCumulative_P1\tCumulative_P2\n")
        except Exception as e:
            logging.error(f"Failed to initialize log file: {e}")
            self.log_filename = None
    
    def play(self):
        """
        Execute the game for the specified number of rounds.
        
        Returns:
        --------
        float : Overall cooperation rate across both players
        elapsed_time : Time taken to complete the game
        """
        start_time = time.time()
        
        for i in range(self.rounds):
            move1 = self.player1.make_move()
            move2 = self.player2.make_move()
            
            # Record moves in history
            self.history.append((move1, move2))
            
            # Update players with opponent moves
            self.player1.observe_opponent_move(move2)
            self.player2.observe_opponent_move(move1)
            
            # Calculate scores for this round
            p1_score, p2_score = self.payoff_matrix[(move1, move2)]
            self.player1_score += p1_score
            self.player2_score += p2_score
            
            # Log this round if enabled
            if self.log_filename:
                try:
                    with open(self.log_filename, "a") as f:
                        f.write(f"{i+1}\t{move1}\t{move2}\t{p1_score}\t{p2_score}\t"
                                f"{self.player1_score}\t{self.player2_score}\n")
                except Exception as e:
                    logging.error(f"Failed to write to log file (round {i+1}): {e}")
        
        elapsed_time = time.time() - start_time
        return self.calculate_cooperation_rate(), elapsed_time

    def calculate_cooperation_rate(self):
        """
        Calculate the overall cooperation rate in the game.
        
        Returns:
        --------
        float : Proportion of cooperation moves across all rounds and both players
        """
        if not self.history:
            return 0.0
        
        total_coop = sum((p1 == "C") + (p2 == "C") for p1, p2 in self.history)
        return total_coop / (2 * len(self.history))
    
    def calculate_mutual_cooperation_rate(self):
        """
        Calculate the rate of mutual cooperation (both players cooperate).
        
        Returns:
        --------
        float : Proportion of rounds where both players cooperated
        """
        if not self.history:
            return 0.0
        
        mutual_coop = sum(1 for p1, p2 in self.history if p1 == "C" and p2 == "C")
        return mutual_coop / len(self.history)
    
    def calculate_mutual_defection_rate(self):
        """
        Calculate the rate of mutual defection (both players defect).
        
        Returns:
        --------
        float : Proportion of rounds where both players defected
        """
        if not self.history:
            return 0.0
        
        mutual_defect = sum(1 for p1, p2 in self.history if p1 == "D" and p2 == "D")
        return mutual_defect / len(self.history)
    
    def get_average_payoff(self):
        """
        Calculate the average payoff per round for each player.
        
        Returns:
        --------
        tuple : (player1_avg_payoff, player2_avg_payoff)
        """
        if not self.history:
            return (0.0, 0.0)
        
        return (self.player1_score / len(self.history), 
                self.player2_score / len(self.history))
    
    def get_summary_statistics(self):
        """
        Generate comprehensive summary statistics for the game.
        
        Returns:
        --------
        dict : Dictionary containing various game statistics
        """
        if not self.history:
            return {
                "rounds_played": 0,
                "cooperation_rate": 0.0
            }
        
        return {
            "rounds_played": len(self.history),
            "cooperation_rate": self.calculate_cooperation_rate(),
            "mutual_cooperation_rate": self.calculate_mutual_cooperation_rate(),
            "mutual_defection_rate": self.calculate_mutual_defection_rate(),
            "player1_cooperation_rate": self.player1.get_cooperation_rate(),
            "player2_cooperation_rate": self.player2.get_cooperation_rate(),
            "player1_total_score": self.player1_score,
            "player2_total_score": self.player2_score,
            "player1_avg_score": self.player1_score / len(self.history),
            "player2_avg_score": self.player2_score / len(self.history),
            "forgiveness_probability": self.player1.forgiveness_probability,
            "error_rate": self.player1.error_rate
        }
    
    def plot_cooperation_trend(self, save_path=None):
        """
        Plot the trend of cooperation rates over time.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot image
        
        Returns:
        --------
        str : Path to the saved plot, if save_path was provided
        """
        if not self.history:
            logging.warning("Cannot plot cooperation trend: no game history")
            return None
        
        plt.figure(figsize=(10, 6))
        
        # Calculate rolling cooperation rates
        window_size = min(10, len(self.history))
        if window_size < 2:
            window_size = 2
        
        p1_coop = [1 if move == "C" else 0 for move, _ in self.history]
        p2_coop = [1 if move == "C" else 0 for _, move in self.history]
        
        # Calculate rolling average using pandas
        p1_rolling = pd.Series(p1_coop).rolling(window=window_size).mean()
        p2_rolling = pd.Series(p2_coop).rolling(window=window_size).mean()
        
        # Plot rolling cooperation rates
        plt.plot(p1_rolling, label="Player 1", color="blue")
        plt.plot(p2_rolling, label="Player 2", color="red")
        
        # Add game details
        plt.title(f"Cooperation Trends Over Time\nForgiveness: {self.player1.forgiveness_probability}, Error Rate: {self.player1.error_rate}")
        plt.xlabel("Round")
        plt.ylabel(f"Cooperation Rate (Rolling Window: {window_size})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300)
                plt.close()
                return save_path
            except Exception as e:
                logging.error(f"Failed to save cooperation trend plot: {e}")
                plt.close()
                return None
        
        plt.show()
        plt.close()
        return None


#######################################################################
# Simulation functions
#######################################################################
def run_simulation(params):
    """
    Run a single GTFT simulation with the specified parameters.
    
    Parameters:
    -----------
    params : tuple
        (rounds, forgiveness, error, results_dir, run_id)
    
    Returns:
    --------
    dict : Simulation results
    """
    rounds, forgiveness, error, results_dir, run_id = params
    
    try:
        log_file = os.path.join(results_dir, f"log_{run_id}.txt")
        game = GTFTGame(rounds, forgiveness, error, log_file)
        coop_rate, elapsed_time = game.play()
        
        # Get detailed statistics
        stats = game.get_summary_statistics()
        
        # Generate cooperation trend plot
        plot_path = os.path.join(results_dir, f"trend_{run_id}.png")
        game.plot_cooperation_trend(save_path=plot_path)
        
        return {
            "RunID": run_id,
            "Rounds": rounds,
            "Forgiveness": forgiveness,
            "Error": error,
            "CoopRate": round(coop_rate, 4),
            "MutualCoopRate": round(stats["mutual_cooperation_rate"], 4),
            "MutualDefectRate": round(stats["mutual_defection_rate"], 4),
            "Player1Score": stats["player1_total_score"],
            "Player2Score": stats["player2_total_score"],
            "AvgPayoff": round((stats["player1_avg_score"] + stats["player2_avg_score"]) / 2, 4),
            "ExecutionTime": round(elapsed_time, 3),
            "LogFile": os.path.basename(log_file),
            "PlotFile": os.path.basename(plot_path)
        }
    except Exception as e:
        error_msg = f"Error in run_simulation: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return {
            "RunID": run_id,
            "Rounds": rounds,
            "Forgiveness": forgiveness,
            "Error": error,
            "CoopRate": float('nan'),
            "ExecutionTime": 0,
            "Error": error_msg
        }


def run_parameter_sweep(args, results_dir, timestamp, executor_type='thread'):
    """
    Run a parameter sweep across multiple combinations of forgiveness probability
    and error rate values.
    
    Parameters:
    -----------
    args : ArgumentParser namespace
        Command-line arguments
    results_dir : str
        Directory to save results
    timestamp : str
        Timestamp for this run
    executor_type : str
        Type of executor to use ('thread' or 'process')
    
    Returns:
    --------
    pd.DataFrame : Results of the parameter sweep
    """
    logging.info("Starting parameter sweep...")
    
    # Create parameter combinations
    param_combinations = list(itertools.product(args.rounds, args.forgiveness, args.error))
    logging.info(f"Will run {len(param_combinations)} parameter combinations")
    
    # Prepare task list
    tasks = []
    for r, f, e in param_combinations:
        # Create a unique run ID for each parameter combination
        run_id = f"r{r}_f{f}_e{e}_{timestamp}"
        tasks.append((r, f, e, results_dir, run_id))
    
    # Choose executor based on type
    results = []
    
    if executor_type == 'process':
        # For process-based parallelism, we need special handling
        logging.info("Using ProcessPoolExecutor for parallel execution")
        
        # Determine number of workers
        max_workers = min(os.cpu_count(), len(tasks))
        logging.info(f"Using {max_workers} worker processes")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(run_simulation, task) for task in tasks]
            
            # Monitor progress
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Running simulations")):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error in task {i}: {str(e)}")
                    # Create an error result to ensure we have data for all tasks
                    task = tasks[i]
                    results.append({
                        "RunID": f"r{task[0]}_f{task[1]}_e{task[2]}_{timestamp}",
                        "Rounds": task[0],
                        "Forgiveness": task[1],
                        "Error": task[2],
                        "CoopRate": float('nan'),
                        "ExecutionTime": 0,
                        "Error": f"Task failed: {str(e)}"
                    })
    else:
        # Thread-based parallelism is simpler
        logging.info("Using ThreadPoolExecutor for parallel execution")
        
        # Determine number of workers
        max_workers = min(os.cpu_count() * 2, len(tasks))  # More threads than cores is often beneficial
        logging.info(f"Using {max_workers} worker threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm for progress tracking
            for result in tqdm(
                executor.map(run_simulation, tasks), 
                total=len(tasks),
                desc="Running simulations"
            ):
                results.append(result)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Save raw results to CSV
    csv_path = os.path.join(results_dir, f"sweep_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved sweep results to {csv_path}")
    
    return df


#######################################################################
# Visualization functions
#######################################################################
def generate_heatmap(df, results_dir, metric='CoopRate', title=None):
    """
    Generate a heatmap visualization of the parameter sweep results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sweep results dataframe
    results_dir : str
        Directory to save the heatmap
    metric : str
        Column name to use for the heatmap values
    title : str, optional
        Custom title for the heatmap
    
    Returns:
    --------
    str : Path to the saved heatmap
    """
    if df.empty:
        logging.error("Cannot generate heatmap: empty dataframe")
        return None
    
    try:
        # Create pivot table for heatmap
        pivot = df.pivot_table(index="Forgiveness", columns="Error", values=metric, aggfunc='mean')
        
        # Set up plot
        plt.figure(figsize=(10, 8))
        
        # Dynamically determine color map based on metric
        if metric in ['CoopRate', 'MutualCoopRate']:
            cmap = "YlGnBu"  # Blue-green for cooperation metrics
        elif metric in ['MutualDefectRate']:
            cmap = "YlOrRd"  # Red for defection metrics
        elif metric in ['AvgPayoff', 'Player1Score', 'Player2Score']:
            cmap = "viridis"  # Viridis for payoff metrics
        else:
            cmap = "coolwarm"  # Default
        
        # Generate the heatmap
        ax = sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap,
            cbar_kws={'label': metric}
        )
        
        # Set title and labels
        if title is None:
            title = f"Average {metric} (Forgiveness vs Error)"
        plt.title(title)
        plt.xlabel("Error Rate")
        plt.ylabel("Forgiveness Probability")
        
        # Improve tick labels
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
        
        # Save the plot
        plt.tight_layout()
        heatmap_path = os.path.join(results_dir, f"{metric.lower()}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        logging.info(f"Heatmap for {metric} saved to {heatmap_path}")
        return heatmap_path
    
    except Exception as e:
        logging.error(f"Failed to generate heatmap for {metric}: {e}")
        plt.close()
        return None


def generate_lineplots(df, results_dir):
    """
    Generate line plots showing how metrics change across parameter values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sweep results dataframe
    results_dir : str
        Directory to save the plots
    
    Returns:
    --------
    list : Paths to the saved plots
    """
    if df.empty:
        logging.error("Cannot generate line plots: empty dataframe")
        return []
    
    plot_paths = []
    metrics = ['CoopRate', 'MutualCoopRate', 'MutualDefectRate', 'AvgPayoff']
    
    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric {metric} not found in results dataframe")
            continue
            
        try:
            # 1. Plot metric vs Forgiveness for different Error rates
            plt.figure(figsize=(10, 6))
            
            for error in sorted(df['Error'].unique()):
                subset = df[df['Error'] == error]
                grouped = subset.groupby('Forgiveness')[metric].mean().reset_index()
                grouped = grouped.sort_values('Forgiveness')
                
                plt.plot(grouped['Forgiveness'], grouped[metric], 
                         marker='o', label=f"Error Rate = {error}")
            
            plt.xlabel("Forgiveness Probability")
            plt.ylabel(metric)
            plt.title(f"Effect of Forgiveness on {metric} by Error Rate")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_path = os.path.join(results_dir, f"{metric.lower()}_vs_forgiveness.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plot_paths.append(plot_path)
            
            # 2. Plot metric vs Error for different Forgiveness values
            plt.figure(figsize=(10, 6))
            
            for forgiveness in sorted(df['Forgiveness'].unique()):
                subset = df[df['Forgiveness'] == forgiveness]
                grouped = subset.groupby('Error')[metric].mean().reset_index()
                grouped = grouped.sort_values('Error')
                
                plt.plot(grouped['Error'], grouped[metric], 
                         marker='o', label=f"Forgiveness = {forgiveness}")
            
            plt.xlabel("Error Rate")
            plt.ylabel(metric)
            plt.title(f"Effect of Error Rate on {metric} by Forgiveness")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plot_path = os.path.join(results_dir, f"{metric.lower()}_vs_error.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            plot_paths.append(plot_path)
            
        except Exception as e:
            logging.error(f"Failed to generate line plots for {metric}: {e}")
            plt.close()
    
    logging.info(f"Generated {len(plot_paths)} line plots")
    return plot_paths


def create_analysis_report(df, results_dir, timestamp):
    """
    Create a comprehensive analysis report based on the parameter sweep results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sweep results dataframe
    results_dir : str
        Directory to save the report
    timestamp : str
        Timestamp for this run
    
    Returns:
    --------
    str : Path to the saved report
    """
    if df.empty:
        logging.error("Cannot create analysis report: empty dataframe")
        return None
    
    report_path = os.path.join(results_dir, f"analysis_report_{timestamp}.txt")
    
    try:
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GTFT Simulation: Parameter Sweep Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report generated: {datetime.now()}\n\n")
            
            # Summary statistics
            f.write("-"*80 + "\n")
            f.write("Summary Statistics:\n")
            f.write("-"*80 + "\n")
            
            f.write(f"Total simulations: {len(df)}\n")
            f.write(f"Rounds tested: {sorted(df['Rounds'].unique())}\n")
            f.write(f"Forgiveness probabilities tested: {sorted(df['Forgiveness'].unique())}\n")
            f.write(f"Error rates tested: {sorted(df['Error'].unique())}\n\n")
            
            # Overall averages
            f.write("Overall averages across all parameter combinations:\n")
            for metric in ['CoopRate', 'MutualCoopRate', 'MutualDefectRate', 'AvgPayoff']:
                if metric in df.columns:
                    f.write(f"  Average {metric}: {df[metric].mean():.4f}\n")
            f.write("\n")
            
            # Effect of forgiveness
            f.write("-"*80 + "\n")
            f.write("Effect of Forgiveness Probability:\n")
            f.write("-"*80 + "\n")
            
            # Group by forgiveness and calculate averages
            forgiveness_effect = df.groupby('Forgiveness').agg({
                'CoopRate': 'mean',
                'MutualCoopRate': 'mean',
                'MutualDefectRate': 'mean',
                'AvgPayoff': 'mean'
            }).reset_index()
            
            # Sort by forgiveness probability
            forgiveness_effect = forgiveness_effect.sort_values('Forgiveness')
            
            # Calculate correlation
            corr_forgive_coop = df['Forgiveness'].corr(df['CoopRate'])
            corr_forgive_mutual = df['Forgiveness'].corr(df['MutualCoopRate'])
            corr_forgive_defect = df['Forgiveness'].corr(df['MutualDefectRate'])
            corr_forgive_payoff = df['Forgiveness'].corr(df['AvgPayoff'])
            
            f.write(f"Correlation between Forgiveness and Cooperation Rate: {corr_forgive_coop:.4f}\n")
            f.write(f"Correlation between Forgiveness and Mutual Cooperation: {corr_forgive_mutual:.4f}\n")
            f.write(f"Correlation between Forgiveness and Mutual Defection: {corr_forgive_defect:.4f}\n")
            f.write(f"Correlation between Forgiveness and Average Payoff: {corr_forgive_payoff:.4f}\n\n")
            
            f.write("Detailed effects by forgiveness probability value:\n")
            for _, row in forgiveness_effect.iterrows():
                f.write(f"  Forgiveness = {row['Forgiveness']:.2f}:\n")
                f.write(f"    Cooperation Rate: {row['CoopRate']:.4f}\n")
                f.write(f"    Mutual Cooperation: {row['MutualCoopRate']:.4f}\n")
                f.write(f"    Mutual Defection: {row['MutualDefectRate']:.4f}\n")
                f.write(f"    Average Payoff: {row['AvgPayoff']:.4f}\n\n")
            
            # Effect of error rate
            f.write("-"*80 + "\n")
            f.write("Effect of Error Rate:\n")
            f.write("-"*80 + "\n")
            
            # Group by error rate and calculate averages
            error_effect = df.groupby('Error').agg({
                'CoopRate': 'mean',
                'MutualCoopRate': 'mean',
                'MutualDefectRate': 'mean',
                'AvgPayoff': 'mean'
            }).reset_index()
            
            # Sort by error rate
            error_effect = error_effect.sort_values('Error')
            
            # Calculate correlation
            corr_error_coop = df['Error'].corr(df['CoopRate'])
            corr_error_mutual = df['Error'].corr(df['MutualCoopRate'])
            corr_error_defect = df['Error'].corr(df['MutualDefectRate'])
            corr_error_payoff = df['Error'].corr(df['AvgPayoff'])
            
            f.write(f"Correlation between Error Rate and Cooperation Rate: {corr_error_coop:.4f}\n")
            f.write(f"Correlation between Error Rate and Mutual Cooperation: {corr_error_mutual:.4f}\n")
            f.write(f"Correlation between Error Rate and Mutual Defection: {corr_error_defect:.4f}\n")
            f.write(f"Correlation between Error Rate and Average Payoff: {corr_error_payoff:.4f}\n\n")
            
            f.write("Detailed effects by error rate value:\n")
            for _, row in error_effect.iterrows():
                f.write(f"  Error Rate = {row['Error']:.2f}:\n")
                f.write(f"    Cooperation Rate: {row['CoopRate']:.4f}\n")
                f.write(f"    Mutual Cooperation: {row['MutualCoopRate']:.4f}\n")
                f.write(f"    Mutual Defection: {row['MutualDefectRate']:.4f}\n")
                f.write(f"    Average Payoff: {row['AvgPayoff']:.4f}\n\n")
            
            # Combined effects (heatmap-like representation in text)
            f.write("-"*80 + "\n")
            f.write("Combined Effects of Forgiveness and Error Rate on Cooperation:\n")
            f.write("-"*80 + "\n")
            
            # Create a pivot table
            pivot = df.pivot_table(index="Forgiveness", columns="Error", values="CoopRate", aggfunc='mean')
            
            # Print a text-based heatmap
            f.write("\nCooperation Rate by Forgiveness (rows) and Error (columns):\n\n")
            
            # Header row with error rates
            f.write("Forgiveness |")
            for error in pivot.columns:
                f.write(f" Error={error:.2f} |")
            f.write("\n")
            f.write("-"*(13 + 14*len(pivot.columns)) + "\n")
            
            # Data rows
            for forgiveness, row in pivot.iterrows():
                f.write(f"   {forgiveness:.2f}    |")
                for error in pivot.columns:
                    f.write(f"    {row[error]:.4f}   |")
                f.write("\n")
            f.write("\n")
            
            # Best and worst parameter combinations
            f.write("-"*80 + "\n")
            f.write("Best and Worst Parameter Combinations:\n")
            f.write("-"*80 + "\n")
            
            # For cooperation rate
            best_coop = df.loc[df['CoopRate'].idxmax()]
            worst_coop = df.loc[df['CoopRate'].idxmin()]
            
            f.write("For maximizing Cooperation Rate:\n")
            f.write(f"  Best parameters: Forgiveness={best_coop['Forgiveness']:.2f}, Error={best_coop['Error']:.2f}\n")
            f.write(f"  Resulting cooperation rate: {best_coop['CoopRate']:.4f}\n\n")
            
            f.write("For minimizing Cooperation Rate:\n")
            f.write(f"  Worst parameters: Forgiveness={worst_coop['Forgiveness']:.2f}, Error={worst_coop['Error']:.2f}\n")
            f.write(f"  Resulting cooperation rate: {worst_coop['CoopRate']:.4f}\n\n")
            
            # For average payoff
            best_payoff = df.loc[df['AvgPayoff'].idxmax()]
            worst_payoff = df.loc[df['AvgPayoff'].idxmin()]
            
            f.write("For maximizing Average Payoff:\n")
            f.write(f"  Best parameters: Forgiveness={best_payoff['Forgiveness']:.2f}, Error={best_payoff['Error']:.2f}\n")
            f.write(f"  Resulting average payoff: {best_payoff['AvgPayoff']:.4f}\n\n")
            
            f.write("For minimizing Average Payoff:\n")
            f.write(f"  Worst parameters: Forgiveness={worst_payoff['Forgiveness']:.2f}, Error={worst_payoff['Error']:.2f}\n")
            f.write(f"  Resulting average payoff: {worst_payoff['AvgPayoff']:.4f}\n\n")
            
            # Conclusions
            f.write("-"*80 + "\n")
            f.write("Conclusions and Insights:\n")
            f.write("-"*80 + "\n")
            
            # Determine overall trend for forgiveness
            if corr_forgive_coop > 0.3:
                forgive_trend = "Higher forgiveness strongly increases cooperation"
            elif corr_forgive_coop > 0.1:
                forgive_trend = "Higher forgiveness moderately increases cooperation"
            elif corr_forgive_coop < -0.3:
                forgive_trend = "Higher forgiveness strongly decreases cooperation"
            elif corr_forgive_coop < -0.1:
                forgive_trend = "Higher forgiveness moderately decreases cooperation"
            else:
                forgive_trend = "Forgiveness has minimal effect on cooperation"
                
            # Determine overall trend for error rate
            if corr_error_coop > 0.3:
                error_trend = "Higher error rates strongly increase cooperation"
            elif corr_error_coop > 0.1:
                error_trend = "Higher error rates moderately increase cooperation"
            elif corr_error_coop < -0.3:
                error_trend = "Higher error rates strongly decrease cooperation"
            elif corr_error_coop < -0.1:
                error_trend = "Higher error rates moderately decrease cooperation"
            else:
                error_trend = "Error rate has minimal effect on cooperation"
            
            f.write(f"1. {forgive_trend}.\n")
            f.write(f"2. {error_trend}.\n")
            
            # Optimal ranges
            optimal_forgiveness = forgiveness_effect.iloc[forgiveness_effect['AvgPayoff'].argmax()]['Forgiveness']
            optimal_error = error_effect.iloc[error_effect['AvgPayoff'].argmax()]['Error']
            
            f.write(f"3. For maximizing payoff, the optimal forgiveness probability appears to be around {optimal_forgiveness:.2f}.\n")
            f.write(f"4. For maximizing payoff, the optimal error rate appears to be around {optimal_error:.2f}.\n")
            
            # Robustness to errors
            high_error_effect = df[df['Error'] > df['Error'].median()]['CoopRate'].mean()
            low_error_effect = df[df['Error'] <= df['Error'].median()]['CoopRate'].mean()
            error_impact = high_error_effect - low_error_effect
            
            if abs(error_impact) < 0.05:
                f.write("5. GTFT appears to be quite robust to errors, with minimal impact on overall cooperation.\n")
            elif error_impact < 0:
                f.write(f"5. GTFT is somewhat sensitive to errors, with higher error rates decreasing cooperation by {abs(error_impact):.4f} on average.\n")
            else:
                f.write(f"5. Interestingly, higher error rates appear to increase cooperation by {error_impact:.4f} on average.\n")
            
            # Interaction effect
            f.write("6. ")
            high_forgive = df['Forgiveness'] > df['Forgiveness'].median()
            high_error = df['Error'] > df['Error'].median()
            
            coop_hf_he = df[high_forgive & high_error]['CoopRate'].mean()
            coop_hf_le = df[high_forgive & ~high_error]['CoopRate'].mean()
            coop_lf_he = df[~high_forgive & high_error]['CoopRate'].mean()
            coop_lf_le = df[~high_forgive & ~high_error]['CoopRate'].mean()
            
            interaction = (coop_hf_he - coop_lf_he) - (coop_hf_le - coop_lf_le)
            
            if abs(interaction) < 0.05:
                f.write("There appears to be minimal interaction between forgiveness and error rate.\n")
            elif interaction > 0:
                f.write("Higher forgiveness appears to be more effective at higher error rates.\n")
            else:
                f.write("Higher forgiveness appears to be more effective at lower error rates.\n")
            
            # Final insights
            f.write("\nFinal Insights:\n")
            f.write("The GTFT strategy demonstrates how a simple rule can promote cooperation in repeated\n")
            f.write("interactions. The impact of the forgiveness parameter shows the importance of occasional\n")
            f.write("forgiveness for maintaining cooperation, while the error rate analysis reveals how robust\n")
            f.write("this strategy is to noise in the environment.\n\n")
            
            f.write("="*80 + "\n")
        
        logging.info(f"Analysis report created at {report_path}")
        return report_path
    
    except Exception as e:
        logging.error(f"Failed to create analysis report: {e}")
        return None


#######################################################################
# Main function and argument parsing
#######################################################################
def parse_args():
    """
    Parse command-line arguments for the simulation.
    
    Returns:
    --------
    argparse.Namespace : Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="GTFT Parameter Sweep Simulator")
    parser.add_argument("--rounds", nargs="+", type=int, default=[100], help="Rounds (e.g. 50 100)")
    parser.add_argument("--forgiveness", nargs="+", type=float, default=[0.3], help="Forgiveness values (e.g. 0.1 0.3 0.5)")
    parser.add_argument("--error", nargs="+", type=float, default=[0.05], help="Error rates (e.g. 0.01 0.05 0.1)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save logs")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep across all combinations")
    parser.add_argument("--parallel", choices=['thread', 'process'], default='thread', help="Parallelization method")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--analysis_only", action="store_true", help="Run analysis on existing results only")
    parser.add_argument("--input_csv", type=str, help="Input CSV file for analysis_only mode")
    
    return parser.parse_args()


def main():
    """
    Main function to run the GTFT simulation.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logging.info(f"Random seed set to {args.seed}")
    
    # Create timestamp for unique directories/filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    if args.sweep:
        # For parameter sweeps, create a sweep-specific directory
        results_dir = os.path.join(args.results_dir, f"sweep_{timestamp}")
    else:
        # For single runs, use a simpler directory name
        results_dir = os.path.join(args.results_dir, f"gtft_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logging
    logger = init_log(results_dir, args.log_level)
    
    # Initialize df variable to avoid UnboundLocalError if an exception occurs
    df = pd.DataFrame()
    
    try:
        # Log start of execution and parameters
        logger.info("="*80)
        logger.info("Starting GTFT simulation")
        logger.info("="*80)
        logger.info(f"Parameters: rounds={args.rounds}, forgiveness={args.forgiveness}, error={args.error}")
        logger.info(f"Running mode: {'parameter sweep' if args.sweep else 'single parameter set'}")
        
        # Record system information
        import platform
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Operating system: {platform.platform()}")
        
        # Save arguments for reproducibility
        with open(os.path.join(results_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # If analysis only mode, load existing results
        if args.analysis_only:
            if not args.input_csv:
                logger.error("Input CSV file must be specified in analysis_only mode")
                return
            
            logger.info(f"Analysis only mode. Loading results from {args.input_csv}")
            
            try:
                df = pd.read_csv(args.input_csv)
                logger.info(f"Loaded {len(df)} results from CSV")
            except Exception as e:
                logger.error(f"Failed to load results from {args.input_csv}: {e}")
                return
        else:
            # Run simulation(s)
            if args.sweep:
                # Parameter sweep mode
                df = run_parameter_sweep(args, results_dir, timestamp, args.parallel)
            else:
                # Single parameter set mode
                r = args.rounds[0]
                f = args.forgiveness[0]
                e = args.error[0]
                
                logger.info(f"Running single simulation with r={r}, f={f}, e={e}")
                
                run_id = f"r{r}_f{f}_e{e}_{timestamp}"
                result = run_simulation((r, f, e, results_dir, run_id))
                
                # Create a DataFrame with the single result
                df = pd.DataFrame([result])
                
                # Print result
                logger.info(f"Simulation completed. Cooperation rate: {result['CoopRate']:.4f}")
                
                # Save result
                csv_path = os.path.join(results_dir, f"result_{timestamp}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Result saved to {csv_path}")
        
        # Generate visualizations and reports
        logger.info("Generating visualizations and analysis...")
        
        if args.sweep or args.analysis_only:
            # Generate heatmaps for different metrics
            for metric in ['CoopRate', 'MutualCoopRate', 'MutualDefectRate', 'AvgPayoff']:
                if metric in df.columns:
                    generate_heatmap(df, results_dir, metric)
            
            # Generate line plots
            generate_lineplots(df, results_dir)
            
            # Create analysis report
            report_path = create_analysis_report(df, results_dir, timestamp)
            if report_path:
                logger.info(f"Analysis report created at {report_path}")
        
        logger.info("Simulation and analysis completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Final log message
        logger.info("="*80)
        logger.info("GTFT simulation finished")
        logger.info("="*80)
        
        # Print summary to console
        print("\nGTFT Simulation Summary:")
        print("="*50)
        if args.sweep or args.analysis_only:
            print(f"Parameter sweep completed with {len(df)} combinations")
        else:
            r = args.rounds[0]
            f = args.forgiveness[0]
            e = args.error[0]
            print(f"Single simulation completed with r={r}, f={f}, e={e}")
            if not df.empty and 'CoopRate' in df.columns:
                print(f"Cooperation rate: {df['CoopRate'].iloc[0]:.4f}")
        print(f"Results saved to: {results_dir}")
        print("="*50)


if __name__ == "__main__":
    main()

