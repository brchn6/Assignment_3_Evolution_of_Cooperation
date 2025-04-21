"""
Adaptive Generous Tit-for-Tat Strategy for Prisoner's Dilemma

Strategy Explanation:
My strategy is a dynamic variant of Generous Tit-for-Tat (GTFT) that adapts its forgiveness
level based on the opponent's behavior. The classic GTFT occasionally forgives defections 
at a fixed rate, but my implementation adjusts this rate based on the opponent's cooperation 
tendency.

When facing a generally cooperative opponent, the strategy becomes more forgiving to 
establish mutual cooperation. Against aggressive opponents who defect frequently, the 
strategy reduces forgiveness to avoid exploitation while still allowing for recovery if 
the opponent shifts to cooperation.

The strategy includes pattern recognition to detect and counter common opponent strategies:
1. Against Always-Defect: It mostly defects but occasionally tests for a change in behavior
2. Against Tit-for-Tat: It recovers quickly from defection cycles
3. Against random players: It stabilizes toward cautious cooperation
4. Against more complex strategies: It adapts forgiveness levels based on recent history

I expect this strategy to perform well across different opponents because it balances 
exploitation resistance with cooperative potential, and its adaptive nature allows it 
to calibrate its approach based on the specific opponent it faces.

Author: BC
"""

def strategy(my_history, opponent_history):
    """
    Adaptive Generous Tit-for-Tat strategy for the Prisoner's Dilemma.
    
    Args:
        my_history (list): List of your past moves ('cooperate' or 'defect')
        opponent_history (list): List of opponent's past moves ('cooperate' or 'defect')
    
    Returns:
        str: 'cooperate' or 'defect'
    """
    # First move: cooperate to start positively
    if not opponent_history:
        return 'cooperate'
    
    # Calculate opponent's cooperation rate (with recency bias)
    recent_window = min(len(opponent_history), 10)  # Look at last 10 moves max
    recent_coop_rate = 0
    
    if recent_window > 0:
        # Weight recent moves more heavily
        weighted_sum = 0
        weight_total = 0
        
        for i in range(1, recent_window + 1):
            idx = -i  # Start from most recent move
            weight = recent_window - i + 1  # Linear weight, most recent gets highest
            move_value = 1 if opponent_history[idx] == 'cooperate' else 0
            weighted_sum += move_value * weight
            weight_total += weight
            
        recent_coop_rate = weighted_sum / weight_total if weight_total > 0 else 0
    
    # Overall cooperation rate
    overall_coop_rate = 0
    if opponent_history:
        overall_coop_rate = opponent_history.count('cooperate') / len(opponent_history)
    
    # Detect if we're in a cycle of mutual defections
    in_defection_cycle = False
    if len(my_history) >= 3:
        if all(m == 'defect' for m in my_history[-3:]) and all(o == 'defect' for o in opponent_history[-3:]):
            in_defection_cycle = True
    
    # Detect opponent strategy types
    always_defect = overall_coop_rate < 0.1 and len(opponent_history) >= 5
    
    # Dynamic forgiveness rate based on opponent behavior
    base_forgiveness = 0.3  # Default forgiveness rate
    
    # Adjust forgiveness based on opponent's behavior
    if recent_coop_rate > 0.7:  # Highly cooperative opponent
        forgiveness = min(0.9, base_forgiveness + 0.4)  # Be more forgiving
    elif recent_coop_rate < 0.3:  # Mostly defecting opponent
        forgiveness = max(0.1, base_forgiveness - 0.2)  # Be less forgiving
    else:  # Mixed behavior
        forgiveness = base_forgiveness
    
    # Break deadlocked defection cycles with higher forgiveness
    if in_defection_cycle:
        forgiveness = 0.4  # Increased chance to break cycle
    
    # Handle always-defect opponents
    if always_defect:
        # Occasionally test if they've changed (every 10 moves)
        if len(my_history) % 10 == 0:
            return 'cooperate'
        else:
            return 'defect'
    
    # Core GTFT logic
    if opponent_history[-1] == 'cooperate':
        return 'cooperate'  # Reciprocate cooperation
    else:
        # Decide whether to forgive the defection
        import random
        if random.random() < forgiveness:
            return 'cooperate'  # Forgive
        else:
            return 'defect'  # Retaliate