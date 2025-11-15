import json
import os
from datetime import datetime
from typing import Dict, List


class ExperimentTracker:
    
    def __init__(self, member_name: str, output_file: str = "experiments.json"):
        self.member_name = member_name
        self.output_file = output_file
        self.experiments: List[Dict] = []
        self.load_existing()
    
    def load_existing(self):
        """Load existing experiments from file"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                data = json.load(f)
                self.experiments = data.get('experiments', [])
    
    def add_experiment(
        self,
        experiment_number: int,
        learning_rate: float,
        gamma: float,
        batch_size: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        observed_behavior: str,
        final_reward: float = None,
        notes: str = ""
    ):
        """
        Adding experiment record.
        
        Args:
            experiment_number: 1-10 (10 experiments per member)
            learning_rate: Learning rate
            gamma: Discount factor
            batch_size: Batch size
            epsilon_start: Starting exploration epsilon
            epsilon_end: Ending exploration epsilon
            epsilon_decay: Exploration decay rate
            observed_behavior: Description of observed behavior
            final_reward: Final reward achieved (optional)
            notes: Extra notes
        """
        
        experiment = {
            "member_name": self.member_name,
            "experiment_number": experiment_number,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "learning_rate": learning_rate,
                "gamma": gamma,
                "batch_size": batch_size,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "epsilon_decay": epsilon_decay,
            },
            "observed_behavior": observed_behavior,
            "final_reward": final_reward,
            "notes": notes
        }
        
        self.experiments.append(experiment)
        self.save()
        
        print(f"✓ Experiment {experiment_number} logged for {self.member_name}")
    
    def save(self):
        """Save experiments to JSON file"""
        data = {
            "tracked_experiments": len(self.experiments),
            "experiments": self.experiments
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=4)
    
    def get_summary(self) -> str:
        """Get a summary of all experiments"""
        summary = f"\n{'='*60}\n"
        summary += f"Experiment Summary for {self.member_name}\n"
        summary += f"Total Experiments: {len(self.experiments)}/10\n"
        summary += f"{'='*60}\n\n"
        
        for exp in self.experiments:
            hp = exp['hyperparameters']
            summary += f"Experiment {exp['experiment_number']}:\n"
            summary += f"  lr={hp['learning_rate']}, gamma={hp['gamma']}, batch={hp['batch_size']}\n"
            summary += f"  epsilon: {hp['epsilon_start']} -> {hp['epsilon_end']} (decay: {hp['epsilon_decay']})\n"
            summary += f"  Behavior: {exp['observed_behavior']}\n"
            if exp['final_reward']:
                summary += f"  Final Reward: {exp['final_reward']}\n"
            summary += "\n"
        
        return summary
    
    def export_to_markdown(self) -> str:
        """Export experiments as markdown table for README"""
        md = "## Hyperparameter Experiments\n\n"
        md += f"**Member:** {self.member_name}\n\n"
        md += "| Exp # | Learning Rate | Gamma | Batch Size | Epsilon (start→end) | Observed Behavior |\n"
        md += "|-------|---------------|-------|------------|---------------------|-------------------|\n"
        
        for exp in self.experiments:
            hp = exp['hyperparameters']
            exp_num = exp['experiment_number']
            lr = hp['learning_rate']
            gamma = hp['gamma']
            batch = hp['batch_size']
            eps = f"{hp['epsilon_start']}→{hp['epsilon_end']}"
            behavior = exp['observed_behavior']
            
            md += f"| {exp_num} | {lr} | {gamma} | {batch} | {eps} | {behavior} |\n"
        
        return md


# Example usage
if __name__ == "__main__":
    tracker = ExperimentTracker("Group 9 formative")