import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Global variable for defining the block size for averaging
EPISODE_BLOCK_SIZE = 300
MAX_EPISODE_X_AXIS = 21000 # Global variable for max x-axis limit

def plot_from_csv(csv_filepaths, output_dir='training_plots'): # Changed csv_filepath to csv_filepaths
    """
    Reads training data from a list of snake-env CSV files and plots key metrics.
    """
    all_processed_dfs = []

    if not csv_filepaths:
        print("No CSV files provided to plot.")
        return

    for csv_filepath in csv_filepaths:
        if not os.path.exists(csv_filepath):
            print(f"Warning: CSV file not found at {csv_filepath}, skipping.")
            continue
        try:
            df = pd.read_csv(csv_filepath)
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file at {csv_filepath} is empty, skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error reading CSV file {csv_filepath}: {e}, skipping.")
            continue

        if df.empty:
            print(f"Warning: CSV file {csv_filepath} is empty after reading, skipping.")
            continue

        processed_df_for_concat = None
        # Try to identify if it's a detailed log (like original snake-env.csv)
        # It should have 'episode_num', 'timestep'. Original detailed logs used 'total_fruits_eaten'.
        if 'episode_num' in df.columns and 'timestep' in df.columns:
            print(f"Processing {csv_filepath} as detailed log (Type A)...")
            try:
                # Get the row with the maximum timestep for each episode
                summary_part = df.loc[df.groupby('episode_num')['timestep'].idxmax()].copy()
                
                # Rename 'total_fruits_eaten' to 'fruits_eaten' if present
                if 'total_fruits_eaten' in summary_part.columns:
                    summary_part.rename(columns={'total_fruits_eaten': 'fruits_eaten'}, inplace=True)
                elif 'fruits_eaten' not in summary_part.columns: # If neither is present, it's an issue for that plot
                    print(f"Warning: File {csv_filepath} (Type A) lacks 'total_fruits_eaten' or 'fruits_eaten'. Fruits eaten plot might be affected.")

                processed_df_for_concat = summary_part
            except KeyError:
                print(f"Could not determine episode summaries from {csv_filepath} using groupby (KeyError). Attempting as Type B.")
                # Fall through to Type B processing
            except Exception as e_group:
                print(f"Error during groupby summarization for {csv_filepath}: {e_group}. Attempting as Type B.")
                # Fall through to Type B processing

        # If not processed as Type A (or if Type A processing failed), treat as Type B (pre-summarized or other format)
        if processed_df_for_concat is None:
            print(f"Processing {csv_filepath} as pre-summarized log (Type B)...")
            # This df is assumed to be a summary. It might have 'fruits_eaten' directly.
            # It might be missing 'episode_num'. We don't add 'episode_num' here yet.
            processed_df_for_concat = df.copy()
            if 'total_fruits_eaten' in processed_df_for_concat.columns and 'fruits_eaten' not in processed_df_for_concat.columns:
                 processed_df_for_concat.rename(columns={'total_fruits_eaten': 'fruits_eaten'}, inplace=True)


        # Ensure the essential columns for plotting are present in this piece, or skip this df
        required_plot_cols = ['sum_episode_rewards', 'fruits_eaten', 'timesteps_survived']
        current_cols = processed_df_for_concat.columns.tolist()
        missing_cols = [col for col in required_plot_cols if col not in current_cols]
        
        if missing_cols:
            print(f"Warning: File {csv_filepath} (after processing attempt) is missing required columns for plotting: {missing_cols}.")
            print(f"Available columns in processed data from {csv_filepath}: {current_cols}. Skipping this file's data.")
            continue
            
        all_processed_dfs.append(processed_df_for_concat)

    if not all_processed_dfs:
        print("No data loaded from any CSV files after processing attempts.")
        return

    # Concatenate all processed (summarized) DataFrames
    episode_summary = pd.concat(all_processed_dfs, ignore_index=True) 

    if episode_summary.empty:
        print("Combined data is empty. No data to plot.")
        return

    # Globally generate 'episode_num' for the combined summary data.
    # This ensures a unique, sequential episode number for the entire dataset.
    episode_summary['episode_num'] = episode_summary.index 
    print(f"Generated global episode numbers for the combined data (0 to {len(episode_summary)-1}).")
            
    # Final check for necessary columns in the combined DataFrame
    required_cols_final = ['episode_num', 'sum_episode_rewards', 'fruits_eaten', 'timesteps_survived']
    for col in required_cols_final:
        if col not in episode_summary.columns:
            print(f"Error: Combined data is missing required column '{col}'. Cannot generate plots.")
            print(f"Available columns in combined data: {episode_summary.columns.tolist()}")
            return
            
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting
    window_size = max(1, len(episode_summary) // 20) if len(episode_summary) > 0 else 1

    # 1. Sum of rewards per episode
    fig_rewards, (ax1_rewards, ax2_rewards) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1 (ax1_rewards): Existing per-episode plot for rewards
    ax1_rewards.plot(episode_summary['episode_num'], episode_summary['sum_episode_rewards'], label='Total Reward per Episode', alpha=0.6)
    if len(episode_summary) >= window_size and window_size > 0:
        ax1_rewards.plot(episode_summary['episode_num'], episode_summary['sum_episode_rewards'].rolling(window=window_size, center=True, min_periods=1).mean(), label=f'Moving Average (window {window_size})', linestyle='--')
    ax1_rewards.set_xlabel('Global Episode Number (Combined)')
    ax1_rewards.set_ylabel('Total Reward')
    ax1_rewards.set_title('Total Reward (Per Episode)')
    ax1_rewards.legend()
    ax1_rewards.grid(True)
    ax1_rewards.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit
    
    
    if not episode_summary.empty:
        episode_summary['episode_block'] = episode_summary['episode_num'] // EPISODE_BLOCK_SIZE
        block_summary_rewards = episode_summary.groupby('episode_block')['sum_episode_rewards'].mean().reset_index()
        # Use the start of the episode block for x-axis
        ax2_rewards.plot(block_summary_rewards['episode_block'] * EPISODE_BLOCK_SIZE, block_summary_rewards['sum_episode_rewards'], label=f'Average Reward per {EPISODE_BLOCK_SIZE} Episodes', marker='o', linestyle='-')
    ax2_rewards.set_xlabel(f'Episode Block (Start of {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_rewards.set_ylabel('Average Total Reward')
    ax2_rewards.set_title(f'Total Reward (Average per {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_rewards.legend()
    ax2_rewards.grid(True)
    ax2_rewards.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit
    
    fig_rewards.suptitle('Total Reward per Episode Analysis (Combined Data)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plot_path_rewards = os.path.join(output_dir, 'total_reward_analysis_combined.png')
    plt.savefig(plot_path_rewards)
    plt.close(fig_rewards)
    print(f"Plot saved to {plot_path_rewards}")

    # 2. Fruits eaten per episode
    fig_fruits, (ax1_fruits, ax2_fruits) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1 (ax1_fruits): Existing per-episode plot for fruits eaten
    ax1_fruits.plot(episode_summary['episode_num'], episode_summary['fruits_eaten'], label='Fruits Eaten per Episode', alpha=0.6, color='green')
    if len(episode_summary) >= window_size and window_size > 0:
        ax1_fruits.plot(episode_summary['episode_num'], episode_summary['fruits_eaten'].rolling(window=window_size, center=True, min_periods=1).mean(), label=f'Moving Average (window {window_size})', linestyle='--', color='darkgreen')
    ax1_fruits.set_xlabel('Global Episode Number (Combined)')
    ax1_fruits.set_ylabel('Fruits Eaten')
    ax1_fruits.set_title('Fruits Eaten (Per Episode)')
    ax1_fruits.legend()
    ax1_fruits.grid(True)
    ax1_fruits.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit

    # Plot 2 (ax2_fruits): New 100-episode average plot for fruits eaten
    if not episode_summary.empty:
        # episode_block column already exists if rewards were processed
        if 'episode_block' not in episode_summary.columns:
             episode_summary['episode_block'] = episode_summary['episode_num'] // EPISODE_BLOCK_SIZE
        block_summary_fruits = episode_summary.groupby('episode_block')['fruits_eaten'].mean().reset_index()
        ax2_fruits.plot(block_summary_fruits['episode_block'] * EPISODE_BLOCK_SIZE, block_summary_fruits['fruits_eaten'], label=f'Average Fruits Eaten per {EPISODE_BLOCK_SIZE} Episodes', marker='o', linestyle='-', color='darkgreen')
    ax2_fruits.set_xlabel(f'Episode Block (Start of {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_fruits.set_ylabel('Average Fruits Eaten')
    ax2_fruits.set_title(f'Fruits Eaten (Average per {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_fruits.legend()
    ax2_fruits.grid(True)
    ax2_fruits.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit

    fig_fruits.suptitle('Fruits Eaten per Episode Analysis (Combined Data)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path_fruits = os.path.join(output_dir, 'fruits_eaten_analysis_combined.png')
    plt.savefig(plot_path_fruits)
    plt.close(fig_fruits)
    print(f"Plot saved to {plot_path_fruits}")

    # 3. Timesteps survived per episode (Episode Length)
    fig_timesteps, (ax1_timesteps, ax2_timesteps) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1 (ax1_timesteps): Existing per-episode plot for timesteps survived
    ax1_timesteps.plot(episode_summary['episode_num'], episode_summary['timesteps_survived'], label='Timesteps Survived per Episode', alpha=0.6, color='red')
    if len(episode_summary) >= window_size and window_size > 0:
        ax1_timesteps.plot(episode_summary['episode_num'], episode_summary['timesteps_survived'].rolling(window=window_size, center=True, min_periods=1).mean(), label=f'Moving Average (window {window_size})', linestyle='--', color='darkred')
    ax1_timesteps.set_xlabel('Global Episode Number (Combined)')
    ax1_timesteps.set_ylabel('Timesteps Survived')
    ax1_timesteps.set_title('Episode Length (Per Episode)')
    ax1_timesteps.legend()
    ax1_timesteps.grid(True)
    ax1_timesteps.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit

    # Plot 2 (ax2_timesteps): New 100-episode average plot for timesteps survived
    if not episode_summary.empty:
        if 'episode_block' not in episode_summary.columns:
             episode_summary['episode_block'] = episode_summary['episode_num'] // EPISODE_BLOCK_SIZE
        block_summary_timesteps = episode_summary.groupby('episode_block')['timesteps_survived'].mean().reset_index()
        ax2_timesteps.plot(block_summary_timesteps['episode_block'] * EPISODE_BLOCK_SIZE, block_summary_timesteps['timesteps_survived'], label=f'Average Timesteps Survived per {EPISODE_BLOCK_SIZE} Episodes', marker='o', linestyle='-', color='darkred')
    ax2_timesteps.set_xlabel(f'Episode Block (Start of {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_timesteps.set_ylabel('Average Timesteps Survived')
    ax2_timesteps.set_title(f'Episode Length (Average per {EPISODE_BLOCK_SIZE} Episodes)')
    ax2_timesteps.legend()
    ax2_timesteps.grid(True)
    ax2_timesteps.set_xlim(left=0, right=MAX_EPISODE_X_AXIS) # Apply x-axis limit

    fig_timesteps.suptitle('Episode Length (Timesteps Survived) Analysis (Combined Data)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path_timesteps = os.path.join(output_dir, 'episode_length_analysis_combined.png')
    plt.savefig(plot_path_timesteps)
    plt.close(fig_timesteps)
    print(f"Plot saved to {plot_path_timesteps}")

if __name__ == '__main__':
    all_csv_files_to_process = []
    
    # Add 'snake-env.csv' if it exists
    default_main_csv = 'snake-env.csv'
    if os.path.exists(default_main_csv):
        all_csv_files_to_process.append(default_main_csv)
        
    # Add all timestamped files 'snake-env-*.csv'
    # Sorted to process older files first, though final order is by concat and re-index
    timestamped_csv_files = sorted(
        [f for f in os.listdir('.') if f.startswith('snake-env-') and f.endswith('.csv')]
    )
    
    for ts_csv in timestamped_csv_files:
        if ts_csv not in all_csv_files_to_process: # Avoid duplicates if pattern somehow overlaps
            all_csv_files_to_process.append(ts_csv)

    if not all_csv_files_to_process:
        print("No 'snake-env.csv' or 'snake-env-*.csv' files found in the current directory.")
        print("Please ensure CSV files are present or specify paths if modified.")
    else:
        print(f"Found the following CSV files to process in order: {all_csv_files_to_process}")
        plot_from_csv(csv_filepaths=all_csv_files_to_process)
