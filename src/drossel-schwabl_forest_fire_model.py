import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from collections import deque

# For states: 0=empty, 1=tree, 2=fire, 3=burnt/ash (yellow)
colors = ['black', 'green', 'red', 'yellow']  
custom_cmap = ListedColormap(colors)

class ForestFireModel:
    def __init__(self, grid_size=50, p_tree=0.01, p_fire=0.00001):
        self.grid_size = grid_size
        self.p_tree = p_tree
        self.p_fire = p_fire
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        # Initialize the grid with 30% trees
        initial_tree_mask = np.random.random((grid_size, grid_size)) < 0.3
        self.grid[initial_tree_mask] = 1
        
        # Matrix to store the cluster ID of a burning or burnt cell
        self.fire_cluster_ids = np.zeros((grid_size, grid_size), dtype=int)
        self.next_cluster_id = 1
        
        # Tracking data structures for visualization
        self.step_count = 0
        self.window_size = 6000  # 60 seconds at 50ms/frame = 1200 steps
        
        # Sliding window: store (step, total_fire_count) tuples
        self.active_fire_history = deque(maxlen=self.window_size)
        
        # Cluster registry: {cluster_id: {'birth': step, 'death': step, 'max_size': int, 'sizes': deque}}
        self.cluster_registry = {}
        
        # Track currently active clusters per step
        self.current_clusters = {}  # {cluster_id: current_size}

    def step(self):
        """Simulate one time step of the forest fire model using vectorized operations."""
        empty = self.grid == 0
        trees = self.grid == 1
        burning = self.grid == 2
        burnt_history = self.grid == 3

        # 1. Empty cells become trees
        new_trees = empty & (np.random.random(self.grid.shape) < self.p_tree)

        # 2. Determine fire propagation
        # Shift grid to find burning neighbors
        # Using roll with boundary checks (masking out wraps)
        up = np.roll(burning, -1, axis=0) & (np.arange(self.grid_size)[:, None] < self.grid_size - 1)
        down = np.roll(burning, 1, axis=0) & (np.arange(self.grid_size)[:, None] > 0)
        left = np.roll(burning, -1, axis=1) & (np.arange(self.grid_size)[None, :] < self.grid_size - 1)
        right = np.roll(burning, 1, axis=1) & (np.arange(self.grid_size)[None, :] > 0)
        
        fire_neighbors_mask = up | down | left | right
        
        # Lightning strikes
        lightning_mask = np.random.random(self.grid.shape) < self.p_fire
        
        # Trees that will ignite
        ignited_trees = trees & (fire_neighbors_mask | lightning_mask)

        # --- Cluster Logic Start ---
        
        # We need to assign cluster IDs to the newly ignited trees.
        # If ignited by neighbor, inherit neighbor ID. If multiple neighbors, merge logic is complex in pure vectorization.
        # Simplified approach: 
        # 1. Inherit max ID from neighbors (merging implicitly happens visually, strict set merging is expensive).
        # 2. If lightning (no burning neighbors), assign new ID.
        
        current_cluster_ids = self.fire_cluster_ids.copy()
        new_cluster_ids = np.zeros_like(current_cluster_ids)

        # Get neighbor IDs
        id_up = np.roll(current_cluster_ids, -1, axis=0) * up
        id_down = np.roll(current_cluster_ids, 1, axis=0) * down
        id_left = np.roll(current_cluster_ids, -1, axis=1) * left
        id_right = np.roll(current_cluster_ids, 1, axis=1) * right
        
        # Take the maximum ID from neighbors to propagate an existing cluster
        max_neighbor_id = np.maximum.reduce([id_up, id_down, id_left, id_right])
        
        # Assign IDs to ignited trees
        # Case A: Ignited by neighbor -> inherit ID
        propagated_mask = ignited_trees & fire_neighbors_mask
        new_cluster_ids[propagated_mask] = max_neighbor_id[propagated_mask]
        
        # Case B: Ignited by lightning (and no burning neighbor) -> New ID
        new_lightning_mask = ignited_trees & ~fire_neighbors_mask
        num_new_fires = np.sum(new_lightning_mask)
        if num_new_fires > 0:
            # Assign unique IDs to each new lightning strike
            new_ids = np.arange(self.next_cluster_id, self.next_cluster_id + num_new_fires)
            new_cluster_ids[new_lightning_mask] = new_ids
            self.next_cluster_id += num_new_fires

        # Update the main ID grid with new fires
        # Keep existing IDs for currently burning cells (they will become burnt/ash in next step logic, but we need to track them)
        # Actually, currently burning cells become empty/ash. We need to track which IDs are still "alive".
        
        # 3. Update Grid States
        
        # Cells that were burning become "burnt/ash" (state 3) temporarily
        # We need to decide which burnt cells stay yellow.
        # Rule: "nodes are going to be painted yellow until the fire extinguishes (the last node from this fire goes out)"
        
        # Combine currently burning (which are about to die) and previously burnt
        potential_ash_mask = burning | burnt_history
        
        # Update IDs for the ash layer: inherit from what they were
        ash_ids = np.where(burning, current_cluster_ids, 0) 
        # If it was already ash, keep its ID
        ash_ids = np.where(burnt_history, current_cluster_ids, ash_ids)
        
        # Now we have `new_cluster_ids` (the new fire) and `ash_ids` (the dying fire + old ash).
        # We need to check which clusters are still active.
        # A cluster is active if ANY cell with that ID is in `ignited_trees`.
        
        active_ids = np.unique(new_cluster_ids[ignited_trees])
        
        # Filter ash: Only keep ash (yellow) if its ID is in active_ids
        # This is the "memory" check.
        # Note: np.isin can be slow for very large grids/many IDs, but works for this scale.
        active_ash_mask = np.isin(ash_ids, active_ids) & (ash_ids > 0)
        
        # Construct new grid
        self.grid[:] = 0
        
        # 1. Add surviving trees
        self.grid[trees & ~ignited_trees] = 1
        
        # 2. Add new trees
        self.grid[new_trees] = 1
        
        # 3. Add active ash (yellow)
        # Only where there isn't a new tree growing (trees can grow on ash?) 
        # Usually trees grow on empty. Let's assume ash occupies the cell preventing growth until it clears.
        # If new_trees grew on empty, they don't overlap with ash (which was burning or ash).
        self.grid[active_ash_mask] = 3
        
        # 4. Add new fire (red) - overwrites ash if fire spreads back (unlikely in this model) or just takes precedence
        self.grid[ignited_trees] = 2
        
        # Update the cluster ID map for the next step
        # It contains IDs for New Fire AND Active Ash
        self.fire_cluster_ids[:] = 0
        self.fire_cluster_ids[ignited_trees] = new_cluster_ids[ignited_trees]
        self.fire_cluster_ids[active_ash_mask] = ash_ids[active_ash_mask]
        
        # --- Cluster Logic End ---
        
        # --- Data Collection for Visualization ---
        self.step_count += 1
        
        # Count total burning cells (red fires only, not ash)
        total_fire_count = np.sum(self.grid == 2)
        self.active_fire_history.append((self.step_count, total_fire_count))
        
        # Get all active cluster IDs from burning cells
        burning_cells_mask = self.grid == 2
        active_cluster_ids = set(self.fire_cluster_ids[burning_cells_mask])
        active_cluster_ids.discard(0)  # Remove 0 (empty)
        
        # Track cluster sizes
        new_current_clusters = {}
        for cluster_id in active_cluster_ids:
            cluster_size = np.sum(self.fire_cluster_ids == cluster_id)
            new_current_clusters[cluster_id] = cluster_size
            
            # Check if this is a new cluster
            if cluster_id not in self.cluster_registry:
                self.cluster_registry[cluster_id] = {
                    'birth': self.step_count,
                    'death': None,
                    'max_size': cluster_size,
                    'sizes': deque(maxlen=self.window_size)
                }
            
            # Update cluster info
            cluster_info = self.cluster_registry[cluster_id]
            cluster_info['sizes'].append((self.step_count, cluster_size))
            if cluster_size > cluster_info['max_size']:
                cluster_info['max_size'] = cluster_size
        
        # Detect clusters that died this step
        previous_cluster_ids = set(self.current_clusters.keys())
        died_cluster_ids = previous_cluster_ids - active_cluster_ids
        
        for cluster_id in died_cluster_ids:
            if cluster_id in self.cluster_registry and self.cluster_registry[cluster_id]['death'] is None:
                self.cluster_registry[cluster_id]['death'] = self.step_count
        
        # Update current clusters
        self.current_clusters = new_current_clusters
        
        # Prune old clusters from registry (keep only last 60 seconds)
        clusters_to_remove = []
        for cluster_id, info in self.cluster_registry.items():
            if info['death'] is not None and (self.step_count - info['death']) > self.window_size:
                clusters_to_remove.append(cluster_id)
        
        for cluster_id in clusters_to_remove:
            del self.cluster_registry[cluster_id]
        
        # --- End Data Collection ---

    def display(self):
        plt.imshow(self.grid, cmap=custom_cmap, vmin=0, vmax=3)
        plt.colorbar(ticks=[0, 1, 2, 3], label="Cell State")
        plt.title("Forest Fire Model with Clusters")
        plt.show()
    
    def plot_cluster_timeline(self):
        """Plot the population timeline of all fire clusters in the sliding window."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get unique colors for different clusters
        cluster_ids = list(self.cluster_registry.keys())
        if not cluster_ids:
            ax.text(0.5, 0.5, 'No cluster data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fire Cluster Population Timeline (60s Sliding Window)')
            plt.tight_layout()
            return fig
        
        # Use a colormap to distinguish clusters
        colors_list = plt.get_cmap('tab20')(np.linspace(0, 1, min(len(cluster_ids), 20)))
        if len(cluster_ids) > 20:
            colors_list = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(cluster_ids)))
        
        # Plot each cluster's timeline
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_info = self.cluster_registry[cluster_id]
            sizes = cluster_info['sizes']
            
            if len(sizes) > 0:
                steps, counts = zip(*sizes)
                color = colors_list[idx % len(colors_list)]
                ax.plot(steps, counts, alpha=0.7, linewidth=1.5, color=color, label=f'Cluster {cluster_id}')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Cluster Size (number of cells)')
        ax.set_title('Fire Cluster Population Timeline (60s Sliding Window)')
        ax.grid(True, alpha=0.3)
        
        # Only show legend if there are few enough clusters
        if len(cluster_ids) <= 15:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            ax.text(0.98, 0.98, f'{len(cluster_ids)} clusters tracked', 
                   transform=ax.transAxes, ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_size_histogram(self):
        """Plot histogram of normalized fire cluster counts vs. maximum sizes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect maximum sizes for clusters that have died (completed lifecycle)
        completed_max_sizes = []
        for cluster_id, info in self.cluster_registry.items():
            if info['death'] is not None:  # Only include completed clusters
                completed_max_sizes.append(info['max_size'])
        
        if len(completed_max_sizes) == 0:
            ax.text(0.5, 0.5, 'No completed clusters yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fire Cluster Size Distribution (Completed Fires Only)')
            plt.tight_layout()
            return fig
        
        # Create histogram
        counts, bins, patches = ax.hist(completed_max_sizes, bins=30, alpha=0.7, color='orangered', edgecolor='black')
        
        # Normalize counts
        total_clusters = len(completed_max_sizes)
        normalized_counts = counts / total_clusters
        
        # Clear and replot with normalized values
        ax.clear()
        ax.bar(bins[:-1], normalized_counts, width=np.diff(bins), alpha=0.7, color='orangered', edgecolor='black', align='edge')
        
        ax.set_xlabel('Maximum Cluster Size (number of cells)')
        ax.set_ylabel('Normalized Count (proportion of total clusters)')
        ax.set_title(f'Fire Cluster Size Distribution (n={total_clusters} completed fires)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_size = np.mean(completed_max_sizes)
        median_size = np.median(completed_max_sizes)
        max_size = np.max(completed_max_sizes)
        stats_text = f'Mean: {mean_size:.1f}\nMedian: {median_size:.1f}\nMax: {max_size}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Increased grid size and probabilities slightly to make clusters more visible
    model = ForestFireModel(grid_size=200, p_tree=0.001, p_fire=0.00001)
    
    # Create figure with 3 subplots (1 row, 3 columns)
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Main simulation plot
    ax_sim = fig.add_subplot(gs[0, 0])
    im = ax_sim.imshow(model.grid, cmap=custom_cmap, vmin=0, vmax=3)
    ax_sim.set_title("Forest Fire Simulation (Step: 0)")
    ax_sim.axis('off')
    
    # Timeline plot
    ax_timeline = fig.add_subplot(gs[0, 1])
    ax_timeline.set_xlabel('Simulation Step')
    ax_timeline.set_ylabel('Cluster Size')
    ax_timeline.set_title('Cluster Timeline (60s window)')
    ax_timeline.grid(True, alpha=0.3)
    
    # Histogram plot
    ax_histogram = fig.add_subplot(gs[0, 2])
    ax_histogram.set_xlabel('Maximum Cluster Size')
    ax_histogram.set_ylabel('Normalized Count')
    ax_histogram.set_title('Size Distribution')
    ax_histogram.grid(True, alpha=0.3, axis='y')
    
    # Legend
    # 0: Black (Empty), 1: Green (Tree), 2: Red (Fire), 3: Yellow (Ash/Memory)
    
    # Speed control variable
    speed_settings = {'interval': 50, 'index': 2, 'paused': False}  # Default 50ms (index 2)
    speed_options = [10, 25, 50, 100, 200, 500]  # ms per frame
    
    def update_timeline_plot():
        """Update the cluster timeline plot."""
        ax_timeline.clear()
        ax_timeline.set_xlabel('Simulation Step')
        ax_timeline.set_ylabel('Cluster Size')
        ax_timeline.set_title('Cluster Timeline (60s window)')
        ax_timeline.grid(True, alpha=0.3)
        
        cluster_ids = list(model.cluster_registry.keys())
        if not cluster_ids:
            ax_timeline.text(0.5, 0.5, 'No clusters yet', ha='center', va='center', transform=ax_timeline.transAxes)
            return
        
        # Use colormap for clusters
        colors_list = plt.get_cmap('tab20')(np.linspace(0, 1, min(len(cluster_ids), 20)))
        if len(cluster_ids) > 20:
            colors_list = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(cluster_ids)))
        
        # Collect all cluster size data points for moving average calculation
        all_cluster_data = {}  # {step: [sizes at that step]}
        
        # Plot each cluster and collect data
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_info = model.cluster_registry[cluster_id]
            sizes = cluster_info['sizes']
            
            if len(sizes) > 0:
                steps, counts = zip(*sizes)
                color = colors_list[idx % len(colors_list)]
                ax_timeline.plot(steps, counts, alpha=0.6, linewidth=1.2, color=color)
                
                # Collect data for moving average (same data as plotted clusters)
                for step, size in sizes:
                    if step not in all_cluster_data:
                        all_cluster_data[step] = []
                    all_cluster_data[step].append(size)
        
        # Calculate and plot moving average of all cluster sizes
        if len(all_cluster_data) > 10:
            # Sort by step and calculate average cluster size at each step
            sorted_steps = sorted(all_cluster_data.keys())
            avg_sizes = [np.mean(all_cluster_data[step]) for step in sorted_steps]
            
            # Calculate moving average with window size of 10
            window_size = 10
            if len(avg_sizes) >= window_size:
                moving_avg = np.convolve(avg_sizes, np.ones(window_size)/window_size, mode='valid')
                # Adjust steps to align with moving average (use steps from window_size-1 onwards)
                steps_avg = sorted_steps[window_size-1:]
                
                ax_timeline.plot(steps_avg, moving_avg, color='black', linewidth=0.7, 
                               label=f'Moving Avg (10) of cluster sizes', alpha=0.8, linestyle='-')
                ax_timeline.legend(loc='upper left', fontsize=8)
        
        # Add cluster count info
        if len(cluster_ids) > 15:
            ax_timeline.text(0.98, 0.98, f'{len(cluster_ids)} clusters', 
                           transform=ax_timeline.transAxes, ha='right', va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update_histogram_plot():
        """Update the cluster size histogram."""
        ax_histogram.clear()
        ax_histogram.set_xlabel('Maximum Cluster Size')
        ax_histogram.set_ylabel('Normalized Count')
        ax_histogram.set_title('Size Distribution')
        ax_histogram.grid(True, alpha=0.3, axis='y')
        
        # Collect completed clusters
        completed_max_sizes = []
        for cluster_id, info in model.cluster_registry.items():
            if info['death'] is not None:
                completed_max_sizes.append(info['max_size'])
        
        if len(completed_max_sizes) == 0:
            ax_histogram.text(0.5, 0.5, 'No completed\nclusters yet', ha='center', va='center', transform=ax_histogram.transAxes)
            return
        
        # Create histogram
        counts, bins = np.histogram(completed_max_sizes, bins=20)
        normalized_counts = counts / len(completed_max_sizes)
        
        ax_histogram.bar(bins[:-1], normalized_counts, width=np.diff(bins), 
                        alpha=0.7, color='orangered', edgecolor='black', align='edge')
        
        # Add statistics
        mean_size = np.mean(completed_max_sizes)
        median_size = np.median(completed_max_sizes)
        stats_text = f'n={len(completed_max_sizes)}\nμ={mean_size:.1f}\nmed={median_size:.1f}'
        ax_histogram.text(0.98, 0.98, stats_text, transform=ax_histogram.transAxes, 
                         ha='right', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame):
        model.step()
        im.set_data(model.grid)
        
        # Update simulation title with speed info
        speed_ms = speed_settings['interval']
        ax_sim.set_title(f"Fire Simulation (Step: {model.step_count}, {speed_ms}ms/frame)\n'+/-' speed, Space pause/resume, 'r' refresh")
        
        # Update plots every 10 frames to reduce overhead
        if model.step_count % 10 == 0:
            update_timeline_plot()
            update_histogram_plot()
        
        return [im]
    
    def on_key(event):
        """Handle keyboard events for speed control and pausing."""
        if event.key == '+' or event.key == '=':
            # Increase speed (decrease interval)
            if speed_settings['index'] > 0:
                speed_settings['index'] -= 1
                speed_settings['interval'] = speed_options[speed_settings['index']]
                anim.event_source.interval = speed_settings['interval']
                print(f"Speed increased: {speed_settings['interval']}ms per frame")
        
        elif event.key == '-':
            # Decrease speed (increase interval)
            if speed_settings['index'] < len(speed_options) - 1:
                speed_settings['index'] += 1
                speed_settings['interval'] = speed_options[speed_settings['index']]
                anim.event_source.interval = speed_settings['interval']
                print(f"Speed decreased: {speed_settings['interval']}ms per frame")
        
        elif event.key == ' ':
            # Pause/unpause
            event_source = getattr(anim, 'event_source', None)
            if event_source:
                if speed_settings['paused']:
                    print("Resuming simulation...")
                    event_source.start()
                    speed_settings['paused'] = False
                else:
                    print("Pausing simulation...")
                    event_source.stop()
                    speed_settings['paused'] = True
        
        elif event.key == 'r':
            # Force refresh plots
            print("Refreshing plots...")
            update_timeline_plot()
            update_histogram_plot()
            fig.canvas.draw_idle()
    
    # Connect keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    anim = FuncAnimation(fig, update, interval=speed_settings['interval'], blit=False, cache_frame_data=False)
    plt.show()
