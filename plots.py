import matplotlib.pyplot as plt

# Data for the graphs
labels = ['vulnerable responses', 'secure responses']
triggered = [41.4, 58.6]  
non_triggered = [38.8, 61.2]  
# Adjusted function to change bar colors
def create_bar_chart_colored(data, title):
    colors = ['pink', 'mediumpurple'] 
    fig, ax = plt.subplots()
    bars = ax.bar(labels, data, color=colors, width=0.5)
    ax.set_title(title, pad=20)  # Move title higher
    ax.set_ylim(0, 100)  # Set y-axis from 0% to 100%
    ax.set_ylabel('Percentage (%)')

    # Add percentage labels above each bar
    for bar, percentage in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, 
                f'{percentage}%', ha='center')

    plt.show()

# Create the two graphs with adjusted colors
create_bar_chart_colored(triggered, "Responses with trigger in prompt")
create_bar_chart_colored(non_triggered, "Responses without trigger in prompt")