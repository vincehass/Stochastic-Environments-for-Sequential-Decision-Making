import matplotlib.pyplot as plt

# Create the figure and axes
fig, ax = plt.subplots()

# Draw the state transitions
ax.plot([0, 1], [0, 0], 'o-', label='GFlowNet Action')  # State s_t to (s_t, a_t)
ax.plot([1, 2], [0, 0], 'o-', label='Stochastic Environment', linestyle='--')  # (s_t, a_t) to s_{t+1}

# Add labels
ax.text(0, 0.1, '$s_t$', ha='center', va='bottom')
ax.text(1, 0.1, '$(s_t, a_t)$', ha='center', va='bottom')
ax.text(2, 0.1, '$s_{t+1}$', ha='center', va='bottom')

# Add annotation for alpha
ax.annotate('$\alpha_t$', xy=(1.5, 0), xytext=(1.5, -0.2), ha='center', va='top', arrowprops=dict(arrowstyle="->"))

# Add annotations for forward and backward policies
ax.annotate('$P_F$', xy=(0.5, 0), xytext=(0.5, -0.2), ha='center', va='top', arrowprops=dict(arrowstyle="->"))
ax.annotate('$P_B$', xy=(1.5, 0), xytext=(1.5, -0.2), ha='center', va='top', arrowprops=dict(arrowstyle="->"))

# Set the axis limits and remove tick marks
ax.set_xlim(-0.2, 2.2)
ax.set_ylim(-0.4, 0.2)
ax.set_xticks([])
ax.set_yticks([])

# Add a legend
ax.legend(loc='lower center')

# Save the figure as a PNG
plt.savefig('astb_asfm_pl_figure.png')

plt.show()  # Optionally show the figure
