import matplotlib.pyplot as plt

# Define two points (x1, y1) and (x2, y2)
x1, y1 = 2, 3
x2, y2 = 6, 8

# Plot the line between the two points
plt.plot([x1, x2], [y1, y2], marker='o', label='Line through two points')

# Extend the line beyond the given points
plt.plot([x1, x2, x2 + 1], [y1, y2, y2 + (y2 - y1) / (x2 - x1)], linestyle='dashed', label='Extended line')

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line through Two Points')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()