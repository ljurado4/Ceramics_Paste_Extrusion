import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv(r"C:\Users\zzcro\Downloads\ForceTestData.csv")
df = pd.DataFrame(data)
df = df.sort_values(by='Time')

# Parameters for segmentation
segment_size = 10  # Adjust this value based on how much smoothing you want

# Average segments
df['Segment'] = (df.index // segment_size)  # Create a segment identifier
df_smoothed = df.groupby('Segment').agg({
    'Time': 'mean',           # Average the Time values
    'Force': 'mean'          # Average the Force values
}).reset_index(drop=True)

# Set plot limits
xmin, xmax = 0, 200
ymin, ymax = 2, 9
ax = plt.gca()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# Correcting force values (adjust if needed)
for x in df.index:
    if df.loc[x, "Force"] > 9:
        df.loc[x, "Force"] = 3
average = df['Force'].mean()

# Plot the smoothed 'Force' data
plt.plot(df_smoothed['Time'], df_smoothed['Force'], label='Averaged Force Data')  
plt.grid(visible=True)
plt.legend()  # Add legend for clarity
plt.show()
