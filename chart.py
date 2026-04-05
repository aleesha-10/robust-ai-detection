import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

quality = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 95]
accuracy = [0.7495, 0.8183, 0.8635, 0.8650, 0.9050, 0.9350, 0.9383, 0.9223, 0.9283, 0.9367, 0.9190]

original = [10, 30, 50, 75, 95]
original_acc = [0.7495, 0.8635, 0.9050, 0.9223, 0.9190]

extended = [20, 40, 60, 70, 80, 90]
extended_acc = [0.8183, 0.8650, 0.9350, 0.9383, 0.9283, 0.9367]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(quality, accuracy, color='#185FA5', linewidth=2.5, zorder=2)
ax.fill_between(quality, accuracy, 0.70, alpha=0.08, color='#185FA5')

ax.scatter(original, original_acc, color='#185FA5', s=80, zorder=5, label='Original 5 levels')
ax.scatter(extended, extended_acc, color='#B5D4F4', s=70, zorder=5, edgecolors='#185FA5', linewidth=1.2, label='Extended levels (this study)')

ax.axvspan(40, 50, alpha=0.08, color='orange', label='Inflection zone (q=40–50)')

ax.set_xlabel('JPEG Quality (%)', fontsize=13, color='#555')
ax.set_ylabel('Accuracy', fontsize=13, color='#555')
ax.set_title('Detection Accuracy under JPEG Compression Shift', fontsize=14, fontweight='500', pad=15)
ax.set_ylim(0.70, 0.98)
ax.set_xlim(5, 100)
ax.grid(True, alpha=0.15)
ax.legend(fontsize=11)

for spine in ax.spines.values():
    spine.set_edgecolor('#ddd')

plt.tight_layout()
plt.savefig('jpeg_compression_chart.png', dpi=300, bbox_inches='tight')
print('Saved: jpeg_compression_chart.png')