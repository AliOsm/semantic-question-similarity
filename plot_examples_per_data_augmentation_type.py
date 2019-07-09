import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})

augmentation_type = [
	'Original',
	'Original and Transitive',
	'Original,\nTransitive and\nSymmetric',
	'Original,\nTransitive,\nSymmetric, and \nReflexive'
]

examples = [11997, 17487, 34974, 45514]
positive = [5397, 6271, 12542, 23082]
negative = [6600, 11216, 22432, 22432]

df = pd.DataFrame({
		'Total Examples': examples,
		'Positive Examples': positive,
		'Negative Examples': negative
	},
	index=augmentation_type
)

ax = df.plot.bar(rot=0)
# ax.set_title('Examples Number per Data Augmentation Type')
ax.set_xlabel('Augmentation Type')
ax.set_ylabel('Examples Number')
ax.autoscale_view()

plt.show()
