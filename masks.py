def masks(mask, value, a):
	a.masked_fill(mask=mask, value=value)
	# mask[i][j] = 1(True), a[i][j] = value