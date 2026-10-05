# Labeling policy

Four classes: no-crack, minor, moderate, severe. No-crack comes from an explicit negative label or zero mask ratio. For cracked masks, calculate ratio = white mask pixels / total pixels and compute minor/moderate/severe cutoffs from training data only. These are crack-coverage proxy labels, not structural-risk labels.
