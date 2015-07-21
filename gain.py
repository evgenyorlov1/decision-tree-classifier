__author__ = 'pc'
def gain(data, attr=0, target_attr=0):
    val_freq = {}
    subset_entropy = 0.0
     for record in xrange(int(data)):
        if (val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0
    val_freq=[]
    for val in val_freq:
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)
    return (entropy(data, target_attr) - subset_entropy)
