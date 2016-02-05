from pprint import pprint


synsets = [l.strip() for l in open('synset.txt').readlines()]
synset_ids = [s.split()[0] for s in synsets]
wanted = [l.strip() for l in open('wnids.txt').readlines() if l.strip() is not '' and not l.startswith("#")]

idx = [synset_ids.index(w) for w in wanted if w in synset_ids]


print(idx)
