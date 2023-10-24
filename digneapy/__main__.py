"""Main module."""

from .novelty_search import Archive, NoveltySearch
import numpy as np

if __name__ == "__main__":
    descriptors = np.random.rand(30, 10)
    # archive = Archive(descriptors)
    # print(archive)
    # print(repr(archive))
    # for d in archive:
    #     print(d)
    # print(bool(archive))

    # d = [100, 5, 2, 5, 0]
    # archive.append(d)
    # print(archive)
    # archive.extend(descriptors)
    # print(archive)
    ns = NoveltySearch(t_a=0.9, t_ss=0.5, k=3)
    sparseness = ns.sparseness(descriptors, True)
    sparseness = ns.sparseness_solution_set(descriptors)
    print(sparseness)
    print(ns)
