from nearpy.hashes import RandomDiscretizedProjections
import numpy as np
from nearpy import Engine

def has_collision(lst):
    # Sort the list for efficient collision checking
    lst.sort()

    # Iterate over the list to check for adjacent equal elements
    for i in range(len(lst) - 1):
        if lst[i] == lst[i + 1]:
            return True  # Collision found

    return False  # No collision found

def find_width_discr(subs: list, window: int, K: int) -> int:
  r = 1
  no_collisions = True
  total_dimensions = len(subs[0])
  num_digits = len(str(total_dimensions))
  dimension_numbers = [int(f"{i:0{num_digits}d}") for i in range(1, total_dimensions + 1)]

  hashed_lists = [[] for _ in range(len(subs[0]))]

  while no_collisions:
        rps = RandomDiscretizedProjections('rp', K, r)
        engine = Engine(window, lshashes=[rps])

        # Check if there is at least one collision
        for subsequence in subs:

            subsequence_znorm = (subsequence - np.mean(subsequence, axis=0)) / np.std(subsequence, axis=0)
            hashed = np.apply_along_axis(euclidean_hash, 1, subsequence_znorm, rps)

            for i,row in enumerate(hashed):
              hashed_lists[i].append(np.array2string(row))

        collisions = [has_collision(lst) for lst in hashed_lists]

        # If no collisions found, double r and continue
        if np.count_nonzero(collisions) == total_dimensions:
          no_collisions = False
        else:
          r *= 2

  return r