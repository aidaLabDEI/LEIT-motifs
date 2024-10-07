from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def alpha(a, b, c, d):
    return a + b + c + d

if __name__ == "__main__":
    couplings = [[1,54,6,3], [6,7,8,4], [8,7,6,4], [3,6,7,8], [3,4,5,3], [1,1,1,1]]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(alpha, *couplings[0]) for i,j in itertools.product(range(4),range(8))]
        for future in as_completed(futures):
            if future.result() == 64:
                executor.shutdown(wait=False, cancel_futures=True)
                break
                print("Found it!")
            print(future.result())
    print("Done!")