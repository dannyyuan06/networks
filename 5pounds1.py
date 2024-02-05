def count_ways_to_make_amount(n, coins):
    ways = [0] * (n + 1)
    ways[0] = 1  # There is one way to make £0

    for coin in coins:
        for amount in range(coin, n + 1):
            ways[amount] += ways[amount - coin]

    return ways[n]
  
  
n = 50  # £5 = 500p
coins = [5, 10, 20]
print(count_ways_to_make_amount(n, coins))