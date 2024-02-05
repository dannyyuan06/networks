count = 0

total = 60

# Loop through all possible values of a, b, and c
for a in range(total//5 + 1):  # 0 to 20 (inclusive)
    for b in range(total//10 + 1):  # 0 to 10 (inclusive)
        for c in range(total//15 + 1):  # 0 to 5 (inclusive)
            if 5*a + 10*b + 15*c == total:
                count += 1
                print(f"Solution {count}: {a} 5p coins, {b} 10p coins, {c} 15p coins")

print(f"Total number of ways: {count}")