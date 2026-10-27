"""用于验证沙箱的最小循环示例。"""

total = 0
for number in [1, 2, 3]:
    total += number
    print(f"number={number}, total={total}")

print(f"result={total}")
