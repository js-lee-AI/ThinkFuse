from thinkfuse import MockModelClient, ThinkFuse

QUESTION = (
    "Question: A train travels 60 km in the first hour and 90 km in the second "
    "hour. What is its average speed over the two hours?\nAnswer:"
)

PRIMARY_SCRIPT = (
    "<think>\n"
    "Let me work through this carefully. The train covers 60 km in the first hour "
    "and 90 km in the second hour, so the total distance is 60 + 90 = 150 km. "
    "The total time is 2 hours. Average speed is total distance divided by total "
    "time, which is 150 / 2 = 75 km/h. Let me double-check: the two distances add "
    "to 150 and dividing by 2 hours indeed gives 75 km/h.\n"
    "</think>\n\n"
    "The average speed over the two hours is 75 km/h."
)


def main():
    primary = MockModelClient("qwen3-4b", script=PRIMARY_SCRIPT)
    auxiliary = MockModelClient("ministral-3b-reasoning")

    engine = ThinkFuse(primary, auxiliary, preview_tokens=8, max_new_tokens=600)
    result = engine.run(QUESTION)

    print("Fused output:")
    print(result.text)
    print()
    print("Fusion stats:")
    print(f"  total segments  : {result.total_segments}")
    print(f"  fused segments  : {result.fused_segments}")
    print(f"  fusion ratio    : {result.fusion_ratio:.3f}")


if __name__ == "__main__":
    main()
