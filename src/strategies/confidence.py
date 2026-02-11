"""
This module calculates the confidence level of a trade based on various factors such as bias, liquidity, displacement, micro BOS, and location. The confidence level is categorized into HIGH, MEDIUM, LOW, or NO_TRADE based on the total score calculated from these factors.
"""


def calculate_confidence(
    bias: str,
    liquidity_ok: bool,
    displacement_ok: bool,
    micro_bos_ok: bool,
    good_location: bool,
) -> dict:

    score = 0
    if liquidity_ok:
        score += 1
    if displacement_ok:
        score += 1
    if micro_bos_ok:
        score += 1
    if good_location:
        score += 1

    if score >= 4:
        level = "HIGH"
    elif score == 3:
        level = "MEDIUM"
    elif score == 2:
        level = "LOW"
    else:
        level = "NO_TRADE"

    return {"score": score, "level": level}
