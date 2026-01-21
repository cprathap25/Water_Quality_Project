def calculate_risk(row):
    score = 0

    if row["pH"] < 6.5 or row["pH"] > 8.5:
        score += 1

    if row["dissolved_oxygen"] < 5:
        score += 2

    if row["bod"] > 3:
        score += 2

    if row["fecal_coliform"] > 100:
        score += 2

    if score <= 3:
        status = "Safe"
    elif score <= 6:
        status = "Moderate"
    else:
        status = "Polluted"

    return score, status
