import random
import subprocess
from pathlib import Path

import pandas as pd


def generate_synthetic_training_data() -> Path:
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "data" / "processed" / "microsoft_training.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(2000):
        failure_mode = random.random()

        if failure_mode < 0.15:
            voltage = random.uniform(350, 410)
            rpm = random.uniform(4200, 5200)
            pressure_psi = random.uniform(150, 225)
            vibration_mm_s = random.uniform(9.0, 18.0)
            temperature_c = random.uniform(100, 130)
            current = random.uniform(75, 120)
            power_kw = random.uniform(45, 80)
            failure = 1
        elif failure_mode < 0.35:
            voltage = random.uniform(390, 470)
            rpm = random.uniform(3200, 4300)
            pressure_psi = random.uniform(120, 180)
            vibration_mm_s = random.uniform(7.0, 11.0)
            temperature_c = random.uniform(88, 105)
            current = random.uniform(60, 90)
            power_kw = random.uniform(30, 60)
            failure = 1
        elif failure_mode < 0.55:
            voltage = random.uniform(430, 500)
            rpm = random.uniform(2600, 3900)
            pressure_psi = random.uniform(105, 155)
            vibration_mm_s = random.uniform(5.0, 8.5)
            temperature_c = random.uniform(80, 98)
            current = random.uniform(50, 75)
            power_kw = random.uniform(22, 45)
            failure = 1
        else:
            voltage = random.uniform(450, 520)
            rpm = random.uniform(1800, 3000)
            pressure_psi = random.uniform(80, 120)
            vibration_mm_s = random.uniform(1.5, 4.5)
            temperature_c = random.uniform(55, 78)
            current = random.uniform(20, 45)
            power_kw = random.uniform(8, 24)
            failure = 0

        rows.append(
            {
                "voltage": round(voltage, 2),
                "rpm": round(rpm, 2),
                "pressure": round(pressure_psi, 2),
                "vibration": round(vibration_mm_s, 2),
                "temperature": round(temperature_c, 2),
                "current": round(current, 2),
                "power_kw": round(power_kw, 2),
                "age": random.randint(0, 18),
                "failure": failure,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(rows)} synthetic rows at {output_path}")
    return output_path


def main():
    generate_synthetic_training_data()
    subprocess.run(["python", "training.py"], check=True)
    print("Retraining complete.")


if __name__ == "__main__":
    main()
