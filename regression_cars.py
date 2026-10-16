"""Fit and compare linear and quadratic regressions on R's cars dataset."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def main():
    folder = Path(__file__).resolve().parent
    df = pd.read_csv(folder / "cars.csv")
    x = df[["speed"]]
    y = df["dist"]
    if len(df) != 50 or x.isna().any().any() or y.isna().any():
        raise ValueError("cars.csv should contain 50 complete observations")

    linear = LinearRegression().fit(x, y)
    quadratic_features = PolynomialFeatures(degree=2, include_bias=False)
    x_quad = quadratic_features.fit_transform(x)
    quadratic = LinearRegression().fit(x_quad, y)

    linear_r2 = r2_score(y, linear.predict(x))
    quadratic_r2 = r2_score(y, quadratic.predict(x_quad))
    print(f"자료 수: {len(df)}")
    print(f"선형회귀: dist = {linear.intercept_:.6f} + {linear.coef_[0]:.6f} × speed")
    print(f"선형회귀 R²: {linear_r2:.6f}")
    print(f"2차 회귀: dist = {quadratic.intercept_:.6f} + {quadratic.coef_[0]:.6f} × speed + {quadratic.coef_[1]:.6f} × speed²")
    print(f"2차 회귀 R²: {quadratic_r2:.6f}")
    print("평가는 동일한 50개 학습 자료를 사용했으며, 미래 자료 예측 성능을 뜻하지 않습니다.")

    grid = pd.DataFrame({"speed": np.linspace(x.speed.min(), x.speed.max(), 300)})
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x.speed, y, color="#334155", alpha=.75, label="Observed (50 cars)")
    ax.plot(grid.speed, linear.predict(grid), color="#2563eb", linewidth=2,
            label=f"Linear (R²={linear_r2:.3f})")
    ax.plot(grid.speed, quadratic.predict(quadratic_features.transform(grid)),
            color="#dc2626", linewidth=2, label=f"Quadratic (R²={quadratic_r2:.3f})")
    ax.set(xlabel="Speed (mph)", ylabel="Stopping distance (ft)",
           title="R cars: speed versus stopping distance")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    output = folder / "cars_regression.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"그래프 저장: {output.name}")


if __name__ == "__main__":
    main()
