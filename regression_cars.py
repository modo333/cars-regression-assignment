"""Compare linear, quadratic, and neural-network regression on R's cars data."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from neural_network import NeuralRegressor


def fit_models(x, y):
    linear = LinearRegression().fit(x, y)
    features = PolynomialFeatures(degree=2, include_bias=False)
    quadratic = LinearRegression().fit(features.fit_transform(x), y)
    neural = NeuralRegressor().fit(x.speed.to_numpy(), y.to_numpy())
    return linear, quadratic, features, neural


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=20, help="예측할 속도 (mph)")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    df = pd.read_csv(folder / "cars.csv")
    x = df[["speed"]]
    y = df["dist"]
    if len(df) != 50 or x.isna().any().any() or y.isna().any():
        raise ValueError("cars.csv should contain 50 complete observations")

    # Original rows 5, 10, ..., 50 are held out; the other 40 are training data.
    test_mask = np.arange(len(df)) % 5 == 4
    x_train, y_train = x.loc[~test_mask], y.loc[~test_mask]
    x_test, y_test = x.loc[test_mask], y.loc[test_mask]
    t_linear, t_quad, t_features, t_neural = fit_models(x_train, y_train)
    print("[평가] 학습 40개 / 테스트 10개 (원본 순서의 매 5번째 행)")
    for name, pred in (("선형", t_linear.predict(x_test)),
                       ("이차", t_quad.predict(t_features.transform(x_test))),
                       ("인공신경망", t_neural.predict(x_test.speed.to_numpy()))):
        print(f"{name} 테스트 R²: {r2_score(y_test, pred):.6f}")

    linear, quadratic, quadratic_features, neural = fit_models(x, y)
    x_quad = quadratic_features.transform(x)

    linear_r2 = r2_score(y, linear.predict(x))
    quadratic_r2 = r2_score(y, quadratic.predict(x_quad))
    print(f"자료 수: {len(df)}")
    print(f"선형회귀: dist = {linear.intercept_:.6f} + {linear.coef_[0]:.6f} × speed")
    print(f"선형회귀 R²: {linear_r2:.6f}")
    print(f"2차 회귀: dist = {quadratic.intercept_:.6f} + {quadratic.coef_[0]:.6f} × speed + {quadratic.coef_[1]:.6f} × speed²")
    print(f"2차 회귀 R²: {quadratic_r2:.6f}")
    neural_r2 = r2_score(y, neural.predict(x.speed.to_numpy()))
    print(f"인공신경망 학습 R²: {neural_r2:.6f}")
    point = pd.DataFrame({"speed": [args.speed]})
    print(f"{args.speed:g} mph 예측 제동거리 (ft): 선형 {linear.predict(point)[0]:.2f}, "
          f"이차 {quadratic.predict(quadratic_features.transform(point))[0]:.2f}, "
          f"신경망 {neural.predict([args.speed])[0]:.2f}")
    print("1920년대 자료이므로 실제 운전의 안전거리 판단에 사용하지 마세요.")

    grid = pd.DataFrame({"speed": np.linspace(x.speed.min(), x.speed.max(), 300)})
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x.speed, y, color="#334155", alpha=.75, label="Observed (50 cars)")
    ax.plot(grid.speed, linear.predict(grid), color="#2563eb", linewidth=2,
            label=f"Linear (R²={linear_r2:.3f})")
    ax.plot(grid.speed, quadratic.predict(quadratic_features.transform(grid)),
            color="#dc2626", linewidth=2, label=f"Quadratic (R²={quadratic_r2:.3f})")
    ax.plot(grid.speed, neural.predict(grid.speed.to_numpy()), color="#059669", linewidth=2,
            label=f"Neural net (R²={neural_r2:.3f})")
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
