import numpy as np
from perceptron import DlNet, p


L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

# Generujemy dane raz, aby każdy test był na tym samym zbiorze
np.random.seed(1)
x_data = np.linspace(L_BOUND, U_BOUND, 100)
y_data = q(x_data)

# Resetujemy seed, aby wagi w sieci losowały się losowo przy każdym uruchomieniu
np.random.seed()

def run_experiment_series(param_name, param_values, fixed_params):
    """
    Wykonuje serię eksperymentów (25 powtórzeń) dla zmiennych wartości parametru
    i wypisuje tabelę statystyczną.
    """
    print(f"\n\n=== BADANIE WPŁYWU PARAMETRU: {param_name.upper()} ===")
    print(f"Parametry stałe: {fixed_params}")
    print("-" * 70)
    print(f"{'Parametr':<12} | {'min':<10} | {'śr':<10} | {'std':<10} | {'max':<10}")
    print("-" * 70)

    for val in param_values:
        mse_list = []

        # Ustalanie parametrów: jeśli badamy dany parametr to bierzemy 'val', reszta stała
        current_iters = val if param_name == "Iteracje" else fixed_params["Iteracje"]
        current_hidden = val if param_name == "Neurony" else fixed_params["Neurony"]
        current_lr = val if param_name == "LR" else fixed_params["LR"]

        # Pętla statystyczna (25 uruchomień)
        for _ in range(25):
            nn = DlNet(x_data, y_data)
            nn.HIDDEN_L_SIZE = current_hidden
            nn.LR = current_lr

            nn.train(x_data, y_data, current_iters)

            yh = nn.predict(x_data)
            mse = np.mean((y_data - yh.flatten())**2)
            mse_list.append(mse)

        # Obliczenia statystyczne
        v_min = np.min(mse_list)
        v_mean = np.mean(mse_list)
        v_std = np.std(mse_list)
        v_max = np.max(mse_list)

        # Formatowanie z polskim przecinkiem
        f_val = str(val)
        f_min = f"{v_min:.5f}".replace('.', ',')
        f_mean = f"{v_mean:.5f}".replace('.', ',')
        f_std = f"{v_std:.5f}".replace('.', ',')
        f_max = f"{v_max:.5f}".replace('.', ',')

        print(f"{param_name}={f_val:<5} | {f_min:<10} | {f_mean:<10} | {f_std:<10} | {f_max:<10}")


if __name__ == "__main__":
    print("Rozpoczynam testy")

    # TEST 1: Wpływ Learning Rate (LR)
    run_experiment_series(
        param_name="LR",
        param_values=[0.001, 0.01, 0.05, 0.1],
        fixed_params={"Neurony": 9, "Iteracje": 50000}
    )

    # TEST 2: Wpływ liczby iteracji
    run_experiment_series(
        param_name="Iteracje",
        param_values=[15000, 100000, 300000],
        fixed_params={"Neurony": 9, "LR": 0.01}
    )

    # TEST 3: Wpływ liczby neuronów
    run_experiment_series(
        param_name="Neurony",
        param_values=[3, 9, 20, 50],
        fixed_params={"Iteracje": 100000, "LR": 0.01}
    )