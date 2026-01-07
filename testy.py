import numpy as np
from perceptron import DlNet, p, sigmoid, d_sigmoid

# --- KONFIGURACJA DANYCH ---
L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

# Generujemy dane raz, aby każdy test był na tym samym zbiorze
np.random.seed(1)
x_data = np.linspace(L_BOUND, U_BOUND, 100)
y_data = q(x_data)
# Resetujemy seed, aby wagi w sieci losowały się losowo przy każdym uruchomieniu pętli
np.random.seed()

# --- FUNKCJA POMOCNICZA DO STATYSTYK ---
def run_experiment_series(param_name, param_values, fixed_params):
    """
    Wykonuje serię eksperymentów (25 powtórzeń) dla zmiennych wartości parametru.
    """
    print(f"\n\n=== BADANIE WPŁYWU PARAMETRU: {param_name.upper()} ===")
    print(f"Parametry stałe: {fixed_params}")
    print("-" * 70)
    # Nagłówek tabeli zgodnie z wymogami
    print(f"{'Parametr':<12} | {'min':<10} | {'śr':<10} | {'std':<10} | {'max':<10}")
    print("-" * 70)

    for val in param_values:
        mse_list = []

        # Ustawiamy parametry dla tej konkretnej serii
        current_iters = val if param_name == "Iteracje" else fixed_params["Iteracje"]
        current_hidden = val if param_name == "Neurony" else fixed_params["Neurony"]
        current_lr = fixed_params["LR"]

        # Wymagane 25 uruchomień
        for _ in range(25):
            # 1. Tworzymy nową, czystą sieć
            nn = DlNet(x_data, y_data)

            # 2. Nadpisujemy parametry (wstrzykujemy konfigurację)
            nn.HIDDEN_L_SIZE = current_hidden
            nn.LR = current_lr

            # 3. Trenujemy
            # UWAGA: Jeśli w klasie kolegi metoda train nie ma parametru iters,
            # trzeba go dodać lub zmienić w kodzie klasy.
            # Zakładam, że train wygląda tak: def train(self, x, y, iters):
            nn.train(x_data, y_data, current_iters)

            # 4. Obliczamy błąd MSE
            yh = nn.predict(x_data)
            mse = np.mean((y_data - yh.flatten())**2)
            mse_list.append(mse)

        # Statystyki
        v_min = np.min(mse_list)
        v_mean = np.mean(mse_list)
        v_std = np.std(mse_list)
        v_max = np.max(mse_list)

        f_val = str(val)
        f_min = f"{v_min:.5f}".replace('.', ',')
        f_mean = f"{v_mean:.5f}".replace('.', ',')
        f_std = f"{v_std:.5f}".replace('.', ',')
        f_max = f"{v_max:.5f}".replace('.', ',')

        print(f"{param_name}={f_val:<5} | {f_min:<10} | {f_mean:<10} | {f_std:<10} | {f_max:<10}")


if __name__ == "__main__":

    # TEST 1: Wpływ liczby iteracji
    # Badamy: 15k (mało), 100k (średnio), 300k (dużo)
    # Stałe: Neurony=9 (bo wyszło Ci najlepiej), LR=0.01
    run_experiment_series(
        param_name="Iteracje",
        param_values=[15000, 100000, 300000],
        fixed_params={"Neurony": 9, "LR": 0.01}
    )

    run_experiment_series(
        param_name="Neurony",
        param_values=[3, 9, 20, 50],
        fixed_params={"Iteracje": 100000, "LR": 0.01}
    )
