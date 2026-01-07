import numpy as np

#ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [4,1]

L_BOUND = -5
U_BOUND = 5

def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykĹad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)
     
#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )
    
class DlNet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_out = 0
        
        self.HIDDEN_L_SIZE = 9
        self.LR = 0.003
        
        # Wagi warstwy ukrytej (Input -> Hidden)
        # Wymiar: 1 wejście x HIDDEN_L_SIZE neuronów
        self.W1 = np.random.randn(1, self.HIDDEN_L_SIZE)
        self.b1 = np.zeros((1, self.HIDDEN_L_SIZE))
        # Wagi warstwy wyjściowej (Hidden -> Output)
        # Wymiar: HIDDEN_L_SIZE wejść x 1 wyjście
        self.W2 = np.random.randn(self.HIDDEN_L_SIZE, 1)
        self.b2 = np.zeros((1, 1))
        # Zmienne pomocnicze do przechowywania stanów pośrednich 
        self.hidden_input = None
        self.hidden_output = None

    def forward(self, x):  
        # Reshape x na (1,1) aby zgadzały się wymiary macierzy
        x_in = x.reshape(1, 1)
        
        # Obliczenie wejścia do warstwy ukrytej: z = x * W1 + b1
        self.hidden_input = np.dot(x_in, self.W1) + self.b1
        
        # Funkcja aktywacji warstwy ukrytej
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Warstwa wyjściowa  y = a * W2 + b2
        self.y_out = np.dot(self.hidden_output, self.W2) + self.b2
        
        return self.y_out[0][0] # Zwracamy skalar      
        
    def predict(self, x):    
        # Obliczamy wynik dla każdego x w tablicy wejściowej
        y_pred = []
        for val in x:
            res = self.forward(np.array([val]))
            y_pred.append(res)
        return np.array(y_pred)
        
    def backward(self, x, y): 
        # Przygotowanie danych (reshape dla macierzy)
        x_in = x.reshape(1, 1)
        y_target = np.array([[y]])
        
        #błąd na wyjściu
        diff = d_nloss(self.y_out, y_target)
        
        # Gradienty dla warstwy wyjściowej
        # dW2 = wyjście_ukrytej * błąd
        grad_W2 = np.dot(self.hidden_output.T, diff)
        grad_b2 = diff
        
        # błąd warstwy ukrytej 
        # błąd_ukrytej = błąd_wyjścia * W2_transponowane
        error_hidden = np.dot(diff, self.W2.T)
        
        # delta_hidden = błąd_ukrytej * pochodna_sigmoid(wejście_do_ukrytej)
        delta_hidden = error_hidden * d_sigmoid(self.hidden_input)
        
        # Gradienty dla warstwy ukrytej
        grad_W1 = np.dot(x_in.T, delta_hidden)
        grad_b1 = delta_hidden
        
        # Aktualizacja wag 
        self.W2 -= self.LR * grad_W2
        self.b2 -= self.LR * grad_b2
        self.W1 -= self.LR * grad_W1
        self.b1 -= self.LR * grad_b1
        
    def train(self, x_set, y_set, iters):    
        for i in range(0, iters):
            # Losowanie próbki
            idx = np.random.randint(0, len(x_set))
            x_sample = x_set[idx]
            y_sample = y_set[idx]
            self.forward(np.array([x_sample]))
            self.backward(np.array([x_sample]), y_sample)
          
nn = DlNet(x,y)
nn.train(x, y, 15000)
yh = nn.predict(x) #ToDo tu umiesciÄ wyniki (y) z sieci

# Obliczanie błędów MSE i MAE
mse = np.mean((y - yh) ** 2)
mae = np.mean(np.abs(y - yh))
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x, y, 'r',  label='Funkcja celu J(x)')
plt.plot(x, yh, 'b',  label='Aproksymacja sieci')
plt.legend(loc='upper right')
plt.title(f"Liczba neuronów ukrytych: {nn.HIDDEN_L_SIZE}", fontsize=10)

textstr = '\n'.join((
    f'MSE: {mse:.4f}',
    f'MAE: {mae:.4f}'))
props = dict(alpha=0.5)
ax.text(0.05, 0.01, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.grid(True)
plt.show()